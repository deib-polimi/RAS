import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from .controller import Controller

# --------------------------------------------------
#  NETWORK DEFINITION
# --------------------------------------------------
class _ActorCritic(nn.Module):
    """Two-layer MLP actor‑critic."""
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
        )
        self.policy_head = nn.Linear(128, act_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.policy_head(h), self.value_head(h)


class _RolloutBuf:
    def __init__(self, cap: int):
        self.cap = cap; self.reset()
    def store(self, *tr):
        self.buf.append(tr)
    def sample(self):
        s,a,lp,r,d,v = zip(*self.buf); return map(np.array,(s,a,lp,r,d,v))
    def reset(self):
        self.buf = []
    def __len__(self): return len(self.buf)


# --------------------------------------------------
#  PPO CONTROLLER
# --------------------------------------------------
class PPOController(Controller):
    """Burst‑aware PPO autoscaler (supports guard/hybrid/none)."""

    _GAMMA = 0.99; _LAMBDA = 0.95; _CLIP = 0.2
    _LR = 1e-4; _EPOCHS = 4; _BATCH = 64; _ROLLOUT = 512; _ENT_COEF = 0.01

    def __init__(self, period, init_cores, *,
                 min_cores=1, max_cores=1000, st=0.8, name=None,
                 train=True,
                 burst_mode="none", burst_threshold_q=20, burst_threshold_r=30,
                 burst_extra=4, trend_features=False,
                 enable_log=True, log_dir="./logs"):
        super().__init__(period, init_cores, st, name=name)
        self.min_cores, self.max_cores = min_cores, max_cores
        self.train = train
        self.burst_mode = burst_mode
        self.burst_threshold_q = burst_threshold_q
        self.burst_threshold_r = burst_threshold_r
        self.burst_extra = burst_extra
        self.trend_features = trend_features
        self.enable_log = enable_log

        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_path = os.path.join(log_dir, f"ppo-{burst_mode}-{ts}.log")
        self.model_path = os.path.join("./controllers", f"ppocontroller-{burst_mode}.pt")

        self.actions = np.arange(-2,3,1)
        obs_dim = 5 + (2 if trend_features else 0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ac = _ActorCritic(obs_dim, len(self.actions)).to(self.device)
        self.opt = optim.Adam(self.ac.parameters(), lr=self._LR)
        self.buf = _RolloutBuf(self._ROLLOUT)
        self.step_cnt = 0
        if os.path.exists(self.model_path): self._load()
        if not self.train: self.ac.eval()

        self.prev_state=self.prev_act=self.prev_logp=self.prev_val=None
        self.prev_p95=self.prev_q=self.prev_rate=None
        self.prev_reward=0.0

    # ------------------- main -------------------
    def control(self, t):
        # burst guard
        if self.burst_mode in ("guard","hybrid") and self._burst():
            self.cores = min(self.cores + self.burst_extra, self.max_cores)
            if self.burst_mode=="guard":
                self._log(t,None,self.burst_extra,guard=True)
                return
        state = self._state()
        logits,val = self.ac(torch.tensor(state,dtype=torch.float32,device=self.device))
        dist=torch.distributions.Categorical(logits=logits)
        a_idx=int(dist.sample().item()); logp=float(dist.log_prob(torch.tensor(a_idx)))
        val=float(val)
        if self.train and self.prev_state is not None:
            self.buf.store(self.prev_state,self.prev_act,self.prev_logp,self._reward(),False,self.prev_val)
            if len(self.buf)>=self._ROLLOUT: self._update(); self.buf.reset()
        delta=int(self.actions[a_idx]); delta=max(self.min_cores-self.cores,min(delta,self.max_cores-self.cores))
        self.cores+=delta
        self.prev_state,self.prev_act,self.prev_logp,self.prev_val=state,a_idx,logp,val
        self.step_cnt+=1
        self._log(t,state,delta)

    # ------------------- helpers -------------------
    def _state(self):
        p95=self.monitoring.getRTp95(); lat_ratio=min(2.0,p95/self.setpoint)
        q=self.monitoring.getQueueLen(); r=self.monitoring.getArrivalRate()
        trend=0.0 if self.prev_p95 is None else (p95-self.prev_p95)/self.setpoint
        self.prev_p95=p95
        base=[lat_ratio,q/100.0,r/1000.0,trend,(self.cores-self.min_cores)/(self.max_cores-self.min_cores)]
        if self.trend_features:
            qd=0.0 if self.prev_q is None else (q-self.prev_q)/100.0
            rd=0.0 if self.prev_rate is None else (r-self.prev_rate)/1000.0
            st=np.array(base+[qd,rd],dtype=np.float32)
        else: st=np.array(base,dtype=np.float32)
        self.prev_q,self.prev_rate=q,r
        return st

    def _reward(self):
        lat_ratio = self.monitoring.getRT() - self.setpoint
        pen_lat = 0.6 * lat_ratio if lat_ratio > 0 else 0.4 * lat_ratio 
        self.prev_reward = -abs(pen_lat) 
        return -abs(pen_lat) 

    def _burst(self):
        return (self.prev_q is not None and (self.monitoring.getQueueLen()-self.prev_q)>self.burst_threshold_q) or \
               (self.prev_rate is not None and (self.monitoring.getArrivalRate()-self.prev_rate)>self.burst_threshold_r)

    def _update(self):
        s,a,lp_old,r,_,v=self.buf.sample(); adv=[]; gae=0; v=np.append(v,v[-1])
        for i in reversed(range(len(r))):
            delta=r[i]+self._GAMMA*v[i+1]-v[i]; gae=delta+self._GAMMA*self._LAMBDA*gae; adv.insert(0,gae)
        adv=np.array(adv,dtype=np.float32); ret=adv+np.array(v[:-1],dtype=np.float32)
        s_t=torch.tensor(s,dtype=torch.float32,device=self.device); a_t=torch.tensor(a,dtype=torch.int64,device=self.device)
        lp_old_t=torch.tensor(lp_old,dtype=torch.float32,device=self.device)
        adv_t=torch.tensor(adv,dtype=torch.float32,device=self.device)
        ret_t=torch.tensor(ret,dtype=torch.float32,device=self.device)
        adv_t=(adv_t-adv_t.mean())/(adv_t.std()+1e-6)
        for _ in range(self._EPOCHS):
            idx=np.random.permutation(len(s))
            for st in range(0,len(s),self._BATCH):
                b=idx[st:st+self._BATCH]; logits,vp=self.ac(s_t[b]); dist=torch.distributions.Categorical(logits=logits)
                lp=dist.log_prob(a_t[b]); ratio=torch.exp(lp-lp_old_t[b])
                s1=ratio*adv_t[b]; s2=torch.clamp(ratio,1-self._CLIP,1+self._CLIP)*adv_t[b]
                pol_loss=-torch.min(s1,s2).mean(); v_loss=nn.functional.mse_loss(vp.squeeze(-1),ret_t[b])
                loss=pol_loss+0.5*v_loss-self._ENT_COEF*dist.entropy().mean()
                self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.ac.parameters(),0.5); self.opt.step()
        if self.train: torch.save({'net':self.ac.state_dict(),'steps':self.step_cnt},self.model_path)

    def _load(self):
        ck=torch.load(self.model_path,map_location=self.device); self.ac.load_state_dict(ck['net']); self.step_cnt=ck['steps']

    def _log(self,t,state,delta,guard=False):
        if not self.enable_log: return
        if state is None:
            line=f"{t:.1f} s | GUARD +{delta} cores"
        else:
            rt=self.monitoring.getRTp95(); line=(f"{t:.1f}s lat={rt:.2f} cores={self.cores} Δ={delta:+d} rew={self.prev_reward:.2f}")
        print(line)
        with open(self.log_path,"a") as f: f.write(line+"\n")

    def setTrain(self, train):
        self.train = train

import random
import math

class FuzzyPPOController(PPOController):
    std_ratio = 0.2
    mean_ratio = 0
    def control(self, t):
        super().control(t)
        mean_noise = self.mean_ratio * self.cores           
        std_noise = self.std_ratio * abs(self.cores)
        noise = random.gauss(mean_noise, std_noise)      
        print(noise)  
        self.cores = math.ceil(self.cores + noise)
        self.cores = max(self.min_cores, min(self.cores, self.max_cores))
        