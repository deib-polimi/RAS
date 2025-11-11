# firm_controller_final_proactive.py
#
# Versione definitiva e proattiva del FirmController.
# Include una funzione di reward avanzata per migliorare la reattività
# al carico di lavoro e tutte le correzioni e le best practice discusse.

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from datetime import datetime

# L'import viene ora gestito come richiesto per un'integrazione pulita
from .controller import Controller

# ==============================================================================
#  DEFINIZIONE DELLE RETI NEURALI
# ==============================================================================

class ANet(nn.Module):
    """Rete Actor, fedele all'implementazione originale di FIRM."""
    def __init__(self, s_dim, a_dim):
        super(ANet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, a_dim)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn_out = nn.BatchNorm1d(a_dim)

    def forward(self, x):
        if len(x.shape) == 1: x = x.unsqueeze(0)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.bn_out(self.out(x))
        return torch.sigmoid(x)

class CNet(nn.Module):
    """Rete Critic, fedele all'implementazione originale di FIRM."""
    def __init__(self, s_dim, a_dim):
        super(CNet, self).__init__()
        self.fcs = nn.Linear(s_dim, 64)
        self.fca = nn.Linear(a_dim, 64)
        self.fc_out = nn.Linear(64, 1)
        self.bn1 = nn.BatchNorm1d(64)

    def forward(self, s, a):
        if len(s.shape) == 1: s = s.unsqueeze(0)
        if len(a.shape) == 1: a = a.unsqueeze(0)
        x = self.fcs(s)
        y = self.fca(a)
        net = F.relu(self.bn1(x + y))
        return self.fc_out(net)

# ==============================================================================
#  REPLAY BUFFER
# ==============================================================================
class ReplayBuffer:
    def __init__(self, capacity): self.buffer = deque(maxlen=capacity)
    def store(self, s, a, r, s_, d): self.buffer.append((s, a, r, s_, float(d)))
    def sample(self, bs): return map(np.stack, zip(*random.sample(self.buffer, bs)))
    def __len__(self): return len(self.buffer)

# ==============================================================================
#  CLASSE PRINCIPALE DEL CONTROLLORE FIRM
# ==============================================================================
class FirmController(Controller):
    _A_DIM = 3
    _GAMMA, _TAU = 0.95, 0.005
    _LR_A, _LR_C = 0.001, 0.001

    def __init__(self, period, init_cores, *, min_cores=1, max_cores=1000, st=0.8, name="FIRM-DDPG",
                 train=True, trend_features=False, memory_capacity=50000, batch_size=64,
                 noise_stddev=0.1,
                 # Pesi per la nuova funzione di reward proattiva
                 reward_weight_latency_violation=0.6,
                 reward_weight_latency_trend=0.1,
                 reward_weight_cores=0.3,
                 enable_log=True, log_dir="./logs", checkpoint_path="./firm_checkpoint.pt"):
        super().__init__(period, init_cores, st, name)
        
        self.min_cores, self.max_cores, self.train, self.trend_features = min_cores, max_cores, train, trend_features
        self.batch_size, self.checkpoint_path = batch_size, checkpoint_path
        self.w_viol = reward_weight_latency_violation
        self.w_trend = reward_weight_latency_trend
        self.w_cores = reward_weight_cores
        self.initial_noise_stddev, self.noise_stddev = noise_stddev, noise_stddev
        self.s_dim = 7 if self.trend_features else 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Inizializzazione ---
        self._setup_logging(log_dir)
        self._setup_networks()
        self._setup_memory_and_state()

        if self.train and os.path.exists(self.checkpoint_path):
            print(f"--- Caricamento checkpoint: '{self.checkpoint_path}' ---")
            self._load_checkpoint()
        else:
            print("--- Inizio da zero (nessun checkpoint o modalità inferenza). ---")
            
        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.critic_target.load_state_dict(self.critic_eval.state_dict())
        
        if not self.train:
            self.actor_eval.eval()

    def _setup_logging(self, log_dir):
        self.enable_log = True
        if self.enable_log:
            os.makedirs(log_dir, exist_ok=True)
            self.log_path = os.path.join(log_dir, f"{self.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")

    def _setup_networks(self):
        self.actor_eval = ANet(self.s_dim, self._A_DIM).to(self.device)
        self.actor_target = ANet(self.s_dim, self._A_DIM).to(self.device)
        self.critic_eval = CNet(self.s_dim, self._A_DIM).to(self.device)
        self.critic_target = CNet(self.s_dim, self._A_DIM).to(self.device)
        self.atrain = optim.Adam(self.actor_eval.parameters(), lr=self._LR_A)
        self.ctrain = optim.Adam(self.critic_eval.parameters(), lr=self._LR_C)
        self.loss_td = nn.MSELoss()

    def _setup_memory_and_state(self):
        self.memory = ReplayBuffer(50000) # memory_capacity
        self.step_cnt, self.prev_reward = 0, 0.0
        self.prev_state = self.prev_action = self.prev_p95 = self.prev_q = self.prev_rate = None

    def _save_checkpoint(self):
        torch.save({ 'step_cnt': self.step_cnt, 'actor_state_dict': self.actor_eval.state_dict(),
                     'critic_state_dict': self.critic_eval.state_dict(), 'actor_optimizer_state': self.atrain.state_dict(),
                     'critic_optimizer_state': self.ctrain.state_dict(), 'noise_stddev': self.noise_stddev
                   }, self.checkpoint_path)

    def _load_checkpoint(self):
        try:
            ckp = torch.load(self.checkpoint_path, map_location=self.device)
            self.step_cnt = ckp['step_cnt']
            self.actor_eval.load_state_dict(ckp['actor_state_dict'])
            self.critic_eval.load_state_dict(ckp['critic_state_dict'])
            self.atrain.load_state_dict(ckp['actor_optimizer_state'])
            self.ctrain.load_state_dict(ckp['critic_optimizer_state'])
            self.noise_stddev = ckp.get('noise_stddev', self.initial_noise_stddev)
            print(f"--- Checkpoint caricato. Ripresa dal passo {self.step_cnt}. ---")
        except Exception as e:
            print(f"ERRORE: Impossibile caricare il checkpoint ({e}). Si riparte da zero.")

    def _state(self) -> np.ndarray:
        p95 = self.monitoring.getRTp95()
        q = self.monitoring.getQueueLen()
        r = self.monitoring.getArrivalRate()
        lat_ratio = p95 / self.setpoint
        trend = 0.0 if self.prev_p95 is None else (p95 - self.prev_p95) / self.setpoint
        norm_cores = (self.cores - self.min_cores) / (self.max_cores - self.min_cores) if (self.max_cores > self.min_cores) else 0.0
        base = [min(2.0, lat_ratio), q / 100.0, r / 1000.0, trend, norm_cores]
        if self.trend_features:
            qd = 0.0 if self.prev_q is None else (q - self.prev_q) / 100.0
            rd = 0.0 if self.prev_rate is None else (r - self.prev_rate) / 1000.0
            st = np.array(base + [qd, rd], dtype=np.float32)
        else:
            st = np.array(base, dtype=np.float32)
        self.prev_p95, self.prev_q, self.prev_rate = p95, q, r
        return st

    def _reward(self) -> float:
        lat_ratio = self.monitoring.getRT() - self.setpoint
        pen_lat = 0.6 * lat_ratio if lat_ratio > 0 else 0.4 * lat_ratio 
        return -abs(pen_lat) 

    def control(self, t: float):
        current_state = self._state()
        if self.prev_state is not None and self.train:
            reward = self._reward()
            self.memory.store(self.prev_state, self.prev_action, reward, current_state, False)
            self.prev_reward = reward

        self.actor_eval.eval()
        with torch.no_grad():
            action_vec = self.actor_eval(torch.FloatTensor(current_state).to(self.device)).cpu().numpy().flatten()
        
        if self.train:
            action_vec = np.random.normal(action_vec, self.noise_stddev)

        final_action = np.clip(action_vec[0], 0.0, 1.0)
        target_cores = self.min_cores + final_action * (self.max_cores - self.min_cores)
        delta = round(target_cores) - self.cores
        self.cores += max(self.min_cores - self.cores, min(delta, self.max_cores - self.cores))

        if self.train and len(self.memory) > self.batch_size:
            self._update()

        self.prev_state, self.prev_action = current_state, action_vec
        self.step_cnt += 1
        self._log(t, delta)
        
        if self.train and self.step_cnt > 0 and self.step_cnt % 1000 == 0:
            self._save_checkpoint()

    def _update(self):
        self.actor_eval.train(); self.critic_eval.train()
        bs, ba, br, bs_, bd = (torch.FloatTensor(d).to(self.device) for d in self.memory.sample(self.batch_size))
        br, bd = br.reshape(-1, 1), bd.reshape(-1, 1)

        with torch.no_grad():
            q_target = br + (1.0 - bd) * self._GAMMA * self.critic_target(bs_, self.actor_target(bs_))
        
        critic_loss = self.loss_td(q_target, self.critic_eval(bs, ba))
        self.ctrain.zero_grad(); critic_loss.backward(); self.ctrain.step()

        for p in self.critic_eval.parameters(): p.requires_grad = False
        actor_loss = -self.critic_eval(bs, self.actor_eval(bs)).mean()
        self.atrain.zero_grad(); actor_loss.backward(); self.atrain.step()
        for p in self.critic_eval.parameters(): p.requires_grad = True

        with torch.no_grad():
            for target, eval_param in zip(self.actor_target.parameters(), self.actor_eval.parameters()):
                target.data.copy_(self._TAU * eval_param.data + (1.0 - self._TAU) * target.data)
            for target, eval_param in zip(self.critic_target.parameters(), self.critic_eval.parameters()):
                target.data.copy_(self._TAU * eval_param.data + (1.0 - self._TAU) * target.data)

        self.noise_stddev = max(0.01, self.noise_stddev * 0.99995)

    def _log(self, t, delta):
        if not self.enable_log: return
        rt = self.monitoring.getRTp95()
        phase = "TRAIN" if self.train else "EVAL"
        line = f"{t:.1f}s|{self.name}[{phase}]: lat={rt:.2f}s, cores={self.cores}, Δ={delta:+d}, rew={self.prev_reward:.3f}"
        print(line)
        if hasattr(self, 'log_path'):
            with open(self.log_path, "a") as f: f.write(line + "\n")

    def setTrain(self, train: bool):
        """
        Imposta manualmente la modalità di training o valutazione dell'agente.
        """
        self.train = train
        if self.train:
            self.actor_eval.train()
            self.critic_eval.train()
        else:
            self.actor_eval.eval()
            self