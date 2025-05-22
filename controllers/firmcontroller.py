import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from .controller import Controller

# === Actor (azione continua ∈ [−1, 1]) ===
class ANet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.fc1 = nn.Linear(s_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, a_dim)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.out.weight.data.uniform_(-0.003, 0.003)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return torch.tanh(self.out(x))  # range [-1, 1]

# === Critic (valuta coppia stato–azione) ===
class CNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.fcs = nn.Linear(s_dim, 64)
        self.fca = nn.Linear(a_dim, 64)
        self.fc = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)
        self.bn = nn.BatchNorm1d(64)

    def forward(self, s, a):
        x = F.relu(self.fcs(s) + self.fca(a))
        x = F.relu(self.bn(self.fc(x)))
        return self.out(x)

# === DDPG buffer minimale (on-policy) ===
class ReplayBuf:
    def __init__(self, cap=10000):
        self.buf = []
        self.cap = cap
    def store(self, s, a, r, s_):
        if len(self.buf) >= self.cap:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_))
    def sample(self, batch):
        idx = np.random.choice(len(self.buf), batch)
        s, a, r, s_ = zip(*[self.buf[i] for i in idx])
        return map(lambda x: torch.tensor(x, dtype=torch.float32), (s, a, r, s_))
    def __len__(self):
        return len(self.buf)

# === Controller integrato ===
class FIRM_DDPG_Controller(Controller):
    def __init__(self, period, init_cores,
                 min_cores=1, max_cores=1000, st=0.8, name="FIRM-DDPG",
                 train=True, log_dir="./logs"):
        super().__init__(period, init_cores, st, name=name)
        self.train = train
        self.min_cores, self.max_cores = min_cores, max_cores
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_path = f"{log_dir}/firm_ddpg-{int(time.time())}.csv" if train else None
        os.makedirs(log_dir, exist_ok=True)

        # === rete ===
        self.s_dim = 5
        self.a_dim = 1
        self.actor = ANet(self.s_dim, self.a_dim).to(self.device)
        self.critic = CNet(self.s_dim, self.a_dim).to(self.device)
        self.target_actor = ANet(self.s_dim, self.a_dim).to(self.device)
        self.target_critic = CNet(self.s_dim, self.a_dim).to(self.device)
        self._sync_targets()
        self.buf = ReplayBuf()

        # === parametri RL ===
        self.lr_a = 1e-4
        self.lr_c = 1e-3
        self.gamma = 0.95
        self.tau = 0.005
        self.batch = 64
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
        self.last_state = None
        self.last_action = None
        self.model_path = "./controllers/firm_ddpg.pt"
        if os.path.exists(self.model_path):
            ck = torch.load(self.model_path, map_location=self.device)
            self.actor.load_state_dict(ck["actor"])
            self.critic.load_state_dict(ck["critic"])

    # === decisione ogni periodo ===
    def control(self, t):
        state = self._state_vec()
        s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            a = self.actor(s_t)[0].cpu().item()
        self.actor.train()
        delta = int(np.round(a * 3))
        delta = max(self.min_cores - self.cores, min(delta, self.max_cores - self.cores))
        self.cores += delta

        # update se in train
        if self.train and self.last_state is not None:
            r = self._reward()
            self.buf.store(self.last_state, [self.last_action], [r], state)
            if len(self.buf) >= self.batch:
                self._learn()

        # caching
        self.last_state = state
        self.last_action = a

        # logging
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(f"{t:.1f},{self.cores},{self.monitoring.getRTp95():.3f}\n")
        if self.train:
            self.save()

    # === stato a 5 dimensioni ===
    def _state_vec(self):
        p95 = self.monitoring.getRTp95()
        lat_ratio = min(2.0, p95 / self.setpoint)
        q_norm = self.monitoring.getQueueLen() / 100.0
        rate = self.monitoring.getArrivalRate()
        rate_norm = rate / 1000.0
        trend = 0.0 if not hasattr(self, "prev_p95") else 10 * (p95 - self.prev_p95) / self.setpoint
        self.prev_p95 = p95
        rep_norm = (self.cores - self.min_cores) / (self.max_cores - self.min_cores)
        return np.array([lat_ratio, q_norm, rate_norm, trend, rep_norm], dtype=np.float32)

    # === reward negativo su SLA violata + costo ===
    def _reward(self):
        ratio = self.monitoring.getRTp95() / self.setpoint
        lat_pen = 10 * max(ratio - 1, 0)
        cost_pen = 0.03 * self.cores
        return -(lat_pen + cost_pen)

    # === training ===
    def _learn(self):
        s, a, r, s_ = self.buf.sample(self.batch)
        s, a, r, s_ = s.to(self.device), a.to(self.device), r.to(self.device), s_.to(self.device)

        # critic loss
        with torch.no_grad():
            a_ = self.target_actor(s_)
            q_ = self.target_critic(s_, a_)
            q_target = r + self.gamma * q_
        q_val = self.critic(s, a)
        critic_loss = nn.MSELoss()(q_val, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # actor loss
        a_pred = self.actor(s)
        actor_loss = -self.critic(s, a_pred).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # soft update target nets
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)

    def _sync_targets(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tp.data * (1 - self.tau) + sp.data * self.tau)

    def save(self):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict()
        }, self.model_path)