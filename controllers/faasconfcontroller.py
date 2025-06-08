import os
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .controller import Controller


class _Attention(nn.Module):
    """Simple scaled attention used by FaaSConf networks."""

    def __init__(self, in_dim, hid_dim):
        super().__init__()
        bound = (4 / in_dim) ** 0.5
        self.q_weight = nn.Parameter(torch.empty(in_dim, hid_dim).uniform_(-bound, bound))
        self.k_weight = nn.Parameter(torch.empty(in_dim, hid_dim).uniform_(-bound, bound))

    def forward(self, s, gmat):
        q = torch.einsum("ijk,km->ijm", s, self.q_weight)
        k = torch.einsum("ijk,km->ijm", s, self.k_weight).permute(0, 2, 1)
        att = torch.square(torch.bmm(q, k)) * gmat
        return att / (att.sum(dim=2, keepdim=True) + 1e-3)


class _Actor(nn.Module):
    """Actor network from the FaaSConf paper."""

    def __init__(self, s_dim, a_dim, n_func, att_dim=32):
        super().__init__()
        self.n_func = n_func
        self.att = _Attention(s_dim, att_dim)
        self.fc1 = nn.Linear(s_dim * 2, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, a_dim)

    def forward(self, s, gmat):
        att = self.att(s, gmat)
        s_aug = torch.cat((s, torch.bmm(att, s)), dim=-1)
        x = s_aug.view(-1, s_aug.size(-1))
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = torch.sigmoid(self.out(x))
        return x.view(-1, self.n_func, -1)


class _Critic(nn.Module):
    """Critic network from the FaaSConf paper."""

    def __init__(self, s_dim, a_dim, n_func, att_dim=32):
        super().__init__()
        self.n_func = n_func
        self.att = _Attention(s_dim, att_dim)
        self.fc1 = nn.Linear((s_dim + a_dim) * 2, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, 1)

    def forward(self, s, a, gmat):
        att = self.att(s, gmat)
        s_aug = torch.cat((s, torch.bmm(att, s)), dim=-1)
        a_aug = torch.cat((a, torch.bmm(att, a)), dim=-1)
        x = torch.cat((s_aug, a_aug), dim=-1)
        x = x.view(-1, x.size(-1))
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.out(x)
        return x.view(-1, self.n_func, 1)


class _ReplayBuf:
    def __init__(self, cap=10000):
        self.cap = cap
        self.buf = []

    def store(self, s, a, r, s_):
        if len(self.buf) >= self.cap:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_))

    def sample(self, batch):
        idx = np.random.choice(len(self.buf), batch)
        s, a, r, s_ = zip(*[self.buf[i] for i in idx])
        return (torch.tensor(s, dtype=torch.float32),
                torch.tensor(a, dtype=torch.float32),
                torch.tensor(r, dtype=torch.float32),
                torch.tensor(s_, dtype=torch.float32))

    def __len__(self):
        return len(self.buf)


class FaaSConfController(Controller):
    """Attention based DDPG autoscaler inspired by FaaSConf."""

    def __init__(self, period, init_cores, *,
                 min_cores=1, max_cores=1000, st=0.8, name="FaaSConf",
                 train=True, log_dir="./logs"):
        super().__init__(period, init_cores, st, name=name)
        self.min_cores, self.max_cores = min_cores, max_cores
        self.train = train
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(log_dir, exist_ok=True)
        ts = int(time.time())
        self.log_path = f"{log_dir}/faasconf-{ts}.csv" if train else None
        self.model_path = "./controllers/faasconfcontroller.pt"

        self.s_dim = 5
        self.a_dim = 1
        self.n_func = 1
        self.gmat = torch.ones(self.n_func, self.n_func, device=self.device)
        self.actor = _Actor(self.s_dim, self.a_dim, self.n_func).to(self.device)
        self.critic = _Critic(self.s_dim, self.a_dim, self.n_func).to(self.device)
        self.target_actor = _Actor(self.s_dim, self.a_dim, self.n_func).to(self.device)
        self.target_critic = _Critic(self.s_dim, self.a_dim, self.n_func).to(self.device)
        self._sync_targets()
        self.buf = _ReplayBuf()

        self.lr_a = 1e-4
        self.lr_c = 1e-3
        self.gamma = 0.95
        self.tau = 0.005
        self.batch = 64
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
        self.last_state = None
        self.last_action = None
        if os.path.exists(self.model_path):
            ck = torch.load(self.model_path, map_location=self.device)
            self.actor.load_state_dict(ck.get("actor", {}))
            self.critic.load_state_dict(ck.get("critic", {}))

    def control(self, t):
        state = self._state_vec()
        s_t = torch.tensor(state, dtype=torch.float32).view(1, self.n_func, -1).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            a = self.actor(s_t, self.gmat.unsqueeze(0))[0, 0].cpu().item()
        self.actor.train()
        delta = int(np.round((a - 0.5) * 6))
        delta = max(self.min_cores - self.cores, min(delta, self.max_cores - self.cores))
        self.cores += delta

        if self.train and self.last_state is not None:
            r = self._reward()
            self.buf.store(self.last_state, [self.last_action], [r], state)
            if len(self.buf) >= self.batch:
                self._learn()

        self.last_state = state
        self.last_action = a

        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(f"{t:.1f},{self.cores},{self.monitoring.getRTp95():.3f}\n")
        if self.train:
            self.save()

    def _state_vec(self):
        p95 = self.monitoring.getRTp95()
        lat_ratio = min(2.0, p95 / self.setpoint)
        q_norm = self.monitoring.getQueueLen() / 100.0
        rate_norm = self.monitoring.getArrivalRate() / 1000.0
        trend = 0.0 if not hasattr(self, "prev_p95") else 10 * (p95 - self.prev_p95) / self.setpoint
        self.prev_p95 = p95
        rep_norm = (self.cores - self.min_cores) / (self.max_cores - self.min_cores)
        return np.array([lat_ratio, q_norm, rate_norm, trend, rep_norm], dtype=np.float32)

    def _reward(self):
        ratio = self.monitoring.getRTp95() / self.setpoint
        lat_pen = 10 * max(ratio - 1, 0)
        cost_pen = 0.03 * self.cores
        return -(lat_pen + cost_pen)

    def _learn(self):
        s, a, r, s_ = self.buf.sample(self.batch)
        s = s.view(-1, self.n_func, self.s_dim).to(self.device)
        a = a.view(-1, self.n_func, self.a_dim).to(self.device)
        r = r.view(-1, self.n_func, 1).to(self.device)
        s_ = s_.view(-1, self.n_func, self.s_dim).to(self.device)

        with torch.no_grad():
            a_ = self.target_actor(s_, self.gmat.unsqueeze(0))
            q_ = self.target_critic(s_, a_, self.gmat.unsqueeze(0))
            q_target = r + self.gamma * q_
        q_val = self.critic(s, a, self.gmat.unsqueeze(0))
        critic_loss = nn.MSELoss()(q_val, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        a_pred = self.actor(s, self.gmat.unsqueeze(0))
        actor_loss = -self.critic(s, a_pred, self.gmat.unsqueeze(0)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)

    def _sync_targets(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tp.data * (1 - self.tau) + sp.data * self.tau)

    def save(self):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, self.model_path)
