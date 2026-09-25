"""Deep genomic prediction models (MLP, 1D-CNN, patch transformer), training
with early stopping on validation Pearson correlation, Monte-Carlo dropout and
deep-ensemble uncertainty, and class-token attention extraction."""
import copy
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)


class MLP(nn.Module):
    def __init__(self, p_in, hidden=256, dropout=0.3, **_):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(p_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CNN1D(nn.Module):
    def __init__(self, p_in, channels=16, dropout=0.3, **_):
        super().__init__()
        c = channels
        self.conv = nn.Sequential(
            nn.Conv1d(1, c, 16, stride=4, padding=6), nn.BatchNorm1d(c), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(c, 2 * c, 8, stride=4, padding=2), nn.BatchNorm1d(2 * c), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(2 * c, 2 * c, 4, stride=2, padding=1), nn.BatchNorm1d(2 * c), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(2 * c * 16, 1))

    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1))).squeeze(-1)


class Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)
        self.last_attn = None

    def forward(self, x, keep_attn=False):
        h = self.n1(x)
        a, w = self.att(h, h, h, need_weights=keep_attn, average_attn_weights=True)
        if keep_attn:
            self.last_attn = w.detach()
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.n2(x)))


class GenomicTransformer(nn.Module):
    """Contiguous marker patches -> tokens, learnable class token, learned positions, pre-norm blocks."""

    def __init__(self, p_in, patch=64, d=64, depth=2, heads=4, dropout=0.1, **_):
        super().__init__()
        assert p_in % patch == 0
        self.patch, self.n_tok = patch, p_in // patch
        self.embed = nn.Linear(patch, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.posn = nn.Parameter(torch.randn(1, self.n_tok + 1, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x, keep_attn=False):
        b = x.shape[0]
        t = self.embed(x.view(b, self.n_tok, self.patch))
        t = torch.cat([self.cls.expand(b, -1, -1), t], 1) + self.posn
        for i, blk in enumerate(self.blocks):
            t = blk(t, keep_attn=keep_attn and i == len(self.blocks) - 1)
        return self.head(self.norm(t[:, 0])).squeeze(-1)

    def cls_attention(self, x):
        """Class-token attention over patches in the last block (heads averaged)."""
        self.eval()
        with torch.no_grad():
            self.forward(torch.as_tensor(x, dtype=torch.float32), keep_attn=True)
        return self.blocks[-1].last_attn[:, 0, 1:].numpy()


ARCH = {"MLP": MLP, "CNN": CNN1D, "TRF": GenomicTransformer}

# Hyperparameter grids searched once per architecture on the two tuning traits.
GRID = {
    "MLP": [dict(hidden=256, dropout=0.3, lr=1e-3, wd=1e-2), dict(hidden=128, dropout=0.5, lr=1e-3, wd=1e-1),
            dict(hidden=512, dropout=0.5, lr=5e-4, wd=1e-2), dict(hidden=64, dropout=0.3, lr=1e-3, wd=1e-1)],
    "CNN": [dict(channels=16, dropout=0.3, lr=1e-3, wd=1e-2), dict(channels=8, dropout=0.3, lr=1e-3, wd=1e-1),
            dict(channels=32, dropout=0.5, lr=5e-4, wd=1e-2), dict(channels=16, dropout=0.5, lr=5e-4, wd=1e-1)],
    "TRF": [dict(d=64, depth=2, dropout=0.1, lr=5e-4, wd=1e-2), dict(d=32, depth=1, dropout=0.1, lr=1e-3, wd=1e-2),
            dict(d=64, depth=1, dropout=0.3, lr=5e-4, wd=1e-1), dict(d=32, depth=2, dropout=0.3, lr=1e-3, wd=1e-1)],
}
MAX_EPOCHS, PATIENCE, BATCH = 150, 30, 32


def _pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def train_net(arch, hp, Xtr, ytr, Xva, yva, seed, criterion="r"):
    """AdamW, cosine LR, grad clip 1.0, MSE loss; early stopping on validation Pearson r
    (criterion="r", the default used throughout) or on validation MSE (criterion="mse")."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = ARCH[arch](Xtr.shape[1], **hp)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EPOCHS)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    Xv = torch.tensor(Xva, dtype=torch.float32)
    best, best_state, best_ep, wait = -np.inf, None, 0, 0
    lossf = nn.MSELoss()
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(yt), generator=g)
        for s in range(0, len(yt), BATCH):
            b = perm[s:s + BATCH]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss = lossf(model(Xt[b]), yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv).numpy()
        r = _pearson(pv, yva) if criterion == "r" else -float(np.mean((pv - yva) ** 2))
        if r > best:
            best, best_state, best_ep, wait = r, copy.deepcopy(model.state_dict()), ep, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best, best_ep


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


def mc_dropout(model, X, passes, seed):
    """Dropout active, batch-norm statistics frozen."""
    torch.manual_seed(seed)
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        draws = np.stack([model(xt).numpy() for _ in range(passes)])
    model.eval()
    return draws.mean(0), draws.std(0, ddof=1)
