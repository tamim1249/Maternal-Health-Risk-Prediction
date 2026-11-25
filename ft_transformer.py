"""FT-Transformer with an ordinal (CORAL) or softmax head, plus a scikit-learn wrapper."""
from .config import *

import torch, torch.nn as nn, torch.nn.functional as F
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FTNet(nn.Module):
    def __init__(self, n_feat, n_classes, d=32, n_layers=3, n_heads=4, drop=0.15, ordinal=True):
        super().__init__()
        self.ordinal, self.K = ordinal, n_classes
        self.W = nn.Parameter(torch.randn(n_feat, d) / math.sqrt(d))   # numerical feature tokenizer
        self.B = nn.Parameter(torch.zeros(n_feat, d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=2*d, dropout=drop,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        if ordinal:
            self.head = nn.Linear(d, 1, bias=False)                          # CORAL: shared weight
            self.ord_bias = nn.Parameter(torch.linspace(1.0, -1.0, n_classes - 1))  # K-1 thresholds
        else:
            self.head = nn.Linear(d, n_classes)
    def forward(self, x):
        tok = x.unsqueeze(-1) * self.W + self.B
        tok = torch.cat([self.cls.expand(x.size(0), -1, -1), tok], dim=1)
        z = self.norm(self.enc(tok)[:, 0])
        return self.head(z) + self.ord_bias if self.ordinal else self.head(z)

def ordinal_logits_to_probs(logits):
    cum = torch.sigmoid(logits)                                   # P(y > k)
    cum = torch.cummin(cum, dim=1).values                         # enforce rank consistency
    K1 = cum.size(1)
    p = [1 - cum[:, 0]] + [cum[:, k-1] - cum[:, k] for k in range(1, K1)] + [cum[:, -1]]
    p = torch.stack(p, 1).clamp_min(1e-6)
    return p / p.sum(1, keepdim=True)

def _loss(model, logits, y, w):
    if model.ordinal:
        levels = (y.unsqueeze(1) > torch.arange(model.K - 1, device=y.device).unsqueeze(0)).float()
        l = F.binary_cross_entropy_with_logits(logits, levels, reduction="none").mean(1)
    else:
        l = F.cross_entropy(logits, y, reduction="none")
    return (l * w).mean()

@torch.no_grad()
def ft_predict(model, X, bs=4096):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i:i+bs], dtype=torch.float32, device=DEVICE)
        lg = model(xb)
        out.append((ordinal_logits_to_probs(lg) if model.ordinal else F.softmax(lg, 1)).cpu().numpy())
    return np.concatenate(out)

def train_ft(Xtr, ytr, Xval, yval, K, ordinal=True, sw=None, seed=42, fixed_epochs=None):
    """Early stopping on (Xval, yval); or, with fixed_epochs, train exactly that many epochs (refit on all data)."""
    set_seed(seed); c = CFG["FT"]
    m = FTNet(Xtr.shape[1], K, c["d_token"], c["n_layers"], c["n_heads"], c["dropout"], ordinal).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=c["lr"], weight_decay=c["wd"])
    Xt = torch.as_tensor(Xtr, dtype=torch.float32, device=DEVICE)
    yt = torch.as_tensor(ytr, dtype=torch.long, device=DEVICE)
    wt = torch.as_tensor(np.ones(len(ytr)) if sw is None else sw, dtype=torch.float32, device=DEVICE)
    best, best_state, best_ep, wait = np.inf, None, 0, 0
    for ep in range(fixed_epochs or c["max_epochs"]):
        m.train(); perm = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), c["batch"]):
            b = perm[i:i+c["batch"]]
            if len(b) < 2: continue
            opt.zero_grad(); loss = _loss(m, m(Xt[b]), yt[b], wt[b]); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if fixed_epochs: continue
        v = log_loss(yval, ft_predict(m, Xval), labels=list(range(K)))
        if v < best - 1e-4: best, best_state, best_ep, wait = v, copy.deepcopy(m.state_dict()), ep, 0
        else:
            wait += 1
            if wait >= c["patience"]: break
    if fixed_epochs: m.best_epoch_ = fixed_epochs - 1
    else: m.load_state_dict(best_state); m.best_epoch_ = best_ep
    m.n_params_ = sum(p.numel() for p in m.parameters())
    return m

class FTClassifier(BaseEstimator, ClassifierMixin):
    """Plain FT-Transformer baseline (softmax, no knowledge features, no fusion)."""
    def __init__(self, ordinal=False, val_frac=0.15, seed=42):
        self.ordinal, self.val_frac, self.seed = ordinal, val_frac, seed
    def fit(self, X, y):
        y = np.asarray(y); self.classes_ = np.unique(y)
        tr, va = next(StratifiedShuffleSplit(1, test_size=self.val_frac, random_state=self.seed).split(X, y))
        self.model_ = train_ft(X[tr], y[tr], X[va], y[va], len(self.classes_), self.ordinal, None, self.seed)
        return self
    def predict_proba(self, X): return ft_predict(self.model_, np.asarray(X, dtype=np.float32))
    def predict(self, X): return self.predict_proba(X).argmax(1)
