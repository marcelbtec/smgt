"""logic_and_lora_demo.py

Shows *both* types of adapters:
  1. Logic Adapter  (A Bᵀ) – enforces symbolic mask at inference time.
  2. LoRA Adapter   (ΔW = UVᵀ) – fine‑tunes Q,K projection weights cheaply.

Steps
-----
• Build tiny heterogeneous graph (Protein / Drug / Gene).
• Train baseline Graph‑Transformer classifier.
• Attach Logic Adapter – violation drops to 0.
• Detach; attach LoRA adapters learned in 100 gradient steps – accuracy changes
  while core weights stay frozen; violation remains high (no logic mask).
• Combine Logic + LoRA – both constraint and LoRA improvement active.

Run:
    python logic_and_lora_demo.py
"""

from __future__ import annotations
import math, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

# reproducibility
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# toy heterogeneous graph
def build_graph():
    n,d=6,8
    x=torch.randn(n,d)
    types=torch.tensor([0,1,0,2,1,0])   # 0 Protein 1 Drug 2 Gene
    y=torch.randint(0,3,(n,))
    return x,types,y

RULE=[(1,2)]  # Drug -> Gene forbidden

# violation metric
def violation(att,types,rule,thr=1e-6):
    v=t=0
    for a,b in rule:
        rows=(types==a).nonzero().view(-1)
        cols=(types==b).nonzero().view(-1)
        if rows.numel() and cols.numel():
            sub=att[rows][:,cols]
            v+=(sub>thr).sum().item(); t+=sub.numel()
    return v/t if t else 0

# logic adapter helpers -------------------------------------------------------
def full_mask(types,rule):
    big_neg=-torch.finfo(torch.float32).max
    n=len(types); M=torch.zeros(n,n)
    for a,b in rule:
        r=(types==a).nonzero().view(-1); c=(types==b).nonzero().view(-1)
        if r.numel() and c.numel(): M[r[:,None],c]=big_neg
    return M

# Common adapter interface
class BaseAdapter(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class LogicAdapter(BaseAdapter):
    def __init__(self, rank: int = 2):
        super().__init__()
        self.rank = rank
        self.A = None
        self.B = None
        self.initialized = False
    
    def load_rules(self, types: torch.Tensor, rules: list[tuple[int, int]]):
        """Load rules and types to initialize the adapter parameters"""
        M = full_mask(types, rules)
        M[M == 0] = -0.5
        U, S, Vh = torch.linalg.svd(M)
        self.A = nn.Parameter(U[:, :self.rank] * S[:self.rank].sqrt(), requires_grad=False)
        self.B = nn.Parameter(Vh[:self.rank, :].T * S[:self.rank].sqrt(), requires_grad=False)
        self.initialized = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            raise RuntimeError("LogicAdapter not initialized. Call load_rules() first.")
        return self.A @ self.B.T

class LoRALinear(BaseAdapter):
    """LoRA‑wrapped Linear: W_out = W + U Vᵀ (rank r)"""
    def __init__(self, orig: nn.Linear, r: int = 2, alpha: float = 1.0):
        super().__init__()
        self.orig = orig
        self.r = r
        self.alpha = alpha
        self.U = nn.Parameter(torch.zeros(orig.out_features, r))
        self.V = nn.Parameter(torch.zeros(r, orig.in_features))
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.zeros_(self.V)
        self.scale = alpha/r
        for p in self.orig.parameters(): p.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.orig.weight + self.scale * (self.U @ self.V), self.orig.bias)

class MaskedAttn(nn.Module):
    def __init__(self, d: int, use_lora: bool = False, adapter: BaseAdapter | None = None):
        super().__init__()
        Lin = nn.Linear
        if use_lora: Lin = lambda *a, **k: LoRALinear(nn.Linear(*a, **k))
        self.q = Lin(d, d, False)
        self.k = Lin(d, d, False)
        self.v = Lin(d, d, False)
        self.adapter = adapter
    
    def attach_adapter(self, adapter: BaseAdapter | None):
        self.adapter = adapter
    
    def forward(self, x: torch.Tensor):
        s = (self.q(x) @ self.k(x).T) / math.sqrt(x.size(1))
        if self.adapter is not None:
            s = s + self.adapter(x)
        a = s.softmax(-1)
        return a @ self.v(x), a

class GT(nn.Module):
    def __init__(self, d: int, use_lora: bool = False, adapter: BaseAdapter | None = None):
        super().__init__()
        self.att = MaskedAttn(d, use_lora, adapter)
        self.head = nn.Linear(d, 3)
    
    def attach_adapter(self, adapter: BaseAdapter | None):
        self.att.attach_adapter(adapter)
    
    def forward(self, x: torch.Tensor):
        h, att = self.att(x)
        return self.head(h), att

# training utils
def train(model,x,y,epochs=200,lr=1e-2):
    opt=torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),lr=lr)
    for _ in range(epochs):
        opt.zero_grad(); out,_=model(x); F.cross_entropy(out,y).backward(); opt.step()

# main demo
def main():
    set_seed()
    x,types,y=build_graph()

    print("=== baseline training ===")
    base=GT(d=x.size(1))
    train(base,x,y)
    _,att=base(x); print("Baseline violation",violation(att,types,RULE))

    print("=== attach logic adapter ===")
    logic = LogicAdapter(rank=2)
    logic.load_rules(types, RULE)  # Load rules during inference
    base.attach_adapter(logic)
    _,att=base(x); print("Logic violation",violation(att,types,RULE))

    print("=== detach logic, add LoRA weight adapters ===")
    base.attach_adapter(None)
    lora_model=GT(d=x.size(1),use_lora=True)   # shares new params
    lora_model.load_state_dict(base.state_dict(),strict=False)  # copy weights
    train(lora_model,x,y,epochs=100,lr=5e-3)   # train only LoRA params
    _,att=lora_model(x); print("LoRA violation",violation(att,types,RULE))

    print("=== combine LoRA + Logic ===")
    logic = LogicAdapter(rank=2)
    logic.load_rules(types, RULE)
    lora_model.attach_adapter(logic)
    _,att=lora_model(x); print("LoRA+Logic violation",violation(att,types,RULE))

if __name__=='__main__': main()
