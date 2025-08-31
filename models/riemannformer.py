import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================
# Utilities
# =============================

def make_skew_symmetric(A: torch.Tensor) -> torch.Tensor:
    """
    Enforce skew-symmetry: X = A - A^T
    """
    return A - A.transpose(-1, -2)

def matrix_exp(X: torch.Tensor, steps: int = 20) -> torch.Tensor:
    """
    Compute matrix exponential via a truncated Taylor series.
    exp(X) = sum_{k=0}^∞ X^k / k!
    """
    out = torch.eye(X.size(-1), device=X.device).unsqueeze(0).expand(X.size(0), -1, -1).clone()
    term = torch.eye(X.size(-1), device=X.device).unsqueeze(0).expand(X.size(0), -1, -1).clone()
    factorial = 1.0
    for k in range(1, steps):
        term = term @ X / k
        out = out + term
    return out

# =============================
# RiemannFormer Attention Core
# =============================

class RiemannFormerAttention(nn.Module):
    def __init__(self, dim, num_heads=8, locality_focusing=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # QKV projection
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Scaling factors s_i (learnable per head)
        self.log_s = nn.Parameter(torch.zeros(num_heads))  # unconstrained param
        # Rotation generator X (skew-symmetric)
        A = torch.randn(num_heads, self.head_dim, self.head_dim) * 0.01
        self.X = nn.Parameter(make_skew_symmetric(A))

        # Optional locality focusing
        self.locality_focusing = locality_focusing
        if locality_focusing:
            self.log_sigma = nn.Parameter(torch.zeros(1))  # learnable bandwidth

    def forward(self, x, positions=None):
        """
        x: (B, L, D)
        positions: (L,) token positions (ints), optional
        """
        B, L, D = x.shape
        H, d = self.num_heads, self.head_dim

        # Q, K, V projections
        Q = self.q_proj(x).view(B, L, H, d).transpose(1, 2)  # (B, H, L, d)
        K = self.k_proj(x).view(B, L, H, d).transpose(1, 2)
        V = self.v_proj(x).view(B, L, H, d).transpose(1, 2)

        # Scaling factors s_i (positive via exp)
        s = torch.exp(self.log_s)  # (H,)

        # Compute rotation matrices exp(iX)
        if positions is None:
            positions = torch.arange(L, device=x.device)

        exp_mX = []
        for pos in positions:
            exp_mX.append(matrix_exp(pos * self.X))  # (H, d, d)
        exp_mX = torch.stack(exp_mX, dim=0)  # (L, H, d, d)

        # Construct T_i = s_i^{-1/2} exp(iX)
        scale = s.view(H, 1, 1, 1).rsqrt()  # (H,1,1,1)
        T = scale * exp_mX.transpose(0, 1)  # (H, L, d, d)

        # Map queries & keys into reference space: T_i^{-1} q_i
        Q_ref = torch.einsum('hlij,bhlj->bhli', T.inverse(), Q)
        K_ref = torch.einsum('hlij,bhlj->bhli', T.inverse(), K)

        # Inner product in reference space
        attn_scores = torch.einsum('bhid,bhjd->bhij', Q_ref, K_ref) / (d ** 0.5)

        # Optional locality focusing
        if self.locality_focusing and positions is not None:
            pos = positions.float()
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (L, L)
            sigma = torch.exp(self.log_sigma)
            locality = torch.exp(- (diff ** 2) / (2 * sigma ** 2))  # Gaussian
            attn_scores = attn_scores + torch.log(locality + 1e-6)  # biasing

        # Attention weights
        attn = F.softmax(attn_scores, dim=-1)

        # Weighted sum of values
        out = torch.einsum('bhij,bhjd->bhid', attn, V)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)

# =============================
# Mini RiemannFormer Transformer Block
# =============================

class RiemannFormerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.attn = RiemannFormerAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, positions=None):
        x = x + self.attn(self.norm1(x), positions=positions)
        x = x + self.mlp(self.norm2(x))
        return x

# =============================
# Example: RiemannFormer Encoder
# =============================

class RiemannFormerEncoder(nn.Module):
    def __init__(self, dim=128, depth=6, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.layers = nn.ModuleList([
            RiemannFormerBlock(dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, positions=None):
        for blk in self.layers:
            x = blk(x, positions=positions)
        return self.norm(x)
