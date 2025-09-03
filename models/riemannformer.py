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

#cayley_exp - ajay
def cayley_exp(X: torch.Tensor) -> torch.Tensor:
    """
    Cayley transform approximation of exp(X).
    X: (..., d, d) skew-symmetric
    Returns: (..., d, d) orthogonal matrix
    """
    d = X.size(-1)
    I = torch.eye(d, device=X.device, dtype=X.dtype).expand_as(X)
    return torch.linalg.solve(I - 0.5 * X, I + 0.5 * X)

#faster matrix_exp - ajay
def fast_matrix_exp(X: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    
    # Batched matrix exponential for all positions × heads.
    # X: (H, d, d)
    # positions: (L,)
    # Returns: (L, H, d, d)
    
    H, d, _ = X.shape
    L = positions.size(0)

    # Scale X by positions
    scaled_X = positions[:, None, None, None] * X[None, :, :, :]   # (L, H, d, d)

    # Batch compute matrix exp on GPU
    exp_mX = torch.matrix_exp(scaled_X.reshape(-1, d, d))          # (L*H, d, d)

    return exp_mX.view(L, H, d, d).to(X.device)

# def matrix_exp(X: torch.Tensor, steps: int = 20) -> torch.Tensor:
#     """
#     Compute matrix exponential via a truncated Taylor series.
#     exp(X) = sum_{k=0}^∞ X^k / k!
#     """
#     out = torch.eye(X.size(-1), device=X.device).unsqueeze(0).expand(X.size(0), -1, -1).clone()
#     term = torch.eye(X.size(-1), device=X.device).unsqueeze(0).expand(X.size(0), -1, -1).clone()
#     factorial = 1.0
#     for k in range(1, steps):
#         term = term @ X / k
#         out = out + term
#     return out

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
        #cast input to float32 - ajay
        x = x.to(self.q_proj.weight.dtype)
        
        B, L, D = x.shape
        H, d = self.num_heads, self.head_dim
        print("ajay 1")
        # Q, K, V projections
        Q = self.q_proj(x).view(B, L, H, d).transpose(1, 2)  # (B, H, L, d)
        K = self.k_proj(x).view(B, L, H, d).transpose(1, 2)
        V = self.v_proj(x).view(B, L, H, d).transpose(1, 2)
        print("ajay 2")
 
        # Scaling factors s_i (positive via exp)
        s = torch.exp(self.log_s)  # (H,)

        # Compute rotation matrices exp(iX)
        if positions is None:
            positions = torch.arange(L, device=x.device)

        print("ajay 3")

        # exp_mX = []
        # for pos in positions:
        #     exp_mX.append(matrix_exp(pos * self.X))  # (H, d, d)
        # exp_mX = torch.stack(exp_mX, dim=0)  # (L, H, d, d)
        
        #faster matrix_exp call - ajay
        #exp_mX = fast_matrix_exp(self.X, positions)  # (L, H, d, d)

        #cayley_exp call - ajay
        # positions: (L,)
        # self.X: (H, d, d)
        
        L = positions.size(0)
        H, d, _ = self.X.shape
        
        # Scale X by positions: (L, H, d, d)
        scaled_X = positions[:, None, None, None] * self.X[None, :, :, :]
        
        # Apply Cayley transform in batch
        exp_mX = cayley_exp(scaled_X)  # (L, H, d, d)

        print("ajay 4")

        # Construct T_i = s_i^{-1/2} exp(iX)
        scale = s.view(H, 1, 1, 1).rsqrt()  # (H,1,1,1)
        T = scale * exp_mX.transpose(0, 1)  # (H, L, d, d)
        
        print("ajay 5")

        # Map queries & keys into reference space: T_i^{-1} q_i
        # Construct T_i = s_i^{-1/2} exp(iX)
        # Since T is orthogonal (Cayley/skew exp), T^{-1} = T^T
        T_T = T.transpose(-1, -2)            # (H, L, d, d)
        
        # Prepare empty output tensors
        Q_ref = torch.empty_like(Q)
        K_ref = torch.empty_like(K)
        
        # 🔹 Chunk over sequence length to save memory
        chunk_size = 32  # tune this for your GPU (16/32/64 usually works)

        print("ajay 6")
        
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
        
            # Slice T and Q/K for this chunk
            T_chunk = T_T[:, start:end]      # (H, chunk, d, d)
            Q_chunk = Q[:, :, start:end, :]  # (B,H,chunk,d)
            K_chunk = K[:, :, start:end, :]
        
            # Reshape for batched matmul
            Q_flat = Q_chunk.reshape(B*H* (end-start), d, 1)
            K_flat = K_chunk.reshape(B*H* (end-start), d, 1)
            T_flat = T_chunk.unsqueeze(0).expand(B, -1, -1, -1, -1) \
                                  .reshape(B*H*(end-start), d, d)
        
            print("ajay 7")
            
            # Multiply
            Q_out = torch.bmm(T_flat, Q_flat).view(B, H, end-start, d)
            K_out = torch.bmm(T_flat, K_flat).view(B, H, end-start, d)

            print("ajay 8")
            
            # Store results
            Q_ref[:,:,start:end,:] = Q_out
            K_ref[:,:,start:end,:] = K_out

            print("ajay 9")

        print("ajay 10")

        # Inner product in reference space
        attn_scores = torch.einsum('bhid,bhjd->bhij', Q_ref, K_ref) / (d ** 0.5)
        print("ajay 11")

        # Optional locality focusing
        if self.locality_focusing and positions is not None:
            pos = positions.float()
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (L, L)
            sigma = torch.exp(self.log_sigma)
            locality = torch.exp(- (diff ** 2) / (2 * sigma ** 2))  # Gaussian
            attn_scores = attn_scores + torch.log(locality + 1e-6)  # biasing

        print("ajay 12")
        
        # Attention weights
        attn = F.softmax(attn_scores, dim=-1)
        print("ajay 13")

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
