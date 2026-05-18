import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from typing import Optional

'''
MTR_model: Defines the Multi-Task Regression Transformer (MTR) model architecture
with SwiGLU activation, Multi-Head Attention with Rotary Positional Embeddings,
and a Transformer stack.
'''


class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim * 2, bias=True)
        self.linear2 = nn.Linear(hidden_dim, input_dim, bias=True)
        self.dropout = nn.Dropout(0.1)  # Add dropout for regularization
    def forward(self, x):
        # x: (N, input_dim)
        x1, x2 = self.linear1(x).chunk(2, dim=-1)
        output = self.linear2(F.silu(x1) * x2)
        return self.dropout(output)


class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verification)
    can be found in the Meta Llama codebase.

    In this implementation we cache the embeddings for each position up to
    ``max_seq_len`` during initialization.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = self._compute_theta(device=torch.device("cpu"))
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def _compute_theta(self, device: torch.device) -> torch.Tensor:
        exponents = torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
        exponents = exponents[: (self.dim // 2)] / float(self.dim)
        base = torch.tensor(float(self.base), dtype=torch.float32, device=device)
        return torch.pow(base, -exponents)

    def _theta_is_valid(self) -> bool:
        if not hasattr(self, "theta"):
            return False
        theta = self.theta
        if not torch.isfinite(theta).all():
            return False
        if not ((theta > 0.0).all() and (theta <= 1.0).all()):
            return False
        return True

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        if not self._theta_is_valid():
            self.theta = self._compute_theta(device=torch.device("cpu"))

        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.size(1)

        if (
            (not self._theta_is_valid())
            or (not hasattr(self, "cache"))
            or (not torch.isfinite(self.cache).all())
            or (self.cache.size(0) < seq_len)
            or (self.theta.device != x.device)
        ):
            theta = self._compute_theta(device=x.device)
            self.theta = theta
            self.build_rope_cache(max(self.max_seq_len, seq_len))

        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]

        # Cast to float to match the reference implementation.
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        x_out = x_out.flatten(3)
        return x_out.type_as(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.rotary = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=max_seq_len)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(0.1)  # Add dropout for regularization

    def forward(self, x, input_pos=None, mask=None):
        B, T, C = x.shape  # Batch, sequence, embedding dim
        
        # project into queries, keys, and values
        q, k, v = self.qkv_proj(x).view(B, T, 3, self.num_heads, self.head_dim).unbind(2)  # (B, T, num_heads, head_dim)

        # Apply rotary positional embeddings to queries and keys
        q, k = self.rotary(q, input_pos=input_pos), self.rotary(k, input_pos=input_pos)

        # Reshape to (B, num_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if mask is not None:
            # set padding positions to -inf
            mask = mask.to(dtype=torch.float32)  # Ensure mask is float
            mask = (1.0 - mask) * -1e9  # Convert to -inf for padding positions
            
            # mask: (B, T) -> (B, 1, 1, T)
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            mask = mask.expand(B, 1, T, T)  # expands to (batch, 1, seqlen, seqlen)
            
        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.out_proj(attn_output)
        return self.dropout(attn_output)


class UnifiedTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, max_seq_len):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, max_seq_len)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim, ffn_hidden_dim)

    # def forward(self, x, mask=None):
    #     x = x + self.attn(self.attn_norm(x), mask=mask)
    #     x = x + self.ffn(self.ffn_norm(x))
    #     return x
    def forward(self, x, input_pos=None, mask=None):
        x = x + self.attn(self.attn_norm(x), input_pos=input_pos, mask=mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

# class TransformerStack(nn.Module):
#     def __init__(self, num_blocks, embed_dim, num_heads, ffn_hidden_dim, max_seq_len):
#         super().__init__()
#         self.blocks = nn.ModuleList([
#             UnifiedTransformerBlock(embed_dim, num_heads, ffn_hidden_dim, max_seq_len)
#             for _ in range(num_blocks)
#         ])
#         self.norm = nn.LayerNorm(embed_dim)

#     def forward(self, x, mask=None):
#         for block in self.blocks:
#             x = block(x, mask=mask)
#         return self.norm(x)

class TransformerStack(nn.Module):
    def __init__(self, num_blocks, embed_dim, num_heads, ffn_hidden_dim, max_seq_len):
        super().__init__()
        self.blocks = nn.ModuleList([
            UnifiedTransformerBlock(embed_dim, num_heads, ffn_hidden_dim, max_seq_len)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, input_pos=None, mask=None):
        for block in self.blocks:
            x = block(x, input_pos=input_pos, mask=mask)
        return self.norm(x)

class MTR_model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_blocks: int,
        num_heads: int,
        ffn_hidden_dim: int,
        output_dim: int,
        max_seq_len: int,
        num_tasks: int = 0,
        head_sizes = None,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.transformer = TransformerStack(
            num_blocks, embed_dim, num_heads, ffn_hidden_dim, max_seq_len
        )
        self.sequence_head = nn.Linear(embed_dim, output_dim, bias=True)

        # regression / classification heads
        self.num_tasks = num_tasks
        head_sizes = head_sizes or []
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, size)
            )
            for size in head_sizes
        ])

    def forward(self, x, mask=None, pad_token_id=0, input_pos=None):
        x = self.embed(x)
        x = self.transformer(x, mask=mask, input_pos=input_pos)
        # generate logits for MLM
        # print(f"x shape: {x.shape}")  # Debugging line to check the shape of x
        logits = self.sequence_head(x)
        # print(f"logits shape: {logits.shape}")  # Debugging line to check the shape of logits

        # mean‐pool
        # remove non-informative tokens (e.g., padding)
        if mask is not None:
            # Expand mask to match x's shape
            # Mask will now be (batch_size, seq_len, embed_dim)
            mask = mask.unsqueeze(-1)  # Add embedding dim: shape becomes (batch_size, seq_len, 1)
            # print(f"mask unsqueeze shape: {mask.shape}")  # Debugging line to check the shape of mask
            # Apply the mask to the embeddings
            masked_x = x * mask  # Shape remains (batch_size, seq_len, embed_dim)
            # print(f"masked_x shape: {masked_x.shape}")  # Debugging line to check the shape of masked_x
            # Sum embeddings across the sequence dimension, considering only masked tokens
            sum_x = masked_x.sum(dim=1)  # Shape = (batch_size, embed_dim)
            # print(f"sum_x shape: {sum_x.shape}")
            # Sum the mask along sequence dimension to get the number of non-padded tokens
            mask_sum = mask.sum(dim=1).clamp(min=1e-6)  # Shape = (batch_size, 1)
            # print(f"mask_sum shape: {mask_sum.shape}")  # Debugging line to check the shape of mask_sum
            # Compute a mean for masked tokens
            mean_pool = sum_x / mask_sum  # Shape = (batch_size, embed_dim)
        else:
            # No mask provided: Mean-pool over the entire sequence dimension
            mean_pool = x.mean(dim=1)  # Shape = (batch_size, embed_dim)

        descriptor_out = self.heads[0](mean_pool)

        return x, logits, descriptor_out

    # def forward(self, x, input_pos=None, mask=None, pad_token_id=0):
    #     # mask = mask.unsqueeze(1).unsqueeze(2) if mask is not None else None

    #     x = self.embed(x)
    #     x = self.transformer(x, input_pos=input_pos, mask=mask)
    #     # generate logits for MLM
    #     logits = self.sequence_head(x)

    #     # mean‐pool
    #     pooled = x.mean(dim=1) # (batch, embed_dim)

    #     # one output per head
    #     descriptor_out = self.heads[0](pooled)

    #     return x, logits, descriptor_out


# set up pytorch lightning module
class pl_model(pl.LightningModule):
    def __init__(self, 
                 model: "MTR_model", 
                 lr: float, 
                 weight_decay: float = 0.0, 
                 warmup_steps: int = 0, 
                 total_steps: int = 0,
                 alpha: float = 0.6,
                 beta: float = 0.4,
):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, input_pos=None, mask=None):
        return self.model(x, input_pos=input_pos, mask=mask)

    def shared_step(self, batch):
        """Shared logic between training_step and validation_step."""
        input_ids = batch.get("input_ids")  # Sequence tokens tensor (B, T)
        labels = batch.get("labels")      # Target labels (B,) or (B, C)
        rdkit_descs = batch.get("rdkit_descs")  # RDKit descriptors (B, D)
        pad_mask = batch.get("pad_masking")

        # Model forward pass
        _, logits, outs = self(input_ids, mask=pad_mask)

        # cross-entropy loss for MLM, mse loss for RDKit descriptors
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        mtr_loss = F.mse_loss(outs, rdkit_descs, reduction='mean') if rdkit_descs is not None else 0.0

        loss = (self.alpha * ce_loss) + (self.beta * mtr_loss)

        # elif hasattr(self.model, "num_tasks") and self.model.num_tasks > 1:
        #     # Multi-task mode
        #     for i, head_out in enumerate(outs):
        #         # Select samples for task i
        #         mask = (task_ids == i)
        #         if mask.any():
        #             preds = head_out[mask]       # (Ni, Ci)
        #             targ = labels[mask]          # (Ni,) or (Ni, Ci)
        #             task_loss = criteria[i](preds, targ)  # Per-task loss
        #             loss += task_loss

        return loss, ce_loss, mtr_loss  # Return both losses for logging

    def training_step(self, batch, batch_idx):
        loss, ce_loss, mtr_loss = self.shared_step(batch)

        # Log training metrics
        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=batch["input_ids"].size(0), sync_dist=True)
        self.log("train_cel", ce_loss, prog_bar=True, logger=True, batch_size=batch["input_ids"].size(0), sync_dist=True)
        self.log("train_mse", mtr_loss, prog_bar=True, logger=True, batch_size=batch["input_ids"].size(0), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, ce_loss, mtr_loss = self.shared_step(batch)

        # Log validation metrics
        self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=batch["input_ids"].size(0), sync_dist=True)
        self.log("val_cel", ce_loss, prog_bar=True, logger=True, batch_size=batch["input_ids"].size(0), sync_dist=True)
        self.log("val_mse", mtr_loss, prog_bar=True, logger=True, batch_size=batch["input_ids"].size(0), sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        # Set weight decay if specified
        if self.weight_decay > 0:
            for param_group in optimizer.param_groups:
                param_group["weight_decay"] = self.weight_decay

        # Set up the scheduler
        scheduler = []
        if self.warmup_steps > 0:
            # Combine Linear Warmup and Cosine Annealing into a single scheduler
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[
                        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=self.warmup_steps),
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=self.total_steps - self.warmup_steps,
                            eta_min=0.1 * self.lr  # Set minimum learning rate to 10% of initial lr
                        )
                    ],
                    milestones=[self.warmup_steps]  # Milestone to switch from warmup to cosine annealing
                ),
                'interval': 'step',
                'frequency': 1
            }
        
        return [optimizer], [scheduler]