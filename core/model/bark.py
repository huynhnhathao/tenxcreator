"""
codes adapted from https://github.com/suno-ai/bark
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    input_vocab_size: int = 10_048
    output_vocab_size: int = 10_048
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )


@dataclass
class FineGPTConfig(GPTConfig):
    n_codes_total: int = 8
    n_codes_given: int = 1


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()

    def forward(self, x) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            # print("WARNING: using slow attention. Flash Attention atm needs PyTorch nightly and dropout=0.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(
        self, x: torch.Tensor, past_kv: torch.Tensor = None, use_cache: bool = False
    ):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        if past_kv is not None:
            past_key = past_kv[0]
            past_value = past_kv[1]
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        FULL_T = k.shape[-2]

        if use_cache is True:
            present = (k, v)
        else:
            present = None

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            if past_kv is not None:
                # When `past_kv` is provided, we're doing incremental decoding and `q.shape[2] == 1`: q only contains
                # the query for the last token. scaled_dot_product_attention interprets this as the first token in the
                # sequence, so if is_causal=True it will mask out all attention from it. This is not what we want, so
                # to work around this we set is_causal=False.
                is_causal = False
            else:
                is_causal = True

            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout, is_causal=is_causal
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(
                self.bias[:, :, FULL_T - T : FULL_T, :FULL_T] == 0, float("-inf")
            )
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return (y, present)


class Block(nn.Module):

    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.layer_idx = layer_idx

    def forward(
        self, x: torch.Tensor, past_kv: torch.Tensor = None, use_cache: bool = False
    ):
        attn_output, prev_kvs = self.attn(
            self.ln_1(x), past_kv=past_kv, use_cache=use_cache
        )
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return (x, prev_kvs)


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.input_vocab_size is not None
        assert config.output_vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.input_vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config, idx) for idx in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.output_vocab_size, bias=False)
        # Note: lm_head lacks bias, implying parameter sharing with wte for efficiency

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        merge_context: bool = False,
        past_kv: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        use_cache: bool = False,
    ):
        device = idx.device
        b, t = idx.size()
        if past_kv is not None:
            # When past_kv is provided, this is optimized for autoregressive generation
            assert (
                t == 1
            ), "should only pass in the last token of the sequence when using kv_cache"
            # Shape: (b, 1, n_embd), single token case
            tok_emb = self.transformer.wte(idx)
        else:
            if merge_context:
                # Custom feature: assumes first 256 tokens are one context, next 256 another, rest is sequence
                assert idx.shape[1] >= 256 + 256 + 1
                t = idx.shape[1] - 256  # Adjusts t for merged context length
            else:
                assert (
                    t <= self.config.block_size
                ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

            if merge_context:
                # Merges two contexts by adding their embeddings, not a standard GPT behavior
                tok_emb = torch.cat(
                    [
                        self.transformer.wte(idx[:, :256])
                        + self.transformer.wte(idx[:, 256 : 256 + 256]),
                        self.transformer.wte(idx[:, 256 + 256 :]),
                    ],
                    dim=1,
                )
            else:
                tok_emb = self.transformer.wte(idx)

        if past_kv is None:
            past_length = 0
            # Empty cache for each layer
            past_kv = tuple([None] * len(self.transformer.h))
        else:
            # Infers prior sequence length from cache
            past_length = past_kv[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(
                past_length, t + past_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
            assert position_ids.shape == (1, t)

        pos_emb = self.transformer.wpe(position_ids)

        x = self.transformer.drop(tok_emb + pos_emb)

        # Prepares cache for key-value pairs if enabled
        new_kv = () if use_cache else None

        for i, (block, past_layer_kv) in enumerate(zip(self.transformer.h, past_kv)):
            x, kv = block(x, past_kv=past_layer_kv, use_cache=use_cache)
            if use_cache:
                new_kv = new_kv + (kv,)  # Accumulates new key-value pairs for caching

        x = self.transformer.ln_f(x)

        # Optimization: only computes logits for the last token, efficient for generation
        logits = self.lm_head(x[:, [-1], :])  # Preserves time dim with [-1]

        return (
            logits,
            new_kv,
        )  # Returns tuple: logits for next token, cache if requested


class NonCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=False
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class FineBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = NonCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class FineGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        del self.lm_head
        self.config = config
        self.n_codes_total = config.n_codes_total
        self.transformer = nn.ModuleDict(
            dict(
                wtes=nn.ModuleList(
                    [
                        nn.Embedding(config.input_vocab_size, config.n_embd)
                        for _ in range(config.n_codes_total)
                    ]
                ),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([FineBlock(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_heads = nn.ModuleList(
            [
                nn.Linear(config.n_embd, config.output_vocab_size, bias=False)
                for _ in range(config.n_codes_given, self.n_codes_total)
            ]
        )
        for i in range(self.n_codes_total - config.n_codes_given):
            self.transformer.wtes[i + 1].weight = self.lm_heads[i].weight

    def forward(self, pred_idx, idx):
        device = idx.device
        b, t, codes = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        assert pred_idx > 0, "cannot predict 0th codebook"
        assert codes == self.n_codes_total, (b, t, codes)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
            0
        )  # shape (1, t)

        # forward the GPT model itself
        tok_embs = [
            wte(idx[:, :, i]).unsqueeze(-1)
            for i, wte in enumerate(self.transformer.wtes)
        ]  # token embeddings of shape (b, t, n_embd)
        tok_emb = torch.cat(tok_embs, dim=-1)
        pos_emb = self.transformer.wpe(
            pos
        )  # position embeddings of shape (1, t, n_embd)
        x = tok_emb[:, :, :, : pred_idx + 1].sum(dim=-1)
        x = self.transformer.drop(x + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_heads[pred_idx - self.config.n_codes_given](x)
        return logits

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            for wte in self.transformer.wtes:
                n_params -= wte.weight.numel()
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
