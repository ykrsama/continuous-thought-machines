import torch

from models.ctm_attention import CTMAttention


def test_ctm_attention_shapes_and_state_propagation():
    torch.manual_seed(0)

    B, L_q, L_kv = 2, 5, 7
    embed_dim = 64
    num_heads = 4
    memory_length = 3
    n_synch_qkv = 8
    n_synch_o = 8
    K_ticks = 3

    module = CTMAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        memory_length=memory_length,
d_model_qkv=16,
        d_model_o=16,
        n_synch_qkv=n_synch_qkv,
        n_synch_o=n_synch_o,
        dropout=0.0,
    )
    module.eval()

    query = torch.randn(B, L_q, embed_dim)
    key = torch.randn(B, L_kv, embed_dim)
    value = torch.randn(B, L_kv, embed_dim)

    sync_size_qkv = n_synch_qkv * (n_synch_qkv + 1) // 2
    sync_size_o = n_synch_o * (n_synch_o + 1) // 2

    state = None
    last_out = None
    for tick in range(K_ticks):
        out, weights, state = module(query, key, value, state=state)
        assert out.shape == (B, L_q, embed_dim)
        assert weights.shape == (B, num_heads, L_q, L_kv)
        assert torch.isfinite(out).all()
        assert torch.isfinite(weights).all()

        # State shapes
        assert state['trace_q'].shape == (B, L_q, 16, memory_length)
        assert state['trace_k'].shape == (B, L_kv, 16, memory_length)
        assert state['trace_v'].shape == (B, L_kv, 16, memory_length)
        assert state['trace_o'].shape == (B, L_q, 16, memory_length)
        assert state['decay_alpha_q'].shape == (B, L_q, sync_size_qkv)
        assert state['decay_beta_q'].shape == (B, L_q, sync_size_qkv)
        assert state['decay_alpha_k'].shape == (B, L_kv, sync_size_qkv)
        assert state['decay_beta_k'].shape == (B, L_kv, sync_size_qkv)
        assert state['decay_alpha_v'].shape == (B, L_kv, sync_size_qkv)
        assert state['decay_beta_v'].shape == (B, L_kv, sync_size_qkv)
        assert state['decay_alpha_o'].shape == (B, L_q, sync_size_o)
        assert state['decay_beta_o'].shape == (B, L_q, sync_size_o)

        # Output should change between ticks (state is being used)
        if last_out is not None:
            assert not torch.allclose(out, last_out)
        last_out = out

    # Attention weights should sum to ~1 over the key dimension
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_ctm_attention_backward():
    torch.manual_seed(1)
    B, L_q, L_kv = 2, 4, 6
    embed_dim = 32
    module = CTMAttention(
        embed_dim=embed_dim,
        num_heads=4,
        memory_length=2,
        d_model_qkv=12,
        d_model_o=12,
        n_synch_qkv=6,
        n_synch_o=6,
    )

    query = torch.randn(B, L_q, embed_dim, requires_grad=True)
    key = torch.randn(B, L_kv, embed_dim)
    value = torch.randn(B, L_kv, embed_dim)

    state = None
    total = 0.0
    for _ in range(2):
        out, _, state = module(query, key, value, state=state)
        total = total + out.sum()

    total.backward()
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()
    # At least one model parameter should have a non-zero gradient.
    has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in module.parameters())
    assert has_grad
