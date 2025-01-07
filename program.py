import torch

import triton
import triton.language as tl

@triton.jit
def _attn_fwd(
    Q, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,
    M, # BATCH_SIZE, NUM_HEADS, SEQ_LEN
    O, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr, # We did not pass the BLOCK_SIZE_Q and BLOCK_SIZE_KV when calling function since 
    BLOCK_SIZE_KV: tl.constexpr, # they will be passed when we apply the Auto Tuning decorator
    # they represent how many queries and keys, values we want to group together when doing block matrix multiplication
    STAAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)
    # We need to understand on which part this function should work on
    # for that we need to use the grid passed when function called [grid]

    # This indicates which block in the sequence length to process
    block_index_q = tl.program_id(0)

    # This indicates which head and batch to process. 
    # Each program is associated with a single head of a single batch
    index_batch_head = tl.program_id(1)
    # This indicates which batch this program is associated with (each batch has NUM_HEADS heads)
    index_batch = index_batch_head // NUM_HEADS
    # This indicates the position of head in the batch
    index_head = index_batch_head % NUM_HEADS


class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        """
        ctx: context. which will be storage for activations which we will need in backward pass
        """
        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        O = torch.empty_like(Q)
        stage = 3 if causal else 1

        # We want to parallelize within batch and also each head and within QUERY BLOCK
        grid = lambda args: (
            # ceil(SEQ_LEN / BLOCK_SIZE_Q) = How many blocks of Q we have
            triton.cdiv(SEQ_LEN, args['BLOCK_SIZE_Q']), # Which group of queries are we going to work with?
            BATCH_SIZE * NUM_HEADS, # Which head of which batch element we are working with?
            1, # Z in the CUDA launch grid
        )
        # Number of parallel programs: (BATCH_SIZE * NUM_HEADS * NUM_BLOCKS_Q)

        # M is the logsumexp for the backward pass, one for each query
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        ) # it's sum of normalization factor and maximum for each row(used in softmax)

        # Since we only get a pointer to starting location in memory of Q,K,V
        # we need to have strides for each matrix to be able to do the indexing
        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M,
            O=O, # Where kernel must fill computed att scores

            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),

            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),

            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),

            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),

            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal

        return O




    

def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device='cuda'
            ).normal_(0.0, std=0.5).requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device='cuda'
            ).normal_(0.0, std=0.5).requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device='cuda'
            ).normal_(0.0, std=0.5).requires_grad_()
    )
    softmax_scale = 1 / (HEAD_DIM ** 0.5)
    dO = torch.rand_like(Q) # Need for backwards pass

    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device='cuda'))
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    if causal:
        P[:, :, MASK == 0] = float('-inf')
    P = torch.softmax(P.float(), dim=-1).half()

    ref_O = torch.matmul(P, V)
    ref_O.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None

    # triton implementation
    tri_out = TritonAttentino.apply(Q, K, V, causal, softmax_scale).half()
    tri_out.backward(dO)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None

    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)

