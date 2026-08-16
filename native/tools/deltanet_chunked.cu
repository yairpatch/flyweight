// Prototype source for the chunked WY-representation gated DeltaNet.
// Validated through native/tools/kernel_harness.py, then embedded into
// colibri_v2_native_kernels.hpp.
//
// The sequential kernels (qwen_delta_recurrent{,_split,_rows,_chunk}) walk one
// token at a time with the state matrix live in registers, so a prefill chunk
// of L tokens costs L serial steps across only `value_heads` blocks.  The
// chunked form rewrites the same recurrence
//
//     S_i = d_i (I - b_i k_i k_i^T) S_{i-1} + b_i k_i v_i^T
//
// so that everything inside a chunk of 64 tokens is matrix work and only the
// chunk-to-chunk state hand-off stays serial, cutting the critical path to
// L/64 steps and widening the grid to chunks x heads.
//
// Derivation.  Let g_i = log d_i and G_i the inclusive cumulative sum of g
// within the chunk (G anchored at 0 before the chunk's first token).  Writing
// S_i = exp(G_i) * Shat_i turns the recurrence into
//
//     Shat_i = Shat_{i-1} + k_i w_i^T,   w_i = exp(-G_i) beta_i v_i
//                                             - Shat_{i-1}^T (beta_i k_i)
//
// and substituting w_i = exp(-G_i) omega_i keeps every factor bounded by 1:
//
//     omega_i + sum_{j<i} A_ij omega_j = beta_i v_i - exp(G_i) (beta_i k_i)^T S_start
//     A_ij = exp(G_i - G_j) * beta_i * (k_i . k_j)      for j < i
//
// With T = (I + A)^-1 (unit lower triangular), Wm = T * diag(exp(G) beta) K and
// U = T * diag(beta) V, both computable without the state:
//
//     Omega   = U - Wm * S_start
//     S_end   = exp(G_last) * S_start + Ktilde^T * Omega,  Ktilde_j = exp(G_last - G_j) k_j
//     core_i  = S_start^T (exp(G_i) q_i) + sum_{j<=i} P_ij * omega_j
//     P_ij    = exp(G_i - G_j) * (q_i . k_j)            for j <= i
//
// q and k are the L2-normalized projections; q additionally carries the
// 1/sqrt(head_dim) scale, exactly as the sequential kernels apply it.
//
// Fixed to head_dim == 128 and chunk == 64; the host keeps the sequential
// kernels for decode and for any other geometry.  Shared memory per block is
// held at or below 32 KB because the driver never opts in past the 48 KB
// default.

#define DELTA_CHUNK 64
#define DELTA_DIM 128
#define DELTA_SLAB 32

// Pass 1: per-token scalars (log-decay cumsum, beta, normalization) and the two
// 64x64 score matrices.  Grid (chunks, value_heads), block 256.
extern "C" __global__
void qwen_delta_wy_scores(
    const float* convolved, const float* beta_logits, const float* decay_logits,
    const float* decay_coefficients, const float* dt_bias,
    float* attn, float* pmat, float* g_cumsum, float* beta_out,
    float* qinv_out, float* kinv_out,
    const int rows, const int key_heads, const int value_heads
) {
    const int chunk = blockIdx.x;
    const int head = blockIdx.y;
    const int base = chunk * DELTA_CHUNK;
    const int valid = rows - base < DELTA_CHUNK ? rows - base : DELTA_CHUNK;
    if (valid <= 0) return;
    const int key_head = head % key_heads;
    const int total_key = key_heads * DELTA_DIM;
    const int key_off = key_head * DELTA_DIM;
    const long long stride = (long long)(total_key * 2 + value_heads * DELTA_DIM);
    const int tid = threadIdx.x;

    // Slabs are stored dimension-major so a thread's four tile rows are
    // contiguous in shared memory; the +1 pad keeps the strided reads off a
    // single bank.
    __shared__ float qs[DELTA_SLAB][DELTA_CHUNK + 1];
    __shared__ float ks[DELTA_SLAB][DELTA_CHUNK + 1];
    __shared__ float gcum[DELTA_CHUNK];
    __shared__ float betas[DELTA_CHUNK];
    __shared__ float qinv[DELTA_CHUNK];
    __shared__ float kinv[DELTA_CHUNK];

    // One warp per token: L2 norms, beta, and the per-token log decay.
    for (int t = tid >> 5; t < DELTA_CHUNK; t += 8) {
        const int lane = tid & 31;
        if (t >= valid) {
            if (lane == 0) { gcum[t] = 0.0f; betas[t] = 0.0f; qinv[t] = 0.0f; kinv[t] = 0.0f; }
            continue;
        }
        const float* row = convolved + (long long)(base + t) * stride;
        float key_square = 0.0f, query_square = 0.0f;
        for (int d = lane; d < DELTA_DIM; d += 32) {
            const float k = row[total_key + key_off + d];
            const float q = row[key_off + d];
            key_square += k * k;
            query_square += q * q;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            key_square += __shfl_down_sync(0xffffffff, key_square, offset);
            query_square += __shfl_down_sync(0xffffffff, query_square, offset);
        }
        if (lane == 0) {
            qinv[t] = rsqrtf(query_square + 1.0e-6f) * rsqrtf((float)DELTA_DIM);
            kinv[t] = rsqrtf(key_square + 1.0e-6f);
            const long long scalar = (long long)(base + t) * value_heads + head;
            betas[t] = 1.0f / (1.0f + expf(-beta_logits[scalar]));
            const float softplus_input = decay_logits[scalar] + dt_bias[head];
            const float softplus = softplus_input > 20.0f
                ? softplus_input : log1pf(expf(softplus_input));
            gcum[t] = decay_coefficients[head] * softplus;
        }
    }
    __syncthreads();
    // Inclusive prefix sum of the log decays. 64 serial adds on one thread is
    // cheaper than the barriers a parallel scan of this length would cost.
    if (tid == 0) {
        float running = 0.0f;
        for (int t = 0; t < valid; ++t) { running += gcum[t]; gcum[t] = running; }
    }
    __syncthreads();
    if (tid < valid) {
        const long long scalar = (long long)(base + tid) * value_heads + head;
        g_cumsum[scalar] = gcum[tid];
        beta_out[scalar] = betas[tid];
        // Published so the solve and the state pass do not each repeat a full
        // pass over the projections just to recover these two scalars.
        qinv_out[scalar] = qinv[tid];
        kinv_out[scalar] = kinv[tid];
    }

    // P = Qn Kn^T and A = Kn Kn^T, both 64x64 over a 128-deep reduction, as one
    // blocked GEMM: 256 threads in a 16x16 arrangement, each holding a 4x4 tile
    // of both outputs. Twelve shared loads feed 32 FMAs per step, where the
    // dot-product-per-thread form managed one load per FMA.
    const int tx = tid & 15, ty = tid >> 4;
    const int i0 = ty * 4, j0 = tx * 4;
    float pacc[4][4] = {}, aacc[4][4] = {};
    for (int slab = 0; slab < DELTA_DIM; slab += DELTA_SLAB) {
        for (int index = tid; index < DELTA_SLAB * DELTA_CHUNK; index += blockDim.x) {
            const int t = index / DELTA_SLAB, d = index % DELTA_SLAB;
            float q = 0.0f, k = 0.0f;
            if (t < valid) {
                const float* row = convolved + (long long)(base + t) * stride;
                // Fold the normalizers in on the way to shared memory.
                q = row[key_off + slab + d] * qinv[t];
                k = row[total_key + key_off + slab + d] * kinv[t];
            }
            qs[d][t] = q;
            ks[d][t] = k;
        }
        __syncthreads();
        for (int d = 0; d < DELTA_SLAB; ++d) {
            float column[4], query_row[4], key_row[4];
            #pragma unroll
            for (int c = 0; c < 4; ++c) column[c] = ks[d][j0 + c];
            #pragma unroll
            for (int r = 0; r < 4; ++r) { query_row[r] = qs[d][i0 + r]; key_row[r] = ks[d][i0 + r]; }
            #pragma unroll
            for (int r = 0; r < 4; ++r)
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    pacc[r][c] += query_row[r] * column[c];
                    aacc[r][c] += key_row[r] * column[c];
                }
        }
        __syncthreads();
    }

    const long long mat = ((long long)chunk * value_heads + head)
        * (DELTA_CHUNK * DELTA_CHUNK);
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int i = i0 + r;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int j = j0 + c;
            // Tail rows and columns of a partial chunk are zeroed here so the
            // solve and the state pass can run the full 64 without branching.
            const float decay = expf(gcum[i] - gcum[j]);
            const bool live = i < valid && j < valid;
            pmat[mat + i * DELTA_CHUNK + j] =
                (live && j <= i) ? pacc[r][c] * decay : 0.0f;
            attn[mat + i * DELTA_CHUNK + j] =
                (live && j < i) ? aacc[r][c] * betas[i] * decay : 0.0f;
        }
    }
}

// Pass 2: invert (I + A) and apply it to the two right-hand sides, producing
// Wm (64 x 128, against the state) and U (64 x 128, the state-free part of
// Omega).  Grid (chunks, value_heads), block 256.
extern "C" __global__
void qwen_delta_wy_solve(
    const float* convolved, const float* attn, const float* g_cumsum,
    const float* beta_in, const float* kinv_in, float* w_rows, float* u_rows,
    const int rows, const int key_heads, const int value_heads
) {
    const int chunk = blockIdx.x;
    const int head = blockIdx.y;
    const int base = chunk * DELTA_CHUNK;
    const int valid = rows - base < DELTA_CHUNK ? rows - base : DELTA_CHUNK;
    if (valid <= 0) return;
    const int key_head = head % key_heads;
    const int total_key = key_heads * DELTA_DIM;
    const int key_off = key_head * DELTA_DIM;
    const long long stride = (long long)(total_key * 2 + value_heads * DELTA_DIM);
    const int tid = threadIdx.x;

    __shared__ float tri[DELTA_CHUNK][DELTA_CHUNK];   // A, overwritten by T
    __shared__ float slab[DELTA_CHUNK][DELTA_CHUNK];  // staged right-hand side
    __shared__ float kinv[DELTA_CHUNK];

    const long long mat = ((long long)chunk * value_heads + head)
        * (DELTA_CHUNK * DELTA_CHUNK);
    for (int index = tid; index < DELTA_CHUNK * DELTA_CHUNK; index += blockDim.x)
        tri[index / DELTA_CHUNK][index % DELTA_CHUNK] = attn[mat + index];
    if (tid < valid)
        kinv[tid] = kinv_in[(long long)(base + tid) * value_heads + head];
    __syncthreads();

    // Forward substitution, one row at a time and in place. Row i reads A[i][m]
    // before it is overwritten, and rows m < i already hold T. Every thread
    // reaches both barriers so the row swap is ordered block-wide.
    for (int i = 0; i < valid; ++i) {
        float sum = 0.0f;
        if (tid < DELTA_CHUNK) {
            sum = (tid == i) ? 1.0f : 0.0f;
            for (int m = 0; m < i; ++m) sum -= tri[i][m] * tri[m][tid];
        }
        __syncthreads();
        if (tid < DELTA_CHUNK) tri[i][tid] = sum;
        __syncthreads();
    }

    // Apply T to both right-hand sides, 64 columns at a time.
    //   Wm[i][d] = sum_m T[i][m] * exp(G_m) * beta_m * kn_m[d]
    //   U[i][d]  = sum_m T[i][m] * beta_m * v_m[d]
    for (int side = 0; side < 2; ++side) {
        for (int slab_base = 0; slab_base < DELTA_DIM; slab_base += DELTA_CHUNK) {
            for (int index = tid; index < DELTA_CHUNK * DELTA_CHUNK; index += blockDim.x) {
                const int m = index / DELTA_CHUNK, d = index % DELTA_CHUNK;
                float value = 0.0f;
                if (m < valid) {
                    const long long scalar = (long long)(base + m) * value_heads + head;
                    const float* row = convolved + (long long)(base + m) * stride;
                    const float beta = beta_in[scalar];
                    if (side == 0) {
                        value = row[total_key + key_off + slab_base + d] * kinv[m]
                            * beta * expf(g_cumsum[scalar]);
                    } else {
                        value = row[total_key * 2 + head * DELTA_DIM + slab_base + d] * beta;
                    }
                }
                slab[m][d] = value;
            }
            __syncthreads();
            for (int index = tid; index < DELTA_CHUNK * DELTA_CHUNK; index += blockDim.x) {
                const int i = index / DELTA_CHUNK, d = index % DELTA_CHUNK;
                float sum = 0.0f;
                if (i < valid)
                    for (int m = 0; m <= i; ++m) sum += tri[i][m] * slab[m][d];
                if (i < valid) {
                    float* target = side == 0 ? w_rows : u_rows;
                    target[((long long)(base + i) * value_heads + head) * DELTA_DIM
                           + slab_base + d] = sum;
                }
            }
            __syncthreads();
        }
    }
}

// Pass 3: the only serial part. Walks chunks in order, carrying the state, and
// emits the un-normalized core outputs. The value dimension is independent
// across the whole recurrence, so the grid splits it into tiles and each block
// owns DELTA_TILE value columns of one head's state.
// Grid (value_heads, DELTA_DIM / DELTA_TILE), block 256.
#define DELTA_TILE 32

extern "C" __global__
void qwen_delta_state_pass(
    const float* convolved, const float* pmat, const float* g_cumsum,
    const float* qinv_in, const float* kinv_in,
    const float* w_rows, const float* u_rows, float* state, float* core,
    const int rows, const int key_heads, const int value_heads
) {
    const int head = blockIdx.x;
    const int tile = blockIdx.y * DELTA_TILE;
    const int key_head = head % key_heads;
    const int total_key = key_heads * DELTA_DIM;
    const int key_off = key_head * DELTA_DIM;
    const long long stride = (long long)(total_key * 2 + value_heads * DELTA_DIM);
    const int tid = threadIdx.x;
    const int chunks = (rows + DELTA_CHUNK - 1) / DELTA_CHUNK;

    __shared__ float carried[DELTA_DIM][DELTA_TILE];   // 16 KB
    __shared__ float omega[DELTA_CHUNK][DELTA_TILE];   // 8 KB
    // One staging buffer, re-carved per stage: the three matrix products need
    // different operands but never two of them at once, and a dedicated buffer
    // for each would not fit under the 48 KB the driver allows.
    __shared__ float scratch[DELTA_CHUNK * (DELTA_CHUNK + 1)];   // 16.6 KB
    // Per-token scales, hoisted out of the matrix loops: both the query and the
    // key factor depend only on the token, so computing them per output element
    // would run one expf per MAC.
    __shared__ float qscale[DELTA_CHUNK];   // qinv_i * exp(G_i)
    __shared__ float kscale[DELTA_CHUNK];   // kinv_j * exp(G_last - G_j)

    // Output tiles. Every stage gives a thread several outputs so the operand
    // loads amortize: the one-output-per-thread form spent two loads per FMA.
    const int col = (tid & 15) * 2;          // 2 value columns
    const int band = tid >> 4;               // 16 bands of rows
    const int row4 = band * 4;               // 4 token rows  (64-row stages)
    const int row8 = band * 8;               // 8 state rows  (128-row stage)

    for (int index = tid; index < DELTA_DIM * DELTA_TILE; index += blockDim.x) {
        const int m = index / DELTA_TILE, d = index % DELTA_TILE;
        carried[m][d] = state[((long long)head * DELTA_DIM + m) * DELTA_DIM + tile + d];
    }

    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int base = chunk * DELTA_CHUNK;
        const int valid = rows - base < DELTA_CHUNK ? rows - base : DELTA_CHUNK;
        const long long mat = ((long long)chunk * value_heads + head)
            * (DELTA_CHUNK * DELTA_CHUNK);
        const float g_last =
            g_cumsum[(long long)(base + valid - 1) * value_heads + head];
        // Fold the decay into the published normalizers; both scales depend
        // only on the token, so they must stay out of the matrix loops.
        if (tid < valid) {
            const long long scalar = (long long)(base + tid) * value_heads + head;
            const float g = g_cumsum[scalar];
            qscale[tid] = qinv_in[scalar] * expf(g);
            kscale[tid] = kinv_in[scalar] * expf(g_last - g);
        }
        __syncthreads();

        // Wm * S_start and Qtilde * S_start share the same S operand and the same
        // shape, so they run as one pass over the reduction dimension and read
        // the state tile once for both.
        float omega_acc[4][2] = {}, core_acc[4][2] = {};
        for (int slab = 0; slab < DELTA_DIM; slab += DELTA_SLAB) {
            // Consecutive threads walk the reduction dimension, which is the
            // contiguous axis of both sources.
            for (int index = tid; index < DELTA_SLAB * DELTA_CHUNK; index += blockDim.x) {
                const int i = index / DELTA_SLAB, m = index % DELTA_SLAB;
                float w = 0.0f, q = 0.0f;
                if (i < valid) {
                    w = w_rows[((long long)(base + i) * value_heads + head) * DELTA_DIM
                               + slab + m];
                    q = convolved[(long long)(base + i) * stride + key_off + slab + m]
                        * qscale[i];
                }
                scratch[m * (DELTA_CHUNK + 1) + i] = w;
                scratch[DELTA_SLAB * (DELTA_CHUNK + 1) + m * (DELTA_CHUNK + 1) + i] = q;
            }
            __syncthreads();
            for (int m = 0; m < DELTA_SLAB; ++m) {
                float state_column[2], w_row[4], q_row[4];
                #pragma unroll
                for (int c = 0; c < 2; ++c) state_column[c] = carried[slab + m][col + c];
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    w_row[r] = scratch[m * (DELTA_CHUNK + 1) + row4 + r];
                    q_row[r] = scratch[DELTA_SLAB * (DELTA_CHUNK + 1)
                                       + m * (DELTA_CHUNK + 1) + row4 + r];
                }
                #pragma unroll
                for (int r = 0; r < 4; ++r)
                    #pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        omega_acc[r][c] += w_row[r] * state_column[c];
                        core_acc[r][c] += q_row[r] * state_column[c];
                    }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = row4 + r;
            #pragma unroll
            for (int c = 0; c < 2; ++c)
                omega[i][col + c] = i < valid
                    ? u_rows[((long long)(base + i) * value_heads + head) * DELTA_DIM
                             + tile + col + c] - omega_acc[r][c]
                    : 0.0f;
        }
        __syncthreads();

        // core_i also takes sum_{j<=i} P_ij omega_j. Both terms are against the
        // incoming state, so this must complete before the state update.
        for (int index = tid; index < DELTA_CHUNK * DELTA_CHUNK; index += blockDim.x)
            scratch[(index / DELTA_CHUNK) * (DELTA_CHUNK + 1) + index % DELTA_CHUNK] =
                pmat[mat + index];
        __syncthreads();
        for (int j = 0; j < DELTA_CHUNK; ++j) {
            float omega_column[2], p_row[4];
            #pragma unroll
            for (int c = 0; c < 2; ++c) omega_column[c] = omega[j][col + c];
            #pragma unroll
            for (int r = 0; r < 4; ++r)
                p_row[r] = scratch[(row4 + r) * (DELTA_CHUNK + 1) + j];
            #pragma unroll
            for (int r = 0; r < 4; ++r)
                #pragma unroll
                for (int c = 0; c < 2; ++c) core_acc[r][c] += p_row[r] * omega_column[c];
        }
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = row4 + r;
            if (i >= valid) continue;
            #pragma unroll
            for (int c = 0; c < 2; ++c)
                core[((long long)(base + i) * value_heads + head) * DELTA_DIM
                     + tile + col + c] = core_acc[r][c];
        }
        __syncthreads();

        // S_end = exp(G_last) S_start + Ktilde^T Omega, over two halves of the
        // 128 state rows so the staged Ktilde slab reuses the same scratch.
        const float closing = expf(g_last);
        for (int half = 0; half < 2; ++half) {
            const int m_base = half * DELTA_CHUNK;
            for (int index = tid; index < DELTA_CHUNK * DELTA_CHUNK; index += blockDim.x) {
                const int j = index / DELTA_CHUNK, m = index % DELTA_CHUNK;
                scratch[j * (DELTA_CHUNK + 1) + m] = j < valid
                    ? convolved[(long long)(base + j) * stride + total_key + key_off
                                + m_base + m] * kscale[j]
                    : 0.0f;
            }
            __syncthreads();
            // This half owns state rows [m_base, m_base + 64); each thread takes
            // four of them against its two value columns.
            float acc[4][2] = {};
            for (int j = 0; j < DELTA_CHUNK; ++j) {
                float omega_column[2], k_row[4];
                #pragma unroll
                for (int c = 0; c < 2; ++c) omega_column[c] = omega[j][col + c];
                #pragma unroll
                for (int r = 0; r < 4; ++r)
                    k_row[r] = scratch[j * (DELTA_CHUNK + 1) + row4 + r];
                #pragma unroll
                for (int r = 0; r < 4; ++r)
                    #pragma unroll
                    for (int c = 0; c < 2; ++c) acc[r][c] += k_row[r] * omega_column[c];
            }
            __syncthreads();
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int m = m_base + row4 + r;
                #pragma unroll
                for (int c = 0; c < 2; ++c)
                    carried[m][col + c] = carried[m][col + c] * closing + acc[r][c];
            }
            __syncthreads();
        }
    }

    for (int index = tid; index < DELTA_DIM * DELTA_TILE; index += blockDim.x) {
        const int m = index / DELTA_TILE, d = index % DELTA_TILE;
        state[((long long)head * DELTA_DIM + m) * DELTA_DIM + tile + d] = carried[m][d];
    }
}

// Pass 4: the per-token epilogue the sequential kernels fold into their inner
// loop -- RMS norm across the head's value dimension, the learned weight, and
// the SiLU gate. Grid (rows, value_heads), block DELTA_DIM.
extern "C" __global__
void qwen_delta_norm_gate(
    const float* core, const float* gates, const float* norm_weights,
    float* output, const int value_heads, const float epsilon
) {
    const int token = blockIdx.x;
    const int head = blockIdx.y;
    const int lane = threadIdx.x;
    const long long offset =
        ((long long)token * value_heads + head) * DELTA_DIM + lane;
    const float value = core[offset];

    __shared__ float sums[4];
    float square = value * value;
    for (int step = 16; step > 0; step >>= 1)
        square += __shfl_down_sync(0xffffffff, square, step);
    if ((lane & 31) == 0) sums[lane >> 5] = square;
    __syncthreads();
    const float total = sums[0] + sums[1] + sums[2] + sums[3];
    const float inverse_rms = rsqrtf(total / (float)DELTA_DIM + epsilon);

    const float gate = gates[offset];
    output[offset] = value * inverse_rms * norm_weights[lane]
        * gate / (1.0f + expf(-fminf(80.0f, fmaxf(-80.0f, gate))));
}
