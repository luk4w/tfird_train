#include <cstdint>
#include <cuda_runtime.h>

extern "C" __declspec(dllexport) void
decode_cuda(const uint8_t *input, float *output, int batch_size);

__device__ __forceinline__ uint8_t get_piece(const uint8_t *board, int sq) {
  uint8_t b = board[sq >> 1];
  return (sq & 1) ? (b >> 4) : (b & 0x0F);
}

__global__ void decode_kernel(const uint8_t *__restrict__ input,
                              float *__restrict__ output, int batch) {
  int b = blockIdx.x;
  int sq = threadIdx.x;
  if (b >= batch || sq >= 64)
    return;

#pragma unroll
  for (int c = 0; c < 19; c++)
    output[(b * 19 + c) * 64 + sq] = 0.0f;

  const uint8_t *entry = input + b * 64;

  __shared__ uint8_t board[64];
  __shared__ uint8_t ep, flags;

  if (sq == 0) {
    ep = entry[32];
    flags = entry[33];
  }

  board[sq] = get_piece(entry, sq);
  __syncthreads();

  auto write = [&](int c, float v) { output[(b * 19 + c) * 64 + sq] = v; };

  uint8_t p = board[sq];

  if (p >= 1 && p <= 12)
    write(p - 1, 1.0f);
  if (ep == sq && ep < 64)
    write(12, 1.0f);
  if (flags & 1)
    write(13, 1.0f);
  if (flags & 2)
    write(14, 1.0f);
  if (flags & 4)
    write(15, 1.0f);
  if (flags & 8)
    write(16, 1.0f);

  float w = 0.0f, b_ = 0.0f;
  int r0 = sq >> 3;
  int f0 = sq & 7;

  const int nd[8][2] = {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
                        {1, -2},  {1, 2},  {2, -1},  {2, 1}};

#pragma unroll
  for (int i = 0; i < 8; i++) {
    int r = r0 + nd[i][0];
    int f = f0 + nd[i][1];
    if (r < 0 || r > 7 || f < 0 || f > 7)
      continue;
    uint8_t q = board[r * 8 + f];
    if (q == 2)
      w += 1.0f;
    if (q == 8)
      b_ += 1.0f;
  }

  const int kd[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
                        {0, 1},   {1, -1}, {1, 0},  {1, 1}};

#pragma unroll
  for (int i = 0; i < 8; i++) {
    int r = r0 + kd[i][0];
    int f = f0 + kd[i][1];
    if (r < 0 || r > 7 || f < 0 || f > 7)
      continue;
    uint8_t q = board[r * 8 + f];
    if (q == 6)
      w += 1.0f;
    if (q == 12)
      b_ += 1.0f;
  }

  if (r0 > 0) {
    if (f0 > 0 && board[(r0 - 1) * 8 + (f0 - 1)] == 1)
      w += 1.0f;
    if (f0 < 7 && board[(r0 - 1) * 8 + (f0 + 1)] == 1)
      w += 1.0f;
  }
  if (r0 < 7) {
    if (f0 > 0 && board[(r0 + 1) * 8 + (f0 - 1)] == 7)
      b_ += 1.0f;
    if (f0 < 7 && board[(r0 + 1) * 8 + (f0 + 1)] == 7)
      b_ += 1.0f;
  }

#pragma unroll
  for (int d = 0; d < 8; d++) {
    for (int s = 1; s < 8; s++) {
      int r = r0 + kd[d][0] * s;
      int f = f0 + kd[d][1] * s;
      if (r < 0 || r > 7 || f < 0 || f > 7)
        break;
      uint8_t q = board[r * 8 + f];
      if (!q)
        continue;

      bool diag = kd[d][0] && kd[d][1];
      if (diag) {
        if (q == 3 || q == 5)
          w += 1.0f;
        if (q == 9 || q == 11)
          b_ += 1.0f;
      } else {
        if (q == 4 || q == 5)
          w += 1.0f;
        if (q == 10 || q == 11)
          b_ += 1.0f;
      }
      break;
    }
  }

  write(17, w);
  write(18, b_);
}

void decode_cuda(const uint8_t *input, float *output, int batch) {
  decode_kernel<<<batch, 64>>>(input, output, batch);
}
