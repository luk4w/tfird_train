#include <cstdint>
#include <cuda_runtime.h>
#include <cmath>

// Exporta a função decode_cuda como símbolo C puro para ser carregada via ctypes no Python
extern "C" __declspec(dllexport) void
decode_cuda(const uint8_t *input, float *output, long long *policy_out, float *value_out, int batch_size);

// Função auxiliar executada na GPU: extrai o nibble (4 bits) de uma casa específica.
// O tabuleiro canonizado começa no byte 8 do registro ChessData.
__device__ __forceinline__ uint8_t get_piece(const uint8_t *board, int sq) {
  uint8_t b = board[sq >> 1];         // sq / 2 -> byte que contém essa casa
  return (sq & 1) ? (b >> 4) : (b & 0x0F);  // escolhe o nibble correto
}

// Remapeia o lance uint16_t (promo<<12 | to<<6 | from) para o índice denso 0..4223
__device__ int32_t remap_move_cuda(uint16_t move) {
    int from_sq = move & 0x3F;
    int to_sq = (move >> 6) & 0x3F;
    int promo = (move >> 12) & 0x0F;

    if (promo == 0) {
        return to_sq * 64 + from_sq;
    } else {
        int from_file = from_sq & 7;
        int to_file = to_sq & 7;
        int direction = to_file - from_file + 1; // 0: esq, 1: frente, 2: dir
        return 4096 + from_file * 12 + direction * 4 + (promo - 1);
    }
}

// Kernel CUDA: cada bloco processa uma posição do batch; cada thread processa uma casa.
__global__ void decode_kernel(const uint8_t *__restrict__ input,
                              float *__restrict__ output,
                              long long *__restrict__ policy_out,
                              float *__restrict__ value_out,
                              int batch) {
  int b  = blockIdx.x;   // índice da posição no batch
  int sq = threadIdx.x;  // índice da casa no tabuleiro (0-63)

  if (b >= batch || sq >= 64)
    return;

  // Zera os 19 canais da casa sq para essa posição
#pragma unroll
  for (int c = 0; c < 19; c++)
    output[(b * 19 + c) * 64 + sq] = 0.0f;

  // Ponteiro para o início do registro de 64 bytes
  const uint8_t *entry = input + b * 64;

  __shared__ uint8_t board[64]; 
  __shared__ uint8_t ep;        
  __shared__ uint8_t rK, rQ, rk, rq;

  if (sq == 0) {
    // Novas offsets baseadas na struct ChessData do miner.cpp
    // zobrist: 0-7, data: 8-39, move: 40-41, cp: 42-43, ...
    ep = entry[50];
    rK = entry[46];
    rQ = entry[47];
    rk = entry[48];
    rq = entry[49];

    // Extração de targets (apenas na thread 0 do bloco)
    uint16_t move_raw = *(uint16_t*)(entry + 40);
    int16_t cp_raw = *(int16_t*)(entry + 42);

    policy_out[b] = (long long)remap_move_cuda(move_raw);
    value_out[b] = tanhf((float)cp_raw / 600.0f);
  }

  // Board começa no byte 8
  board[sq] = get_piece(entry + 8, sq);

  __syncthreads();

  auto write = [&](int c, float v) { output[(b * 19 + c) * 64 + sq] = v; };

  uint8_t p = board[sq];
  if (p >= 1 && p <= 12)
    write(p - 1, 1.0f);

  if (ep == sq && ep < 64)
    write(12, 1.0f);

  if (rK) write(13, 1.0f);
  if (rQ) write(14, 1.0f);
  if (rk) write(15, 1.0f);
  if (rq) write(16, 1.0f);

  // Canais 17-18: mapa de influência
  float w  = 0.0f;
  float b_ = 0.0f;
  int r0 = sq >> 3;
  int f0 = sq & 7;

  const int nd[8][2] = {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
                        {1, -2},  {1, 2},  {2, -1},  {2, 1}};
#pragma unroll
  for (int i = 0; i < 8; i++) {
    int r = r0 + nd[i][0];
    int f = f0 + nd[i][1];
    if (r < 0 || r > 7 || f < 0 || f > 7) continue;
    uint8_t q = board[r * 8 + f];
    if (q == 2) w  += 1.0f;
    if (q == 8) b_ += 1.0f;
  }

  const int kd[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
                        {0, 1},   {1, -1}, {1, 0},  {1, 1}};
#pragma unroll
  for (int i = 0; i < 8; i++) {
    int r = r0 + kd[i][0];
    int f = f0 + kd[i][1];
    if (r < 0 || r > 7 || f < 0 || f > 7) continue;
    uint8_t q = board[r * 8 + f];
    if (q == 6)  w  += 1.0f;
    if (q == 12) b_ += 1.0f;
  }

  if (r0 > 0) {
    if (f0 > 0 && board[(r0 - 1) * 8 + (f0 - 1)] == 1) w  += 1.0f;
    if (f0 < 7 && board[(r0 - 1) * 8 + (f0 + 1)] == 1) w  += 1.0f;
  }
  if (r0 < 7) {
    if (f0 > 0 && board[(r0 + 1) * 8 + (f0 - 1)] == 7) b_ += 1.0f;
    if (f0 < 7 && board[(r0 + 1) * 8 + (f0 + 1)] == 7) b_ += 1.0f;
  }

#pragma unroll
  for (int d = 0; d < 8; d++) {
    for (int s = 1; s < 8; s++) {
      int r = r0 + kd[d][0] * s;
      int f = f0 + kd[d][1] * s;
      if (r < 0 || r > 7 || f < 0 || f > 7) break;
      uint8_t q = board[r * 8 + f];
      if (!q) continue;

      bool diag = kd[d][0] && kd[d][1];
      if (diag) {
        if (q == 3 || q == 5) w  += 1.0f;
        if (q == 9 || q == 11) b_ += 1.0f;
      } else {
        if (q == 4 || q == 5) w  += 1.0f;
        if (q == 10 || q == 11) b_ += 1.0f;
      }
      break;
    }
  }

  write(17, w);
  write(18, b_);
}

// Função host (CPU) que lança o kernel na GPU.
void decode_cuda(const uint8_t *input, float *output, long long *policy_out, float *value_out, int batch) {
  decode_kernel<<<batch, 64>>>(input, output, policy_out, value_out, batch);
}

