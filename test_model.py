"""
Inferência interativa do ChessNet treinado — versão CPU pura.

Não usa decoder.dll: replica o kernel CUDA em numpy puro.
Roda em paralelo com o treino sem brigar pela GPU.

Uso:
    python test_model.py                              # checkpoint_latest.pt
    python test_model.py --checkpoint checkpoint_epoch1_ft.pt
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse

from train import ChessNet

DEVICE = torch.device('cpu')

# ----------------------------------------------------------------------
# Tabelas
# ----------------------------------------------------------------------
CHAR_TO_NIBBLE = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
}
FLIP_PIECE  = [0, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
PROMO_CHARS = {1: 'n', 2: 'b', 3: 'r', 4: 'q'}

# Offsets dos 8 knight moves e 8 king moves (idênticos ao decoder.cu)
KNIGHT_DELTAS = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
KING_DELTAS   = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

# ----------------------------------------------------------------------
# Parser FEN → buffer compacto de 64 bytes (layout do miner.cpp/decoder.cu)
# ----------------------------------------------------------------------
def parse_fen(fen):
    parts = fen.split()
    board_str, turn, castling, ep_str = parts[0], parts[1], parts[2], parts[3]

    data = bytearray(64)

    # 1. Tabuleiro (bytes 8..39)
    rank, file = 7, 0
    for c in board_str:
        if c == '/':
            rank -= 1
            file = 0
        elif c.isdigit():
            file += int(c)
        else:
            sq = rank * 8 + file
            piece = CHAR_TO_NIBBLE[c]
            idx = 8 + (sq >> 1)
            if sq & 1:
                data[idx] |= piece << 4
            else:
                data[idx] |= piece
            file += 1

    flip = (turn == 'b')

    # 2. Flip visual (se vez das pretas, vira tabuleiro e inverte cores)
    if flip:
        tmp = bytes(data[8:40])
        for i in range(8, 40):
            data[i] = 0
        for sq in range(64):
            byte_idx = sq >> 1
            b = tmp[byte_idx]
            piece = (b >> 4) if (sq & 1) else (b & 0x0F)
            if piece == 0:
                continue
            t = sq ^ 56
            fp = FLIP_PIECE[piece]
            idx = 8 + (t >> 1)
            if t & 1:
                data[idx] |= fp << 4
            else:
                data[idx] |= fp

    # 3. Roques (bytes 46..49)
    cK = cQ = ck = cq = 0
    for c in castling:
        if   c == 'K': cK = 1
        elif c == 'Q': cQ = 1
        elif c == 'k': ck = 1
        elif c == 'q': cq = 1
    if flip:
        data[46], data[47], data[48], data[49] = ck, cq, cK, cQ
    else:
        data[46], data[47], data[48], data[49] = cK, cQ, ck, cq

    # 4. En passant (byte 50)
    if ep_str == '-':
        data[50] = 255
    else:
        ep = (ord(ep_str[1]) - ord('1')) * 8 + (ord(ep_str[0]) - ord('a'))
        data[50] = (ep ^ 56) if flip else ep

    return bytes(data), flip


def decode_cpu(compact):
    """
    Replica decoder.cu em CPU puro.
    Entrada: buffer de 64 bytes (uma posição).
    Saída: ndarray float32 (19, 8, 8).
    """
    out = np.zeros((19, 64), dtype=np.float32)

    # Extrai tabuleiro (bytes 8..39) → array de 64 nibbles
    board = np.zeros(64, dtype=np.uint8)
    for sq in range(64):
        b = compact[8 + (sq >> 1)]
        board[sq] = (b >> 4) if (sq & 1) else (b & 0x0F)

    # Metadados (bytes 46..50)
    rK = compact[46]
    rQ = compact[47]
    rk = compact[48]
    rq = compact[49]
    ep = compact[50]

    # Canais 0..11 — peças (one-hot por classe)
    for sq in range(64):
        p = int(board[sq])
        if 1 <= p <= 12:
            out[p - 1, sq] = 1.0

    # Canal 12 — en passant
    if ep < 64:
        out[12, ep] = 1.0

    # Canais 13..16 — flags de roque (propagadas para todas as casas)
    if rK: out[13, :] = 1.0
    if rQ: out[14, :] = 1.0
    if rk: out[15, :] = 1.0
    if rq: out[16, :] = 1.0

    # Canais 17..18 — mapa de influência (ataques w/b sobre cada casa)
    for sq in range(64):
        r0, f0 = sq >> 3, sq & 7
        w = 0.0
        b_ = 0.0

        # Cavalos (peças 2=N, 8=n)
        for dr, df in KNIGHT_DELTAS:
            r, f = r0 + dr, f0 + df
            if 0 <= r <= 7 and 0 <= f <= 7:
                q = board[r * 8 + f]
                if q == 2:  w  += 1.0
                if q == 8:  b_ += 1.0

        # Reis (6=K, 12=k)
        for dr, df in KING_DELTAS:
            r, f = r0 + dr, f0 + df
            if 0 <= r <= 7 and 0 <= f <= 7:
                q = board[r * 8 + f]
                if q == 6:   w  += 1.0
                if q == 12:  b_ += 1.0

        # Peões brancos atacam diagonais para cima
        if r0 > 0:
            if f0 > 0 and board[(r0 - 1) * 8 + (f0 - 1)] == 1: w += 1.0
            if f0 < 7 and board[(r0 - 1) * 8 + (f0 + 1)] == 1: w += 1.0
        # Peões pretos atacam diagonais para baixo
        if r0 < 7:
            if f0 > 0 and board[(r0 + 1) * 8 + (f0 - 1)] == 7: b_ += 1.0
            if f0 < 7 and board[(r0 + 1) * 8 + (f0 + 1)] == 7: b_ += 1.0

        # Peças deslizantes — bispos (3/9), torres (4/10), damas (5/11)
        # Para cada direção: anda até bater em peça; se for deslizante na direção certa, soma
        for d_idx, (dr, df) in enumerate(KING_DELTAS):
            for step in range(1, 8):
                r, f = r0 + dr * step, f0 + df * step
                if r < 0 or r > 7 or f < 0 or f > 7:
                    break
                q = board[r * 8 + f]
                if q == 0:
                    continue
                # Achou peça — para o raio e classifica
                is_diag = (dr != 0 and df != 0)
                if is_diag:
                    if q == 3 or q == 5:  w  += 1.0
                    if q == 9 or q == 11: b_ += 1.0
                else:
                    if q == 4 or q == 5:   w  += 1.0
                    if q == 10 or q == 11: b_ += 1.0
                break  # raio bloqueado

        out[17, sq] = w
        out[18, sq] = b_

    return out.reshape(19, 8, 8)


def dense_to_move(idx, flip):
    """Inverte remap_move_cuda: índice 0..4223 → string UCI."""
    if idx < 4096:
        to_sq   = idx // 64
        from_sq = idx % 64
        promo   = 0
    else:
        promo_idx = idx - 4096
        from_file = promo_idx // 12
        rem       = promo_idx % 12
        direction = rem // 4
        promo     = (rem % 4) + 1
        from_sq   = 48 + from_file
        to_file   = from_file + direction - 1
        to_sq     = 56 + to_file

    if flip:
        from_sq ^= 56
        to_sq   ^= 56

    move = chr(ord('a') + (from_sq & 7)) + chr(ord('1') + (from_sq >> 3))
    move += chr(ord('a') + (to_sq   & 7)) + chr(ord('1') + (to_sq   >> 3))
    if promo:
        move += PROMO_CHARS[promo]
    return move


# ----------------------------------------------------------------------
# Predição
# ----------------------------------------------------------------------
def predict(model, fen):
    compact, flip = parse_fen(fen)
    inflated_np   = decode_cpu(compact)                         # (19, 8, 8)
    board_input   = torch.from_numpy(inflated_np).unsqueeze(0)  # (1, 19, 8, 8)

    with torch.no_grad():
        policy, value = model(board_input)

    probs = F.softmax(policy[0].float(), dim=0)
    topk  = torch.topk(probs, 10)

    val = value[0].item()
    if flip:
        val = -val

    print(f"\n{'='*50}")
    print(f"  FEN: {fen}")
    cp_est = 600 * np.arctanh(max(min(val, 0.999), -0.999))
    print(f"  Avaliação: {val:+.3f}  (tanh(cp/600); cp ≈ {cp_est:+.0f})")
    print(f"{'='*50}")
    print(f"\n  Top 10 lances:")
    for i, (prob, idx) in enumerate(zip(topk.values, topk.indices)):
        move = dense_to_move(idx.item(), flip)
        bar  = "█" * int(prob.item() * 50)
        print(f"  {i+1:2d}. {move:<6s} {prob.item()*100:5.1f}%  {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoint_latest.pt')
    parser.add_argument('--channels',   type=int, default=128)
    parser.add_argument('--res-blocks', type=int, default=12)
    parser.add_argument('--bn-train',   action='store_true',
                        help='Mantém BatchNorm em modo train (útil quando o checkpoint ainda tem poucas épocas)')
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    model = ChessNet(channels=args.channels, num_res_blocks=args.res_blocks).to(DEVICE)
    model.load_state_dict(checkpoint['model'])

    if args.bn_train:
        model.train()
        print(f"BatchNorm em modo TRAIN (running stats ignorados).")
    else:
        model.eval()

    print(f"Modelo carregado ({args.checkpoint}, época {checkpoint['epoch']})")

    START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    while True:
        fen = input(f"\nFEN (enter = posição inicial, 'q' = sair): ").strip()
        if fen.lower() == 'q':
            break
        if not fen:
            fen = START_FEN
        try:
            predict(model, fen)
        except Exception as e:
            print(f"Erro: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()