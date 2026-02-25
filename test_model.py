import torch
import torch.nn.functional as F
import numpy as np
import ctypes

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cuda_lib = ctypes.CDLL('./decoder.dll')
cuda_lib.decode_cuda.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

from train import ChessNet

CHAR_TO_NIBBLE = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
}

FLIP_PIECE = [0, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
PROMO_CHARS = {1: 'n', 2: 'b', 3: 'r', 4: 'q'}

def parse_fen(fen):
    parts = fen.split()
    board_str, turn, castling, ep_str = parts[0], parts[1], parts[2], parts[3]
    
    data = bytearray(64)
    
    # 1. Tabuleiro
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
            idx = sq >> 1
            if sq & 1:
                data[idx] |= piece << 4
            else:
                data[idx] |= piece
            file += 1
    
    # 2. Turno
    flip = (turn == 'b')
    if flip:
        data[33] |= 1
    
    # 3. Roques
    castling_temp = 0
    for c in castling:
        if c == 'K': castling_temp |= (1 << 1)
        elif c == 'Q': castling_temp |= (1 << 2)
        elif c == 'k': castling_temp |= (1 << 3)
        elif c == 'q': castling_temp |= (1 << 4)
    
    if flip:
        data[33] |= ((castling_temp & 0x06) << 2) | ((castling_temp & 0x18) >> 2)
    else:
        data[33] |= castling_temp
    
    # 4. En passant
    data[32] = 255
    if ep_str != '-':
        ep = (ord(ep_str[1]) - ord('1')) * 8 + (ord(ep_str[0]) - ord('a'))
        data[32] = (ep ^ 56) if flip else ep
    
    # 5. Flip visual para pretas
    if flip:
        tmp = bytearray(data[:32])
        for i in range(32):
            data[i] = 0
        
        for sq in range(64):
            byte_idx = sq >> 1
            b = tmp[byte_idx]
            piece = (b >> 4) if (sq & 1) else (b & 0x0F)
            if piece == 0:
                continue
            t = sq ^ 56
            fp = FLIP_PIECE[piece]
            idx = t >> 1
            if t & 1:
                data[idx] |= fp << 4
            else:
                data[idx] |= fp
    
    return bytes(data), flip

def dense_to_move(idx, flip):
    """Converte índice denso (0-4223) → string UCI (ex: e2e4, e7e8q)."""
    if idx < 4096:
        from_sq = idx % 64
        to_sq = idx // 64
        promo = 0
    else:
        promo_idx = idx - 4096
        from_file = promo_idx // 12
        rem = promo_idx % 12
        direction = rem // 4
        promo = (rem % 4) + 1
        from_sq = 48 + from_file  # rank 6
        to_file = from_file + direction - 1
        to_sq = 56 + to_file      # rank 7
    
    # Des-canonizar se era pretas
    if flip:
        from_sq ^= 56
        to_sq ^= 56
    
    move = chr(ord('a') + (from_sq & 7)) + chr(ord('1') + (from_sq >> 3))
    move += chr(ord('a') + (to_sq & 7)) + chr(ord('1') + (to_sq >> 3))
    if promo:
        move += PROMO_CHARS[promo]
    return move


# ==========================================
# Predição
# ==========================================
def predict(model, fen):
    """Roda inferência para uma posição FEN."""
    compact, flip = parse_fen(fen)
    
    compact_np = np.frombuffer(compact, dtype=np.uint8).reshape(1, 64)
    compact_gpu = torch.from_numpy(compact_np.copy()).to(DEVICE).contiguous()
    
    inflated = torch.empty((1, 19, 64), dtype=torch.float32, device=DEVICE).contiguous()
    cuda_lib.decode_cuda(compact_gpu.data_ptr(), inflated.data_ptr(), 1)
    board_input = inflated.view(1, 19, 8, 8)
    
    with torch.no_grad(), torch.amp.autocast('cuda'):
        policy, value = model(board_input)
    
    # Top 10 lances
    probs = F.softmax(policy[0].float(), dim=0)
    topk = torch.topk(probs, 10)
    
    val = value[0].item()
    if flip:
        val = -val
    
    print(f"\n{'='*50}")
    print(f"  FEN: {fen}")
    print(f"  Avaliação: {val:+.3f}")
    print(f"{'='*50}")
    print(f"\n  Top 10 lances:")
    for i, (prob, idx) in enumerate(zip(topk.values, topk.indices)):
        move = dense_to_move(idx.item(), flip)
        bar = "█" * int(prob.item() * 50)
        print(f"  {i+1:2d}. {move:<6s} {prob.item()*100:5.1f}%  {bar}")

def main():
    # Carrega modelo
    checkpoint = torch.load('checkpoint_latest.pt', map_location=DEVICE, weights_only=True)
    model = ChessNet(channels=128, num_res_blocks=10).to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"Modelo carregado (época {checkpoint['epoch']})")
    
    # Posição inicial padrão
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
