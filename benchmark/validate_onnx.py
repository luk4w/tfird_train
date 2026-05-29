"""
Valida 1 modelo ONNX exportado: compara saidas com PyTorch nativo.

Pega uma FEN, roda nos dois e mede diff. Se atol < 1e-4, OK.

Uso:
    python validate_onnx.py --pt ../../checkpoints/checkpoint_epoch12_ft.pt \
                            --onnx ../../checkpoints/checkpoint_epoch12_ft.onnx
"""

import argparse
import os
import sys
import numpy as np
import torch

# importa ChessNet e decode_cpu da pasta pai
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from train import ChessNet
from test_model import parse_fen, decode_cpu

try:
    import onnxruntime as ort
except ImportError:
    print("ERRO: onnxruntime nao instalado. Roda:")
    print("  pip install onnxruntime          (CPU)")
    print("  pip install onnxruntime-gpu      (GPU)")
    sys.exit(1)


# FENs de teste — variedade: inicial, meio-jogo, final, posicao com promocao
TEST_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
]


def run_pytorch(model, board_input_np):
    """Inferencia em PyTorch CPU."""
    x = torch.from_numpy(board_input_np).unsqueeze(0)  # (1, 19, 8, 8)
    with torch.no_grad():
        policy, value = model(x)
    return policy[0].numpy(), value[0].numpy()


def run_onnx(session, board_input_np):
    """Inferencia em ONNX Runtime."""
    x = board_input_np[np.newaxis, ...].astype(np.float32)  # (1, 19, 8, 8)
    outputs = session.run(['policy', 'value'], {'board': x})
    return outputs[0][0], outputs[1][0]


def compare(name, pt_out, onnx_out, atol=1e-4, rtol=1e-3):
    diff = np.abs(pt_out - onnx_out)
    max_diff = diff.max()
    mean_diff = diff.mean()
    ok = np.allclose(pt_out, onnx_out, atol=atol, rtol=rtol)
    status = "OK " if ok else "FAIL"
    print(f"  [{status}] {name:8s}  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt',   required=True, help='Caminho .pt')
    parser.add_argument('--onnx', required=True, help='Caminho .onnx')
    parser.add_argument('--channels', type=int, default=128)
    parser.add_argument('--res-blocks', type=int, default=12)
    parser.add_argument('--atol', type=float, default=1e-4)
    args = parser.parse_args()

    if not os.path.isfile(args.pt):
        print(f"ERRO: {args.pt} nao existe")
        sys.exit(1)
    if not os.path.isfile(args.onnx):
        print(f"ERRO: {args.onnx} nao existe")
        sys.exit(1)

    # --- Carrega PyTorch ---
    print(f"Carregando PyTorch: {args.pt}")
    ckpt = torch.load(args.pt, map_location='cpu', weights_only=True)
    model = ChessNet(channels=args.channels, num_res_blocks=args.res_blocks)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # --- Carrega ONNX ---
    print(f"Carregando ONNX:    {args.onnx}")
    providers = ['CPUExecutionProvider']
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(args.onnx, providers=providers)
    print(f"Providers ativos: {session.get_providers()}\n")

    # --- Compara em multiplas FENs ---
    all_ok = True
    for i, fen in enumerate(TEST_FENS):
        print(f"FEN #{i+1}: {fen[:60]}...")
        compact, flip = parse_fen(fen)
        board_np = decode_cpu(compact)  # (19, 8, 8)

        pt_policy, pt_value = run_pytorch(model, board_np)
        ox_policy, ox_value = run_onnx(session, board_np)

        ok_pol = compare("policy", pt_policy, ox_policy, atol=args.atol)
        ok_val = compare("value",  pt_value,  ox_value,  atol=args.atol)
        all_ok = all_ok and ok_pol and ok_val
        print()

    print("=" * 60)
    if all_ok:
        print("RESULTADO: Modelo ONNX valido. Pode exportar todos.")
        sys.exit(0)
    else:
        print("RESULTADO: Divergencia detectada. NAO exportar todos ainda.")
        sys.exit(2)


if __name__ == '__main__':
    main()
