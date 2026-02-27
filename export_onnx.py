import glob
import os
import argparse
import torch
import torch.nn as nn
from train import ChessNet

# Shared model configuration
MODEL_KWARGS = dict(channels=128, num_res_blocks=10)

class FP16Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        # Converte os pesos da rede neural inteira para FP16 (Meia precisão)
        self.model = model.half() 

    def forward(self, x):
        # 1. x entra como Float32 (vindo do C++)
        # 2. Converte pra Float16 e roda na rede neural
        policy, value = self.model(x.half())
        # 3. Converte as saídas de volta pra Float32 pra não quebrar o C++!
        return policy.float(), value.float()
def export_checkpoint(pt_path: str) -> None:
    """Load a checkpoint and export the corresponding ONNX file."""
    checkpoint = torch.load(pt_path, map_location='cpu', weights_only=True)
    
    # Carrega a rede normal
    base_model = ChessNet(**MODEL_KWARGS)
    base_model.load_state_dict(checkpoint['model'])
    base_model.eval()

    # EMBRULHA a rede na nossa "Capa" que gerencia a conversão de bits
    wrapper = FP16Wrapper(base_model)

    # O Tensor Dummy volta a ser Float32 (pois é o que o C++ vai mandar!)
    dummy = torch.randn(1, 19, 8, 8, dtype=torch.float32)
    
    base = os.path.splitext(os.path.basename(pt_path))[0]
    onnx_path = base + ".onnx" 
    
    torch.onnx.export(
        wrapper, dummy, onnx_path, # Exporta o WRAPPER e não o base_model!
        input_names=["board"],
        output_names=["policy", "value"],
        dynamic_axes={"board": {0: "batch"},
                      "policy": {0: "batch"},
                      "value": {0: "batch"}}
    )
    epoch = checkpoint.get('epoch', '?')
    print(f"Exported {onnx_path} (epoch {epoch}) com Wrapper FP16!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export one or more PyTorch checkpoints to ONNX."
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="paths or glob patterns to .pt files. Defaults to *.pt",
    )
    args = parser.parse_args()

    patterns = args.models or ["*.pt"]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat))

    if not paths:
        parser.error("no checkpoint files found for patterns: %r" % patterns)

    for pt in sorted(set(paths)):
        export_checkpoint(pt)
