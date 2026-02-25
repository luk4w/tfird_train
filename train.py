import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import ctypes
from tqdm import tqdm
import os

# ==========================================
# 1. CONFIGURAÇÕES GERAIS E INTEGRAÇÃO C++
# ==========================================
ENTRY_SIZE = 64
NUM_ENTRIES = os.path.getsize('dataset.bin') // ENTRY_SIZE
BATCH_SIZE = 4096
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cuda_lib = ctypes.CDLL('./decoder.dll')
cuda_lib.decode_cuda.argtypes = [
    ctypes.c_void_p,  
    ctypes.c_void_p,  
    ctypes.c_int      
]

# ==========================================
# 2. DATASET
# ==========================================
class ChessIterableDataset(IterableDataset):
    def __init__(self, file_path, batch_size=1024, chunk_size=102400):
        self.file_path = file_path
        self.entry_size = 64
        self.batch_size = batch_size
        self.chunk_size = (chunk_size // batch_size) * batch_size

    def __iter__(self):
        with open(self.file_path, 'rb') as f:
            while True:
                bytes_data = f.read(self.chunk_size * self.entry_size)
                if not bytes_data:
                    break
                
                num_entries_read = len(bytes_data) // self.entry_size
                chunk_arr = np.frombuffer(bytes_data, dtype=np.uint8).reshape((num_entries_read, self.entry_size)).copy()
                
                for i in range(0, num_entries_read, self.batch_size):
                    yield chunk_arr[i : i + self.batch_size]

# ==========================================
# 3. ARQUITETURA DA REDE
# ==========================================
# --------------------------------------------------------------------------
# SEBlock (Squeeze-and-Excitation)
# --------------------------------------------------------------------------
# Imagine que a rede tem 128 "canais", cada um detectando algo diferente:
#   Canal 1: detecta peões conectados
#   Canal 2: detecta colunas abertas
#   Canal 3: detecta cavalos centralizados
#   ... etc.
#
# O problema: nem todos os canais importam pra TODA posição.
# Numa posição fechada, o canal de colunas abertas é inútil.
#
# O SEBlock resolve isso em 3 passos:
#   1. SQUEEZE:  Resume cada canal 8x8 num único número (média global)
#   2. EXCITATE: Passa essa "ficha resumo" por uma mini-rede que decide
#                um peso de 0.0 a 1.0 pra cada canal
#   3. SCALE:    Multiplica cada canal pelo seu peso
#
# Resultado: canais relevantes ficam fortes, irrelevantes ficam silenciados.
# Isso acontece DINAMICAMENTE — o mesmo canal pode ter peso 0.9 numa
# posição e 0.1 em outra.
# --------------------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        # Squeeze: Comprime cada canal 8x8 → 1 número (média do canal)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # Excitation: Mini-rede que decide o peso de cada canal
        # channels → channels/16 → channels (gargalo pra eficiência)
        # Sigmoid no final garante que os pesos ficam entre 0 e 1
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()                    # b=batch, c=canais (128)
        y = self.squeeze(x).view(b, c)           # [B, 128, 8, 8] → [B, 128]
        y = self.excitation(y).view(b, c, 1, 1)  # [B, 128] → [B, 128, 1, 1]
        return x * y.expand_as(x)                # Multiplica cada canal pelo seu peso

# --------------------------------------------------------------------------
# ResSEBlock (Bloco Residual + SE)
# --------------------------------------------------------------------------
# Empilha duas convoluções + o SEBlock, com uma "conexão residual".
#
# A conexão residual é o "atalho" que soma a entrada original à saída:
#   saída = F(x) + x
#
# Por que isso importa?
#   - Sem residual: a rede precisa aprender a transformação completa
#   - Com residual: a rede só aprende o DELTA (a diferença)
#   - Isso permite empilhar muitos blocos (10+) sem que o gradiente
#     "desapareça" durante o treinamento
#
# Fluxo:
#   x → Conv3x3 → BatchNorm → ReLU → Conv3x3 → BatchNorm → SE → (+x) → ReLU
# --------------------------------------------------------------------------
class ResSEBlock(nn.Module):
    def __init__(self, channels):
        super(ResSEBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)   # Normaliza os valores (estabiliza treino)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)            # Recalibra os canais

    def forward(self, x):
        residual = x                                 # Salva a entrada original
        out = F.relu(self.bn1(self.conv1(x)))        # 1ª conv + norm + ativação
        out = self.bn2(self.conv2(out))              # 2ª conv + norm (sem ReLU aqui)
        out = self.se(out)                           # SE recalibra os canais
        out += residual                              # Soma o atalho residual
        return F.relu(out)                           # ReLU final

# --------------------------------------------------------------------------
# ChessNet — A rede completa
# --------------------------------------------------------------------------
# Entrada: Tabuleiro representado como tensor [Batch, 19, 8, 8]
#   - 12 canais: peças (P, N, B, R, Q, K × brancas e pretas)
#   - 1 canal:   en passant
#   - 4 canais:  direitos de roque (KQkq)
#   - 2 canais:  mapa de influencia (brancas e pretas)
#   Total: 19 canais, cada um é um grid 8x8
#
# Arquitetura:
#   Input [19, 8, 8]
#     ↓
#   Conv3x3 inicial: 19 canais → 128 canais (expande a representação)
#     ↓
#   10× ResSEBlock: processa padrões cada vez mais complexos
#     ↓ (bifurca em duas "cabeças")
#   ┌──────────────────────────────────────────────────┐
#   │ Policy Head: "Qual lance jogar?"                 │
#   │   Conv1x1 → 66 canais → Flatten → 4224 classes   │
#   │   Cada classe = um lance possível (from-to-promo)│
#   ├──────────────────────────────────────────────────┤
#   │ Value Head: "Quem está ganhando?"                │
#   │   Conv1x1 → 1 canal → Flatten → FC → Tanh        │
#   │   Saída: um número de -1 (pretas) a +1 (brancas) │
#   └──────────────────────────────────────────────────┘
# --------------------------------------------------------------------------
class ChessNet(nn.Module):
    def __init__(self, channels=128, num_res_blocks=6):
        super(ChessNet, self).__init__()
        
        # Camada inicial: expande 19 canais de input → 128 canais de features
        self.conv_inicial = nn.Conv2d(19, channels, kernel_size=3, padding=1, bias=False)
        self.bn_inicial = nn.BatchNorm2d(channels)
        
        # Torre de blocos residuais — aqui é onde a "inteligência" mora
        # Cada bloco refina a representação, detectando padrões mais abstratos
        # Bloco 1-3: aprende padrões simples (peças atacadas, estrutura de peões)
        # Bloco 4-7: padrões táticos (forks, pins, skewers)
        # Bloco 8-10: padrões estratégicos (outposts, colunas abertas, rei exposto)
        self.res_blocks = nn.Sequential(
            *[ResSEBlock(channels) for _ in range(num_res_blocks)]
        )
        
        # POLICY HEAD — Prevê o melhor lance
        # Conv1x1 com 66 canais: 66 × 64 casas = 4224 classes de lances
        # - Índices 0–4095: lances normais (to × 64 + from)
        # - Índices 4096–4191: promoções (file × 12 + dir × 4 + promo)
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 66, kernel_size=1),
            nn.Flatten()
        )
        
        # VALUE HEAD — Prevê quem ganha
        # Comprime tudo pra um único número entre -1 e +1
        # -1 = pretas ganhando, 0 = empate, +1 = brancas ganhando
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),   # 128 canais → 1 canal
            nn.Flatten(),                            # [B, 1, 8, 8] → [B, 64]
            nn.Linear(64, 128),                      # 64 → 128 neurônios
            nn.ReLU(),
            nn.Linear(128, 1),                       # 128 → 1 número
            nn.Tanh()                                # Comprime pra [-1, +1]
        )

    def forward(self, x):
        x = F.relu(self.bn_inicial(self.conv_inicial(x)))   # Input → 128 canais
        x = self.res_blocks(x)                              # 10 blocos de refinamento
        policy = self.policy_head(x)                        # → 4224 probabilidades
        value = self.value_head(x)                          # → 1 avaliação
        return policy, value

# --------------------------------------------------------------------------
# Custom Loss — Função de custo
# --------------------------------------------------------------------------
# A rede aprende minimizando dois erros ao mesmo tempo:
#
# 1. POLICY LOSS (Cross-Entropy):
#    Mede quão longe a distribuição prevista está do lance correto.
#    Se a rede dá 80% pro lance certo → loss baixo
#    Se a rede dá 1% pro lance certo → loss alto
#    Valor teórico mínimo: 0 (acertou 100%)
#
# 2. VALUE LOSS (MSE — Mean Squared Error):
#    Mede o erro quadrático entre a avaliação prevista e a real.
#    Previu +0.5 mas era +0.3 → erro = (0.5 - 0.3)² = 0.04
#    Quanto menor, mais precisa a avaliação da posição.
#
# O loss total é a SOMA dos dois. A rede precisa acertar TANTO o lance
# quanto a avaliação. Se focar só num, o outro piora.
# --------------------------------------------------------------------------
def custom_loss(policy_pred, policy_target, value_pred, value_target):
    loss_policy = F.cross_entropy(policy_pred, policy_target)
    loss_value = F.mse_loss(value_pred.view(-1), value_target.float())
    total_loss = loss_policy + loss_value
    return total_loss, loss_policy, loss_value

# --------------------------------------------------------------------------
# Remapeamento de lances: encoding cru → índice denso
# --------------------------------------------------------------------------
# O dataset guarda cada lance como um uint16 com o formato:
#   (promo << 12) | (to << 6) | from
#
# Isso dá índices até 20479, mas a maioria são impossíveis.
# Remapeamos pra um encoding denso com 4224 classes:
#   - Lances normais: to × 64 + from → 0..4095
#   - Promoções:      4096 + file × 12 + dir × 4 + (promo-1) → 4096..4191
#   - Padding:        4192..4223 (não usados, existem pro Conv2d fechar)
# --------------------------------------------------------------------------
def remap_move_to_dense(raw_moves_np):
    from_sq = (raw_moves_np & 0x3F).astype(np.int32)         # Bits 0-5: casa de origem
    to_sq   = ((raw_moves_np >> 6) & 0x3F).astype(np.int32)  # Bits 6-11: casa de destino
    promo   = ((raw_moves_np >> 12) & 0xF).astype(np.int32)  # Bits 12-15: tipo promoção
    
    normal_idx = to_sq * 64 + from_sq  # Encoding normal: 0..4095
    
    # Encoding de promoção: compacta file, direção e tipo em 96 valores
    from_file = from_sq & 7
    to_file   = to_sq & 7
    direction = to_file - from_file + 1  # -1→0, 0→1, +1→2 (esq, frente, dir)
    promo_idx = 4096 + from_file * 12 + direction * 4 + (promo - 1)
    
    return np.where(promo == 0, normal_idx, promo_idx)

# --------------------------------------------------------------------------
# Extração de targets do batch compacto
# --------------------------------------------------------------------------
# Cada entrada do dataset.bin tem 64 bytes:
#   Bytes 0-31:  Tabuleiro (32 bytes, 2 casas por byte em nibbles)
#   Byte 32:     En passant
#   Byte 33:     Flags (turno, roques)
#   Bytes 34-43: 5 lances (uint16 cada)
#   Bytes 44-63: 5 scores (float32 cada)
#
# Usamos moves[0] como target de policy e scores[0] como target de value.
# --------------------------------------------------------------------------
def extract_targets_from_batch(compact_batch_tensor):
    arr_np = compact_batch_tensor.numpy()
    
    policy_np = arr_np[:, 34:36].copy().view(np.uint16).squeeze()  # moves[0]
    value_np = arr_np[:, 44:48].copy().view(np.float32).squeeze()  # scores[0]
    
    policy_np = remap_move_to_dense(policy_np)  # uint16 cru → índice 0..4223
    
    policy_targets = torch.tensor(policy_np, dtype=torch.long, device=DEVICE)
    value_targets = torch.tensor(value_np, dtype=torch.float32, device=DEVICE)
    
    # Verificação de segurança: se algum índice está fora do range, clampa
    min_val = policy_targets.min().item()
    max_val = policy_targets.max().item()
    
    if min_val < 0 or max_val >= 4224:
        print(f"\n[ALERTA BINÁRIO] Lances fora do limite: Min={min_val}, Max={max_val}")
        policy_targets = torch.clamp(policy_targets, 0, 4223)
        
    return policy_targets, value_targets

# --------------------------------------------------------------------------
# Run Batch — Processa um batch de posições
# --------------------------------------------------------------------------
# Fluxo de um batch:
#   1. Extrai targets (lance correto + avaliação correta) dos bytes crus
#   2. Envia os 64 bytes compactos pra GPU
#   3. O decoder CUDA expande 64 bytes → 19 canais × 8×8 (tensor float)
#   4. Passa pela rede neural → policy + value
#   5. Calcula o loss comparando previsão vs realidade
# --------------------------------------------------------------------------
def run_batch(model, compact_batch_numpy):
    policy_target, value_target = extract_targets_from_batch(compact_batch_numpy)
    
    compact_gpu = compact_batch_numpy.to(DEVICE).contiguous()
    current_batch_size = compact_gpu.shape[0]
    
    # Decoder CUDA: descompacta 64 bytes → tensor [B, 19, 64]
    inflated_gpu = torch.empty((current_batch_size, 19, 64), dtype=torch.float32, device=DEVICE).contiguous()
    cuda_lib.decode_cuda(compact_gpu.data_ptr(), inflated_gpu.data_ptr(), current_batch_size)
    tabuleiros_input = inflated_gpu.view(current_batch_size, 19, 8, 8)
    
    # Mixed Precision (FP16): GPU processa em meia precisão = ~2x mais rápido
    with torch.amp.autocast('cuda'):
        policy_pred, value_pred = model(tabuleiros_input)
        loss, l_pol, l_val = custom_loss(
            policy_pred, policy_target, value_pred, value_target
        )
    
    return loss, l_pol, l_val

# --------------------------------------------------------------------------
# Loop de Treinamento
# --------------------------------------------------------------------------
# Em cada época, a rede vê TODAS as 318M posições do dataset.
# Em cada batch de 4096 posições:
#   1. Forward:  Rede faz previsões
#   2. Loss:     Calcula o erro
#   3. Backward: Calcula gradientes ("pra qual lado ajustar cada peso")
#   4. Step:     Ajusta os pesos na direção que reduz o erro
#
# O GradScaler gerencia o Mixed Precision, escalando o loss pra evitar
# underflow em FP16 e des-escalando os gradientes antes de aplicar.
# --------------------------------------------------------------------------
def train():
    model = ChessNet(channels=128, num_res_blocks=10).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler()
    
    # Resume do último checkpoint se existir
    start_epoch = 0
    if os.path.exists('checkpoint_latest.pt'):
        checkpoint = torch.load('checkpoint_latest.pt', map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch']

        for param_group in optimizer.param_groups:
            param_group['lr'] = LEARNING_RATE

        print(f"Resumindo da época {start_epoch + 1} com LR=1e-4!")
    
    total_batches = NUM_ENTRIES // BATCH_SIZE
    
    print("Iniciando treinamento...")
    for epoch in range(start_epoch, EPOCHS):
        # Recria dataset/dataloader a cada época (IterableDataset esgota o iterator)
        dataset = ChessIterableDataset('dataset.bin', batch_size=BATCH_SIZE)
        dataloader = DataLoader(dataset, batch_size=None)
        
        model.train()  # Ativa modo treino (BatchNorm e Dropout se comportam diferente)
        pbar = tqdm(enumerate(dataloader), total=total_batches, desc=f"Época {epoch+1}/{EPOCHS}", unit="batch")
        
        for batch_idx, compact_batch_numpy in pbar:
            loss, l_pol, l_val = run_batch(model, compact_batch_numpy)
            
            optimizer.zero_grad()           # Zera gradientes do batch anterior
            scaler.scale(loss).backward()   # Calcula gradientes (backpropagation)
            scaler.step(optimizer)          # Atualiza os pesos da rede
            scaler.update()                 # Ajusta a escala do FP16
            
            pbar.set_postfix({
                'Loss': f"{loss.item():.3f}", 
                'Pol': f"{l_pol.item():.3f}", 
                'Val': f"{l_val.item():.3f}"
            })
        
        # Salva tudo necessário pra retomar o treino depois
        checkpoint = {
            'epoch': epoch + 1,
            'model': model.state_dict(),            # Pesos da rede
            'optimizer': optimizer.state_dict(),    # Estado do otimizador (momentums)
            'scaler': scaler.state_dict(),          # Estado do FP16 scaler
        }
        torch.save(checkpoint, 'checkpoint_latest.pt')
        torch.save(checkpoint, f'checkpoint_epoch{epoch+1}_ft.pt')
        print(f"Modelo salvo: checkpoint_epoch{epoch+1}_ft.pt")

if __name__ == '__main__':
    train()