import numpy as np
import time
import sys

# ==========================================
# ESPELHO DA STRUCT C++ (64 Bytes)
# ==========================================
chess_struct = np.dtype([
    ('data', np.uint8, (32,)),
    ('en_passant', np.uint8),
    ('flags', np.uint8),
    ('moves', np.uint16, (5,)),
    ('scores', np.float32, (5,))
])

def analisar_dataset(bin_path, step=0.025, chunk_size=10_000_000):
    print(f"[{time.strftime('%H:%M:%S')}] Mapeando o arquivo {bin_path} direto na RAM...\n")
    
    start_time = time.time()
    
    # Mapeia o arquivo instantaneamente
    dataset = np.memmap(bin_path, dtype=chess_struct, mode='r')
    total_pos = len(dataset)
    
    # Prepara os intervalos de -1.0 a +1.0
    bins = np.arange(-1.0, 1.0 + step + 0.0001, step)
    total_counts = np.zeros(len(bins) - 1, dtype=np.int64)
    total_validos = 0
    
    print(f"Total de posições no arquivo: {total_pos:,}")
    print("Iniciando varredura...\n")
    
    # Processa em blocos para não estourar a RAM e dar feedback visual
    for start_idx in range(0, total_pos, chunk_size):
        end_idx = min(start_idx + chunk_size, total_pos)
        
        # Puxa o lote atual
        chunk = dataset['scores'][start_idx:end_idx, 0]
        
        # Filtra lances válidos
        valid_chunk = chunk[chunk >= -1.0]
        total_validos += len(valid_chunk)
        
        # Calcula o histograma só desse lote e soma no total
        counts, _ = np.histogram(valid_chunk, bins=bins)
        total_counts += counts
        
        # ==========================================
        # FEEDBACK EM TEMPO REAL
        # ==========================================
        progress = (end_idx / total_pos) * 100
        # O '\r' faz ele sobrescrever a mesma linha no terminal
        sys.stdout.write(f"\rProcessando: {end_idx:>15,} / {total_pos:,} [{progress:>5.1f}%]")
        sys.stdout.flush()
        
    read_time = time.time() - start_time
    
    print(f"\n\n[{time.strftime('%H:%M:%S')}] Análise Concluída em {read_time:.2f} segundos!")
    print(f"=" * 65)
    print(f"Posições com Lance Válido: {total_validos:,}")
    print(f"=" * 65)
    
    # ==========================================
    # DESENHANDO O GRÁFICO ASCII
    # ==========================================
    max_count = np.max(total_counts) if np.max(total_counts) > 0 else 1
        
    print(f"{'Intervalo (Tanh)':<18} | {'Qtd':<10} | Distribuição")
    print("-" * 65)
    
    for i in range(len(total_counts)):
        b_min = bins[i]
        b_max = bins[i+1]
        
        is_center = (b_min <= 0.0) and (b_max > 0.0)
        marker = "--> " if is_center else "    "
        
        range_str = f"{marker}[{b_min:>6.3f} a {b_max:>6.3f})"
        
        bar_len = int((total_counts[i] / max_count) * 35)
        bar = "█" * bar_len
        
        print(f"{range_str:<18} | {total_counts[i]:<10,} | {bar}")

if __name__ == "__main__":
    analisar_dataset("dataset.bin", step=0.025)