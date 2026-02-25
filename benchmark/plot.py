import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# CONFIGURAÇÃO VISUAL DO GRÁFICO
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial']
})

sns.set_theme(style="whitegrid")

# Criar pasta para exportação de dados brutos se não existir
output_folder = 'benchmark_data'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("Carregando benchmark_results.csv...")
df = pd.read_csv('benchmark_results.csv')

# Prepara as faixas de Rating (500 a 2800)
df['Rating_Bin'] = (df['Rating'] // 100) * 100
df_filt = df[(df['Rating_Bin'] >= 500) & (df['Rating_Bin'] <= 2800)].copy()

# Identifica as colunas dos modelos dinamicamente e ordena
modelos = sorted([col for col in df.columns if 'checkpoint_epoch' in col])

# Substitui o Rank 0 (não encontrado) por Infinito para facilitar o cálculo
for mod in modelos:
    df_filt[mod] = df_filt[mod].replace(0, np.inf)

# Configurações de processamento
tops_to_plot = [1, 3, 5, 7, 10]
palette = ['#000000', '#E6194B', '#3CB44B', '#4363D8', '#F58231', '#911EB4', '#008080', '#F032E6', '#800000', '#9A6324']

for k in tops_to_plot:
    print(f"Processando Top-{k}...")
    plt.figure(figsize=(12, 6))
    
    # DataFrame para exportação dos dados processados (Tabela de acurácia)
    df_accuracy_k = pd.DataFrame()
    
    for idx, mod in enumerate(modelos):
        # Extração de informações para legenda
        match = re.search(r'epoch(\d+)', mod)
        num_epoch = int(match.group(1)) if match else 0
        is_ft = '_ft' in mod

        if is_ft:
            label_name = rf"Época {num_epoch} (Ajuste Fino, LR=$10^{{-4}}$)"
            line_style = '--'
        else:
            label_name = rf"Época {num_epoch} (LR=$10^{{-3}}$)"
            line_style = '-'
            
        # Destaque para a Época 7
        line_width = 3.5 if (num_epoch == 7 and not is_ft) else 2.0
        
        # Cálculo da acurácia por faixa de rating
        acc_series = df_filt.groupby('Rating_Bin')[mod].apply(lambda x: (x <= k).mean() * 100)
        
        # Armazena na tabela de exportação
        df_accuracy_k[label_name] = acc_series
        
        # Plota no gráfico
        plt.plot(acc_series.index, acc_series.values, 
                 linestyle=line_style, linewidth=line_width,
                 color=palette[idx % len(palette)], 
                 label=label_name)

    # Título e rótulos
    plt.title(f'Desempenho no Benchmark Lichess: Acurácia Top-{k} por Rating')
    plt.xlabel('Rating do Puzzle (Glicko-2)')
    plt.ylabel(f'Acurácia Top-{k} (%)')
    
    # Posicionamento da legenda
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Modelos Avaliados")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # Salva o Gráfico em alta resolução
    plt.savefig(f'graph_top{k}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Salva os dados brutos em CSV (útil para criar tabelas em documentos)
    csv_path = os.path.join(output_folder, f'accuracy_top{k}.csv')
    df_accuracy_k.to_csv(csv_path)
    print(f"  -> Gráfico 'graph_top{k}.png' e dados em '{csv_path}' gerados.")

print("\nProcessamento concluído com sucesso!")