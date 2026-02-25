#include <iostream>
#include <random>
#include <cstdio>

#pragma pack(push, 1)
struct ChessData {
    uint8_t  data[32];
    uint8_t  en_passant;
    uint8_t  flags;
    uint16_t moves[5];
    float    scores[5];
};
#pragma pack(pop)

int main() {
    // Focando cirurgicamente no "Everest"
    const float MIN_VAL = -0.025f;
    const float MAX_VAL = 0.0001f; // Um pouquinho acima de 0 para garantir que pega o 0.0f exato do float

    // 22% de 45.7M = ~10 Milhões (nivelando com o resto do gráfico)
    const float CHANCE_MANTER = 0.22f; 

    FILE* in = fopen("dataset_lichess.bin", "rb");
    FILE* out = fopen("dataset.bin", "wb");

    if (!in || !out) {
        std::cerr << "Erro ao abrir arquivos. Verifique se dataset.bin esta na pasta!" << std::endl;
        return 1;
    }

    // Fixando a seed para reprodutibilidade (se rodar de novo, deleta as mesmas posições)
    std::mt19937 generator(42); 
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    ChessData cd;
    uint64_t lidas = 0;
    uint64_t salvas = 0;
    uint64_t podadas = 0;

    std::cout << "Iniciando poda cirurgica do Everest [-0.025 a 0.000]..." << std::endl;

    // Lendo bloco a bloco na velocidade do HD
    while (fread(&cd, sizeof(ChessData), 1, in) == 1) {
        lidas++;
        float score = cd.scores[0];
        bool manter = true;

        // Se o score cair EXATAMENTE no nosso pico problemático...
        if (score >= MIN_VAL && score <= MAX_VAL) {
            // Roda os dados: só sobrevive se tirar menos que 0.22 (22% de chance)
            if (distribution(generator) > CHANCE_MANTER) {
                manter = false;
                podadas++;
            }
        }

        if (manter) {
            fwrite(&cd, sizeof(ChessData), 1, out);
            salvas++;
        }

        // Feedback a cada 10 milhões de posições lidas
        if (lidas % 10000000 == 0) {
            std::cout << "\rLidas: " << lidas / 1000000 << "M | Salvas: " << salvas / 1000000 
                      << "M | Podadas: " << podadas / 1000000 << "M" << std::flush;
        }
    }

    fclose(in);
    fclose(out);

    std::cout << "\n\n=== RESULTADO FINAL ===" << std::endl;
    std::cout << "Posicoes Lidas:   " << lidas << std::endl;
    std::cout << "Posicoes Salvas:  " << salvas << std::endl;
    std::cout << "Posicoes Podadas: " << podadas << " (Exterminamos o Everest!)" << std::endl;

    return 0;
}