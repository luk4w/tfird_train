// =====================================================================
// model_benchmark.cpp — V2
// Diferencas vs protótipo:
//   1. Decoder com canais 17-18 (mapa de influencia, identico ao decode_cpu)
//   2. Saidas extras no CSV: rank, prob_correto, entropy_full, entropy_legal, value
//   3. Salva tambem top-10 lances + probs em binario (para analise posterior)
// =====================================================================
#include "chess.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <iomanip>
#include <memory>
#include <filesystem>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <string_view>
#include <charconv>
#include <onnxruntime_cxx_api.h>

constexpr int CHANNELS    = 19;
constexpr int BOARD_SQ    = 64;
constexpr int POLICY_SIZE = 4224;
constexpr int BATCH_SIZE  = 1024;
constexpr int MAX_QUEUE_SIZE = 4;
constexpr int TOP_K       = 10;  // top-N lances salvos no .bin

static int charToNibble(char c) {
    switch (c) {
        case 'P': return 1;  case 'N': return 2;  case 'B': return 3;  case 'R': return 4;
        case 'Q': return 5;  case 'K': return 6;  case 'p': return 7;  case 'n': return 8;
        case 'b': return 9;  case 'r': return 10; case 'q': return 11; case 'k': return 12;
        default:  return 0;
    }
}

static const int FLIP_PIECE[13] = {0, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6};

// Offsets de cavalo e rei (mesmos do decode_cpu)
static const int KNIGHT_DR[8] = {-2,-2,-1,-1, 1, 1, 2, 2};
static const int KNIGHT_DF[8] = {-1, 1,-2, 2,-2, 2,-1, 1};
static const int KING_DR[8]   = {-1,-1,-1, 0, 0, 1, 1, 1};
static const int KING_DF[8]   = {-1, 0, 1,-1, 1,-1, 0, 1};

static std::vector<std::string_view> split_csv_fast(std::string_view line) {
    std::vector<std::string_view> out;
    out.reserve(5);
    size_t start = 0, pos = line.find(',');
    while (pos != std::string_view::npos) {
        out.emplace_back(line.data() + start, pos - start);
        start = pos + 1;
        pos = line.find(',', start);
    }
    out.emplace_back(line.data() + start, line.size() - start);
    return out;
}

static std::vector<std::string_view> split_spaces_fast(std::string_view str) {
    std::vector<std::string_view> out;
    size_t start = 0, pos = str.find(' ');
    while (pos != std::string_view::npos) {
        if (pos > start) out.emplace_back(str.data() + start, pos - start);
        start = pos + 1;
        pos = str.find(' ', start);
    }
    if (start < str.size()) out.emplace_back(str.data() + start, str.size() - start);
    return out;
}

struct Decoder {
    // Replica decode_cpu (test_model.py) bit-a-bit, agora com canais 17-18.
    static bool decode(std::string_view fen, float *output, bool &flip) {
        std::array<uint8_t, 64> compact{};
        auto parts = split_spaces_fast(fen);
        if (parts.size() < 4) return false;

        std::string_view p0 = parts[0], p1 = parts[1], p2 = parts[2], p3 = parts[3];

        // ---- Tabuleiro compacto (4 bits por casa) ----
        int rank = 7, file = 0;
        for (char c : p0) {
            if (c == '/') { rank--; file = 0; }
            else if (c >= '1' && c <= '8') file += c - '0';
            else {
                int sq = rank * 8 + file;
                int piece = charToNibble(c);
                int idx = sq >> 1;
                if (sq & 1) compact[idx] |= piece << 4;
                else        compact[idx] |= piece;
                file++;
            }
        }

        flip = (p1 == "b");
        if (flip) compact[33] |= 1;  // marca side-to-move=black

        // ---- Castle rights ----
        uint8_t castle = 0;
        for (char c : p2) {
            if (c == 'K') castle |= 2;
            if (c == 'Q') castle |= 4;
            if (c == 'k') castle |= 8;
            if (c == 'q') castle |= 16;
        }
        if (flip) compact[33] |= ((castle & 0x06) << 2) | ((castle & 0x18) >> 2);
        else      compact[33] |= castle;

        // ---- En passant ----
        compact[32] = 255;
        if (p3 != "-") {
            int ep = (p3[1] - '1') * 8 + (p3[0] - 'a');
            compact[32] = flip ? (ep ^ 56) : ep;
        }

        // ---- Flip do tabuleiro se vez das pretas ----
        if (flip) {
            uint8_t tmp[32];
            std::memcpy(tmp, compact.data(), 32);
            std::memset(compact.data(), 0, 32);
            for (int sq = 0; sq < 64; sq++) {
                int bi = sq >> 1;
                uint8_t b = tmp[bi];
                int piece = (sq & 1) ? (b >> 4) : (b & 0x0F);
                if (!piece) continue;
                int t = sq ^ 56;
                int fp = FLIP_PIECE[piece];
                int idx = t >> 1;
                if (t & 1) compact[idx] |= fp << 4;
                else       compact[idx] |= fp;
            }
        }

        // ---- Expande tabuleiro pra array de 64 ----
        std::memset(output, 0, CHANNELS * BOARD_SQ * sizeof(float));
        uint8_t boardArr[64];
        for (int sq = 0; sq < 64; sq++) {
            uint8_t b = compact[sq >> 1];
            boardArr[sq] = (sq & 1) ? (b >> 4) : (b & 0x0F);
        }

        uint8_t ep = compact[32], flags = compact[33];
        auto write = [&](int c, int sq, float v) { output[c * 64 + sq] = v; };

        // ---- Canais 0..16 (peças, EP, castle) ----
        for (int sq = 0; sq < 64; sq++) {
            uint8_t p = boardArr[sq];
            if (p >= 1 && p <= 12) write(p - 1, sq, 1.0f);
            if (ep == sq && ep < 64) write(12, sq, 1.0f);
            if (flags & 1)  write(13, sq, 1.0f);
            if (flags & 2)  write(14, sq, 1.0f);
            if (flags & 4)  write(15, sq, 1.0f);
            if (flags & 8)  write(16, sq, 1.0f);
        }

        // ---- Canais 17..18 (mapa de influência w/b) ----
        for (int sq = 0; sq < 64; sq++) {
            int r0 = sq >> 3, f0 = sq & 7;
            float w = 0.0f, b_ = 0.0f;

            // Cavalos (2=N, 8=n)
            for (int i = 0; i < 8; ++i) {
                int r = r0 + KNIGHT_DR[i], f = f0 + KNIGHT_DF[i];
                if (r >= 0 && r <= 7 && f >= 0 && f <= 7) {
                    uint8_t q = boardArr[r * 8 + f];
                    if (q == 2) w  += 1.0f;
                    if (q == 8) b_ += 1.0f;
                }
            }

            // Reis (6=K, 12=k)
            for (int i = 0; i < 8; ++i) {
                int r = r0 + KING_DR[i], f = f0 + KING_DF[i];
                if (r >= 0 && r <= 7 && f >= 0 && f <= 7) {
                    uint8_t q = boardArr[r * 8 + f];
                    if (q == 6)  w  += 1.0f;
                    if (q == 12) b_ += 1.0f;
                }
            }

            // Peões brancos atacam diagonais pra cima (1 = P)
            if (r0 > 0) {
                if (f0 > 0 && boardArr[(r0 - 1) * 8 + (f0 - 1)] == 1) w += 1.0f;
                if (f0 < 7 && boardArr[(r0 - 1) * 8 + (f0 + 1)] == 1) w += 1.0f;
            }
            // Peões pretos atacam diagonais pra baixo (7 = p)
            if (r0 < 7) {
                if (f0 > 0 && boardArr[(r0 + 1) * 8 + (f0 - 1)] == 7) b_ += 1.0f;
                if (f0 < 7 && boardArr[(r0 + 1) * 8 + (f0 + 1)] == 7) b_ += 1.0f;
            }

            // Deslizantes (bispos 3/9, torres 4/10, damas 5/11) — para no primeiro bloqueio
            for (int d = 0; d < 8; ++d) {
                int dr = KING_DR[d], df = KING_DF[d];
                for (int step = 1; step < 8; ++step) {
                    int r = r0 + dr * step, f = f0 + df * step;
                    if (r < 0 || r > 7 || f < 0 || f > 7) break;
                    uint8_t q = boardArr[r * 8 + f];
                    if (q == 0) continue;
                    bool is_diag = (dr != 0 && df != 0);
                    if (is_diag) {
                        if (q == 3 || q == 5)  w  += 1.0f;
                        if (q == 9 || q == 11) b_ += 1.0f;
                    } else {
                        if (q == 4 || q == 5)   w  += 1.0f;
                        if (q == 10 || q == 11) b_ += 1.0f;
                    }
                    break;  // raio bloqueado
                }
            }

            write(17, sq, w);
            write(18, sq, b_);
        }
        return true;
    }
};

static int uciToIndex(const std::string &uci, bool flip) {
    if (uci.length() < 4) return -1;
    int from = (uci[1] - '1') * 8 + (uci[0] - 'a');
    int to   = (uci[3] - '1') * 8 + (uci[2] - 'a');
    if (flip) { from ^= 56; to ^= 56; }

    if (uci.length() == 4) return to * 64 + from;
    else if (uci.length() == 5) {
        int p = 0;
        if      (uci[4] == 'n') p = 0;
        else if (uci[4] == 'b') p = 1;
        else if (uci[4] == 'r') p = 2;
        else if (uci[4] == 'q') p = 3;
        int ff  = from - 48;
        int dir = to - 56 - ff + 1;
        int rem = dir * 4 + p;
        return 4096 + (ff * 12 + rem);
    }
    return -1;
}

struct ScoredMove {
    std::string uci;
    float logit;
    int idx;
};

struct PuzzleMeta {
    std::string puzzle_id;
    int rating;
    int solution_length;
    bool flip;
    chess::Movelist legal_moves;
    std::string correct_move;
};

// Metricas calculadas por (puzzle, modelo)
struct PuzzleResult {
    int rank;              // posicao do lance correto no ranking (1=top1, 0=nao achado)
    float prob_correct;    // probabilidade (softmax full) do lance correto
    float entropy_full;    // H sobre todos os 4224 logits
    float entropy_legal;   // H renormalizada sobre lances legais
    float value;           // saida da value head (em [-1, 1])
    // Top-K lances salvos no binario auxiliar
    std::array<int,   TOP_K> top_idx;
    std::array<float, TOP_K> top_prob;
    int top_n;             // quantos realmente preenchidos (pode ser < TOP_K se poucos lances legais)
};

struct BatchJob {
    std::vector<float> batched_inputs;
    std::vector<PuzzleMeta> batched_meta;
};

struct WriterJob {
    std::vector<PuzzleMeta> batched_meta;
    // [puzzle_idx][model_idx] -> resultado
    std::vector<std::vector<PuzzleResult>> batch_results;
};

class BenchmarkEngine {
    Ort::Env env;
    Ort::Session session{nullptr};

public:
    BenchmarkEngine(const std::string &model_path) : env(ORT_LOGGING_LEVEL_WARNING, "bench") {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        session_options.AppendExecutionProvider_CUDA(cuda_options);

#ifdef _WIN32
        session = Ort::Session(env, std::wstring(model_path.begin(), model_path.end()).c_str(), session_options);
#else
        session = Ort::Session(env, model_path.c_str(), session_options);
#endif
    }

    void evaluateBatch(float *batched_inputs, int current_batch_size,
                       const std::vector<PuzzleMeta> &meta_batch,
                       std::vector<std::vector<PuzzleResult>> &batch_results,
                       int model_index) {

        std::array<int64_t, 4> shape = {current_batch_size, CHANNELS, 8, 8};
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto tensor = Ort::Value::CreateTensor<float>(mem, batched_inputs,
            current_batch_size * CHANNELS * BOARD_SQ, shape.data(), shape.size());

        const char *in_names[]  = {"board"};
        const char *out_names[] = {"policy", "value"};

        auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 2);
        float *policy_batch = outputs[0].GetTensorMutableData<float>();
        float *value_batch  = outputs[1].GetTensorMutableData<float>();

        for (int b = 0; b < current_batch_size; ++b) {
            float *policy = policy_batch + (b * POLICY_SIZE);
            const auto &m = meta_batch[b];
            PuzzleResult &res = batch_results[b][model_index];

            // ---- Softmax full (estavel) ----
            float maxLogit = policy[0];
            for (int i = 1; i < POLICY_SIZE; ++i)
                if (policy[i] > maxLogit) maxLogit = policy[i];

            double sumExp = 0.0;
            // Reusa buffer numa thread-local pra evitar alocacao por puzzle
            thread_local std::vector<float> probs(POLICY_SIZE);
            for (int i = 0; i < POLICY_SIZE; ++i) {
                probs[i] = std::exp(policy[i] - maxLogit);
                sumExp  += probs[i];
            }
            float invSum = static_cast<float>(1.0 / sumExp);
            for (int i = 0; i < POLICY_SIZE; ++i) probs[i] *= invSum;

            // ---- Entropia full ----
            double H_full = 0.0;
            for (int i = 0; i < POLICY_SIZE; ++i) {
                if (probs[i] > 1e-12f)
                    H_full -= probs[i] * std::log(probs[i]);
            }
            res.entropy_full = static_cast<float>(H_full);

            // ---- Coleta logits dos lances legais ----
            std::vector<ScoredMove> ranked;
            ranked.reserve(m.legal_moves.size());
            int correct_idx = -1;
            for (const auto &move : m.legal_moves) {
                std::string uci_str = chess::uci::moveToUci(move);
                int idx = uciToIndex(uci_str, m.flip);
                if (idx >= 0 && idx < POLICY_SIZE) {
                    ranked.push_back({uci_str, policy[idx], idx});
                    if (uci_str == m.correct_move) correct_idx = idx;
                }
            }

            // ---- Softmax APENAS nos legais (entropia_legal) ----
            float maxLegal = -1e30f;
            for (const auto &mv : ranked) if (mv.logit > maxLegal) maxLegal = mv.logit;
            double sumLegal = 0.0;
            std::vector<float> probs_legal(ranked.size());
            for (size_t i = 0; i < ranked.size(); ++i) {
                probs_legal[i] = std::exp(ranked[i].logit - maxLegal);
                sumLegal      += probs_legal[i];
            }
            float invSumL = static_cast<float>(1.0 / sumLegal);
            for (size_t i = 0; i < probs_legal.size(); ++i) probs_legal[i] *= invSumL;
            double H_legal = 0.0;
            for (float p : probs_legal) if (p > 1e-12f) H_legal -= p * std::log(p);
            res.entropy_legal = static_cast<float>(H_legal);

            // ---- Ranking de lances legais por logit ----
            std::sort(ranked.begin(), ranked.end(), [](const ScoredMove &a, const ScoredMove &b) {
                return a.logit > b.logit;
            });

            int rank_found = 0;
            for (size_t i = 0; i < ranked.size(); ++i) {
                if (ranked[i].uci == m.correct_move) { rank_found = static_cast<int>(i + 1); break; }
            }
            res.rank = rank_found;

            res.prob_correct = (correct_idx >= 0) ? probs[correct_idx] : 0.0f;
            res.value        = value_batch[b];

            // ---- Top-K do ranking de lances legais ----
            int top_n = std::min<int>(TOP_K, static_cast<int>(ranked.size()));
            res.top_n = top_n;
            for (int i = 0; i < top_n; ++i) {
                res.top_idx[i]  = ranked[i].idx;
                res.top_prob[i] = probs[ranked[i].idx];  // prob na distribuicao FULL
            }
            for (int i = top_n; i < TOP_K; ++i) {
                res.top_idx[i]  = -1;
                res.top_prob[i] = 0.0f;
            }
        }
    }
};

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    std::cerr << "[INFO] Iniciando benchmark V2 (decoder corrigido + metricas extras)" << std::endl;

    std::vector<std::string> model_names;
    for (const auto &entry : std::filesystem::directory_iterator(".")) {
        if (entry.path().extension() == ".onnx") {
            model_names.push_back(entry.path().filename().string());
        }
    }
    if (model_names.empty()) {
        std::cerr << "[ERRO] Nenhum .onnx encontrado no diretorio atual." << std::endl;
        return 1;
    }
    // Ordena por numero de epoca (epoch1, epoch2, ..., epoch10, epoch11, epoch12)
    auto epoch_of = [](const std::string &n) {
        size_t p = n.find("epoch");
        if (p == std::string::npos) return 0;
        int e = 0; std::from_chars(n.data() + p + 5, n.data() + n.size(), e);
        return e;
    };
    std::sort(model_names.begin(), model_names.end(),
              [&](const std::string &a, const std::string &b) { return epoch_of(a) < epoch_of(b); });

    std::vector<std::unique_ptr<BenchmarkEngine>> engines;
    for (size_t i = 0; i < model_names.size(); ++i) {
        std::cerr << "[INFO] Carregando: " << model_names[i] << std::endl;
        engines.push_back(std::make_unique<BenchmarkEngine>(model_names[i]));
    }

    // ============================================================
    // CSV principal: 1 linha por puzzle, com 5 colunas por modelo
    // ============================================================
    std::ofstream csv_out("benchmark_results.csv");
    csv_out << "PuzzleId,Rating,SolutionLength";
    for (const auto &m : model_names) {
        // strip ".onnx"
        std::string base = m.substr(0, m.size() - 5);
        csv_out << "," << base << "_rank"
                << "," << base << "_prob"
                << "," << base << "_Hfull"
                << "," << base << "_Hlegal"
                << "," << base << "_value";
    }
    csv_out << "\n";
    csv_out << std::fixed << std::setprecision(6);

    // ============================================================
    // Binário: top-10 lances + probs (formato compacto)
    //   header: int32 num_models, int32 top_k
    //   por puzzle: char[32] puzzle_id (null-padded), int32 rating
    //               para cada modelo: int32[top_k] idx, float[top_k] prob
    // ============================================================
    std::ofstream bin_out("benchmark_top10.bin", std::ios::binary);
    int32_t hdr_nm = static_cast<int32_t>(model_names.size());
    int32_t hdr_tk = TOP_K;
    bin_out.write(reinterpret_cast<const char*>(&hdr_nm), sizeof(hdr_nm));
    bin_out.write(reinterpret_cast<const char*>(&hdr_tk), sizeof(hdr_tk));

    // Header de nomes (cada nome: int32 len + chars)
    for (const auto &m : model_names) {
        int32_t len = static_cast<int32_t>(m.size());
        bin_out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        bin_out.write(m.data(), len);
    }

    // ============================================================
    // Pipeline reader -> GPU -> writer
    // ============================================================
    std::queue<BatchJob> gpu_queue;
    std::mutex gpu_mtx;
    std::condition_variable cv_gpu_consumer, cv_gpu_producer;
    bool reader_finished = false;

    std::queue<WriterJob> writer_queue;
    std::mutex writer_mtx;
    std::condition_variable cv_writer;
    bool gpu_finished = false;

    int total_puzzles_processed = 0;

    auto gpu_worker = [&]() {
        while (true) {
            BatchJob job;
            {
                std::unique_lock<std::mutex> lock(gpu_mtx);
                cv_gpu_consumer.wait(lock, [&] { return !gpu_queue.empty() || reader_finished; });
                if (gpu_queue.empty() && reader_finished) break;
                job = std::move(gpu_queue.front());
                gpu_queue.pop();
            }
            cv_gpu_producer.notify_one();

            if (job.batched_meta.empty()) continue;
            int cbs = static_cast<int>(job.batched_meta.size());
            std::vector<std::vector<PuzzleResult>> batch_results(cbs,
                std::vector<PuzzleResult>(engines.size()));

            for (size_t e = 0; e < engines.size(); ++e)
                engines[e]->evaluateBatch(job.batched_inputs.data(), cbs,
                                          job.batched_meta, batch_results, e);

            {
                std::unique_lock<std::mutex> lock(writer_mtx);
                writer_queue.push({std::move(job.batched_meta), std::move(batch_results)});
            }
            cv_writer.notify_one();
        }
        {
            std::unique_lock<std::mutex> lock(writer_mtx);
            gpu_finished = true;
        }
        cv_writer.notify_all();
    };

    auto writer_worker = [&]() {
        while (true) {
            WriterJob wjob;
            {
                std::unique_lock<std::mutex> lock(writer_mtx);
                cv_writer.wait(lock, [&] { return !writer_queue.empty() || gpu_finished; });
                if (writer_queue.empty() && gpu_finished) break;
                wjob = std::move(writer_queue.front());
                writer_queue.pop();
            }

            for (size_t b = 0; b < wjob.batched_meta.size(); ++b) {
                const auto &m = wjob.batched_meta[b];
                // ---- CSV ----
                csv_out << m.puzzle_id << "," << m.rating << "," << m.solution_length;
                for (size_t e = 0; e < engines.size(); ++e) {
                    const auto &r = wjob.batch_results[b][e];
                    csv_out << "," << r.rank
                            << "," << r.prob_correct
                            << "," << r.entropy_full
                            << "," << r.entropy_legal
                            << "," << r.value;
                }
                csv_out << "\n";

                // ---- Binario (top-10) ----
                char pid_buf[32] = {0};
                std::memcpy(pid_buf, m.puzzle_id.data(),
                            std::min<size_t>(31, m.puzzle_id.size()));
                bin_out.write(pid_buf, 32);
                int32_t rating32 = m.rating;
                bin_out.write(reinterpret_cast<const char*>(&rating32), sizeof(rating32));
                for (size_t e = 0; e < engines.size(); ++e) {
                    const auto &r = wjob.batch_results[b][e];
                    bin_out.write(reinterpret_cast<const char*>(r.top_idx.data()),
                                  sizeof(int) * TOP_K);
                    bin_out.write(reinterpret_cast<const char*>(r.top_prob.data()),
                                  sizeof(float) * TOP_K);
                }
            }
            total_puzzles_processed += wjob.batched_meta.size();
            std::cerr << "\rPuzzles processados: " << total_puzzles_processed << std::flush;
        }
    };

    std::thread t_gpu(gpu_worker);
    std::thread t_writer(writer_worker);

    std::vector<float> batched_inputs;
    batched_inputs.reserve(BATCH_SIZE * CHANNELS * BOARD_SQ);
    std::vector<PuzzleMeta> batched_meta;
    batched_meta.reserve(BATCH_SIZE);

    std::string line;
    size_t line_no = 0;
    int total_skipped = 0;

    while (std::getline(std::cin, line)) {
        line_no++;
        if (line.empty() || (line_no == 1 && line.rfind("PuzzleId", 0) == 0)) continue;

        auto tokens = split_csv_fast(line);
        if (tokens.size() < 4) { total_skipped++; continue; }

        std::string puzzle_id(tokens[0]);
        std::string_view fen = tokens[1];
        std::string_view moves_str = tokens[2];
        int rating = 0;
        std::from_chars(tokens[3].data(), tokens[3].data() + tokens[3].size(), rating);

        auto moves = split_spaces_fast(moves_str);
        if (moves.size() < 2) { total_skipped++; continue; }

        int solution_length = static_cast<int>(moves.size()) - 1;

        std::string fen_str(fen);
        chess::Board board(fen_str);
        chess::Move first_move = chess::uci::uciToMove(board, std::string(moves[0]));
        if (first_move == chess::Move::NO_MOVE) { total_skipped++; continue; }
        board.makeMove(first_move);

        std::string current_fen = board.getFen();
        bool flip;
        float input[CHANNELS * BOARD_SQ];
        if (!Decoder::decode(current_fen, input, flip)) { total_skipped++; continue; }

        chess::Movelist legal_moves;
        chess::movegen::legalmoves(legal_moves, board);

        batched_inputs.insert(batched_inputs.end(), input, input + (CHANNELS * BOARD_SQ));
        batched_meta.push_back({puzzle_id, rating, solution_length, flip, legal_moves, std::string(moves[1])});

        if (batched_meta.size() == BATCH_SIZE) {
            {
                std::unique_lock<std::mutex> lock(gpu_mtx);
                cv_gpu_producer.wait(lock, [&] { return gpu_queue.size() < MAX_QUEUE_SIZE; });
                gpu_queue.push({std::move(batched_inputs), std::move(batched_meta)});
            }
            cv_gpu_consumer.notify_one();
            batched_inputs = std::vector<float>();
            batched_inputs.reserve(BATCH_SIZE * CHANNELS * BOARD_SQ);
            batched_meta = std::vector<PuzzleMeta>();
            batched_meta.reserve(BATCH_SIZE);
        }
    }

    if (!batched_meta.empty()) {
        std::unique_lock<std::mutex> lock(gpu_mtx);
        gpu_queue.push({std::move(batched_inputs), std::move(batched_meta)});
        cv_gpu_consumer.notify_one();
    }

    {
        std::unique_lock<std::mutex> lock(gpu_mtx);
        reader_finished = true;
    }
    cv_gpu_consumer.notify_all();

    t_gpu.join();
    t_writer.join();
    csv_out.close();
    bin_out.close();

    std::cerr << "\n\n======================================================\n";
    std::cerr << "Puzzles processados:  " << total_puzzles_processed << "\n";
    std::cerr << "Puzzles ignorados:    " << total_skipped << "\n";
    std::cerr << "Modelos avaliados:    " << model_names.size() << "\n";
    std::cerr << "Saidas:\n";
    std::cerr << "  - benchmark_results.csv  (metricas escalares)\n";
    std::cerr << "  - benchmark_top10.bin    (top-10 lances + probs)\n";
    std::cerr << "======================================================\n\n";

    return 0;
}
