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

constexpr int CHANNELS = 19;
constexpr int BOARD_SQ = 64;
constexpr int POLICY_SIZE = 4224;
constexpr int BATCH_SIZE = 1024;
constexpr int MAX_QUEUE_SIZE = 4;

static int charToNibble(char c) {
    switch (c) {
        case 'P': return 1; case 'N': return 2; case 'B': return 3; case 'R': return 4;
        case 'Q': return 5; case 'K': return 6; case 'p': return 7; case 'n': return 8;
        case 'b': return 9; case 'r': return 10; case 'q': return 11; case 'k': return 12;
        default: return 0;
    }
}

static const int FLIP_PIECE[13] = {0, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6};

// OTIMIZAÇÃO: Usando string_view para ZERO alocações de memória!
static std::vector<std::string_view> split_csv_fast(std::string_view line) {
    std::vector<std::string_view> out;
    out.reserve(5); // CSV tem poucos campos
    size_t start = 0;
    size_t pos = line.find(',');
    while (pos != std::string_view::npos) {
        out.emplace_back(line.data() + start, pos - start);
        start = pos + 1;
        pos = line.find(',', start);
    }
    out.emplace_back(line.data() + start, line.size() - start);
    return out;
}

// OTIMIZAÇÃO: Separador de lances super rápido, sem std::stringstream
static std::vector<std::string_view> split_spaces_fast(std::string_view str) {
    std::vector<std::string_view> out;
    size_t start = 0;
    size_t pos = str.find(' ');
    while (pos != std::string_view::npos) {
        if (pos > start) out.emplace_back(str.data() + start, pos - start);
        start = pos + 1;
        pos = str.find(' ', start);
    }
    if (start < str.size()) out.emplace_back(str.data() + start, str.size() - start);
    return out;
}

struct Decoder {
    // OTIMIZAÇÃO: Sem istringstream. Varre a string nativamente.
    static bool decode(std::string_view fen, float *output, bool &flip) {
        std::array<uint8_t, 64> compact{};
        auto parts = split_spaces_fast(fen);
        if(parts.size() < 4) return false;

        std::string_view p0 = parts[0], p1 = parts[1], p2 = parts[2], p3 = parts[3];

        int rank = 7, file = 0;
        for (char c : p0) {
            if (c == '/') { rank--; file = 0; }
            else if (c >= '1' && c <= '8') file += c - '0';
            else {
                int sq = rank * 8 + file;
                int piece = charToNibble(c);
                int idx = sq >> 1;
                if (sq & 1) compact[idx] |= piece << 4;
                else compact[idx] |= piece;
                file++;
            }
        }

        flip = (p1 == "b");
        if (flip) compact[33] |= 1;

        uint8_t castle = 0;
        for (char c : p2) {
            if (c == 'K') castle |= 2;
            if (c == 'Q') castle |= 4;
            if (c == 'k') castle |= 8;
            if (c == 'q') castle |= 16;
        }
        if (flip) compact[33] |= ((castle & 0x06) << 2) | ((castle & 0x18) >> 2);
        else compact[33] |= castle;

        compact[32] = 255;
        if (p3 != "-") {
            int ep = (p3[1] - '1') * 8 + (p3[0] - 'a');
            compact[32] = flip ? (ep ^ 56) : ep;
        }

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
                else compact[idx] |= fp;
            }
        }

        std::memset(output, 0, CHANNELS * BOARD_SQ * sizeof(float));
        uint8_t boardArr[64];
        for (int sq = 0; sq < 64; sq++) {
            uint8_t b = compact[sq >> 1];
            boardArr[sq] = (sq & 1) ? (b >> 4) : (b & 0x0F);
        }

        uint8_t ep = compact[32], flags = compact[33];
        auto write = [&](int c, int sq, float v) { output[c * 64 + sq] = v; };

        for (int sq = 0; sq < 64; sq++) {
            uint8_t p = boardArr[sq];
            if (p >= 1 && p <= 12) write(p - 1, sq, 1.0f);
            if (ep == sq && ep < 64) write(12, sq, 1.0f);
            if (flags & 1) write(13, sq, 1.0f);
            if (flags & 2) write(14, sq, 1.0f);
            if (flags & 4) write(15, sq, 1.0f);
            if (flags & 8) write(16, sq, 1.0f);
        }
        return true;
    }
};

static int uciToIndex(const std::string &uci, bool flip) {
    if (uci.length() < 4) return -1;
    int from = (uci[1] - '1') * 8 + (uci[0] - 'a');
    int to = (uci[3] - '1') * 8 + (uci[2] - 'a');
    if (flip) { from ^= 56; to ^= 56; }

    if (uci.length() == 4) return to * 64 + from;
    else if (uci.length() == 5) {
        int p = 0;
        if (uci[4] == 'n') p = 0;
        else if (uci[4] == 'b') p = 1;
        else if (uci[4] == 'r') p = 2;
        else if (uci[4] == 'q') p = 3;
        int ff = from - 48;
        int dir = to - 56 - ff + 1;
        int rem = dir * 4 + p;
        return 4096 + (ff * 12 + rem);
    }
    return -1;
}

struct ScoredMove {
    std::string uci;
    float prob; // Apesar do nome, agora armazena os logits puros!
};

struct PuzzleMeta {
    std::string puzzle_id;
    int rating;
    int solution_length;
    bool flip;
    chess::Movelist legal_moves;
    std::string correct_move;
};

struct BatchJob {
    std::vector<float> batched_inputs;
    std::vector<PuzzleMeta> batched_meta;
};

struct WriterJob {
    std::vector<PuzzleMeta> batched_meta;
    std::vector<std::vector<int>> batch_ranks;
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
                       std::vector<std::vector<int>> &batch_ranks, int model_index) {

        std::array<int64_t, 4> shape = {current_batch_size, CHANNELS, 8, 8};
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto tensor = Ort::Value::CreateTensor<float>(mem, batched_inputs, current_batch_size * CHANNELS * BOARD_SQ, shape.data(), shape.size());

        const char *in_names[] = {"board"};
        const char *out_names[] = {"policy", "value"};

        auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 2);
        float *policy_batch = outputs[0].GetTensorMutableData<float>();

        for (int b = 0; b < current_batch_size; ++b) {
            float *policy = policy_batch + (b * POLICY_SIZE);
            const auto &m = meta_batch[b];

            // OTIMIZAÇÃO: Removido o loop duplo e o cálculo de Softmax (std::exp).
            std::vector<ScoredMove> ranked_moves;
            ranked_moves.reserve(m.legal_moves.size());

            for (const auto &move : m.legal_moves) {
                std::string uci_str = chess::uci::moveToUci(move);
                int idx = uciToIndex(uci_str, m.flip);
                if (idx >= 0 && idx < POLICY_SIZE) {
                    ranked_moves.push_back({uci_str, policy[idx]}); // Salva o logit cru!
                }
            }

            // Ordena usando os logits puros. Extremamente rápido!
            std::sort(ranked_moves.begin(), ranked_moves.end(), [](const ScoredMove &a, const ScoredMove &b) {
                return a.prob > b.prob; 
            });

            int rank_found = 0;
            for (size_t i = 0; i < ranked_moves.size(); i++) {
                if (ranked_moves[i].uci == m.correct_move) {
                    rank_found = static_cast<int>(i + 1);
                    break;
                }
            }
            batch_ranks[b][model_index] = rank_found;
        }
    }
};

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    std::cerr << "[INFO] Iniciando PIPELINE OTIMIZADO" << std::endl;

    std::vector<std::string> model_names;
    for (const auto &entry : std::filesystem::directory_iterator(".")) {
        if (entry.path().extension() == ".onnx") {
            model_names.push_back(entry.path().filename().string());
        }
    }

    if (model_names.empty()) return 1;
    std::sort(model_names.begin(), model_names.end());

    std::vector<std::unique_ptr<BenchmarkEngine>> engines;
    for (size_t i = 0; i < model_names.size(); ++i) {
        std::cerr << "[INFO] Carregando modelo na VRAM: " << model_names[i] << "..." << std::endl;
        engines.push_back(std::make_unique<BenchmarkEngine>(model_names[i]));
    }

    std::ofstream csv_out("benchmark_results.csv");
    csv_out << "PuzzleId,Rating,SolutionLength";
    for (const auto &m_name : model_names) csv_out << "," << m_name;
    csv_out << "\n";

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

            int current_batch_size = static_cast<int>(job.batched_meta.size());
            std::vector<std::vector<int>> batch_ranks(current_batch_size, std::vector<int>(engines.size(), 0));

            for (size_t e = 0; e < engines.size(); ++e) {
                engines[e]->evaluateBatch(job.batched_inputs.data(), current_batch_size, job.batched_meta, batch_ranks, e);
            }

            {
                std::unique_lock<std::mutex> lock(writer_mtx);
                writer_queue.push({std::move(job.batched_meta), std::move(batch_ranks)});
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
                csv_out << wjob.batched_meta[b].puzzle_id << ","
                        << wjob.batched_meta[b].rating << ","
                        << wjob.batched_meta[b].solution_length;
                for (size_t e = 0; e < engines.size(); ++e) {
                    csv_out << "," << wjob.batch_ranks[b][e];
                }
                csv_out << "\n";
            }
            total_puzzles_processed += wjob.batched_meta.size();
            std::cerr << "\rPuzzles salvos: " << total_puzzles_processed << std::flush;
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

    while (std::getline(std::cin, line)) {
        line_no++;
        if (line.empty() || (line_no == 1 && line.rfind("PuzzleId", 0) == 0)) continue;

        auto tokens = split_csv_fast(line);
        if (tokens.size() < 4) continue;

        std::string puzzle_id(tokens[0]);
        std::string_view fen = tokens[1];
        std::string_view moves_str = tokens[2];
        
        int rating = 0;
        std::from_chars(tokens[3].data(), tokens[3].data() + tokens[3].size(), rating);

        auto moves = split_spaces_fast(moves_str);
        if (moves.size() < 2) continue;

        int solution_length = moves.size() - 1;

        std::string fen_str(fen);
        chess::Board board(fen_str);
        chess::Move first_move = chess::uci::uciToMove(board, std::string(moves[0]));
        
        if (first_move == chess::Move::NO_MOVE) continue;
        board.makeMove(first_move);

        std::string current_fen = board.getFen();
        bool flip;
        float input[CHANNELS * BOARD_SQ];
        Decoder::decode(current_fen, input, flip);
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

    std::cerr << "\n\n======================================================\n";
    std::cerr << "Total de puzzles processados: " << total_puzzles_processed << "\n";
    std::cerr << "======================================================\n\n";

    return 0;
}