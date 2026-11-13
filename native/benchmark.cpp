#include "native.hpp"

#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

namespace {
double percentile(std::vector<double> samples, double fraction) {
    std::sort(samples.begin(), samples.end());
    const double index = fraction * (samples.size() - 1);
    const size_t low = static_cast<size_t>(index);
    const size_t high = std::min(low + 1, samples.size() - 1);
    return samples[low] + (samples[high] - samples[low]) * (index - low);
}

int argmax(const Matrix &matrix, int row) {
    auto start = matrix.values.begin() + row * matrix.cols;
    return static_cast<int>(std::max_element(start, start + matrix.cols) - start);
}

double accuracy(const Matrix &output, const Model &model) {
    int correct = 0;
    for (int i = 0; i < output.rows; ++i) {
        correct += argmax(output, i) == model.labels[i % model.inputs.rows];
    }
    return double(correct) / output.rows;
}

void write_row(std::ostream &csv, int batch, const std::string &algorithm, const std::string &metric, const std::vector<double> &samples, double error, double correct) {
    csv << batch << ',' << algorithm << ',' << metric << ',' << samples.size() << ',' << percentile(samples, .5) << ',' << percentile(samples, .1) << ',' << percentile(samples, .9) << ',' << error << ',' << correct << '\n';
    std::cout << "batch=" << batch << " algorithm=" << algorithm << " metric=" << metric << " median_ms=" << percentile(samples, .5) << '\n';
}
} // namespace

int main(int argc, char **argv) {
    try {
        std::filesystem::path data = "data/native";
        std::filesystem::path output = "results/native";
        int repetitions = 21;
        int warmup = 5;
        bool cpu_only = false;
        bool self_test = false;
        bool require_gpu = false;
        std::vector<int> batches = {1, 8, 64, 128, 256, 1024};

        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--cpu-only") {
                cpu_only = true;
            }
            else if (argument == "--require-gpu") {
                require_gpu = true;
            }
            else if (argument == "--self-test") {
                self_test = true;
            }
            else if (argument == "--help") {
                std::cout << "--data FOLDER --output FOLDER --repetitions 21 --warmup 5 "
                             "--batches 1,8,64,128,256,1024 --cpu-only --require-gpu --self-test\n";
                return 0;
            }
            else {
                if (i + 1 >= argc) {
                    throw std::runtime_error("Option requires a value: " + argument);
                }
                const std::string value = argv[++i];
                if (argument == "--data") {
                    data = value;
                }
                else if (argument == "--output") {
                    output = value;
                }
                else if (argument == "--repetitions") {
                    repetitions = std::stoi(value);
                }
                else if (argument == "--warmup") {
                    warmup = std::stoi(value);
                }
                else if (argument == "--batches") {
                    batches.clear();
                    std::stringstream stream(value);
                    std::string item;
                    while (std::getline(stream, item, ',')) {
                        batches.push_back(std::stoi(item));
                    }
                }
                else {
                    throw std::runtime_error("Unknown option: " + argument);
                }
            }
        }

        if (repetitions < 3 || warmup < 1 || batches.empty()) {
            throw std::runtime_error("Use at least 3 repetitions, 1 warmup, and one batch.");
        }
        for (int batch : batches) {
            if (batch < 1 || batch > 100000) {
                throw std::runtime_error("Batch size outside range 1..100000.");
            }
        }

        self_test_cpu();
        bool gpu_available = false;
        std::string gpu_info = "not compiled with CUDA";
#ifdef HAVE_CUDA
        if (!cpu_only) {
            try {
                gpu_info = gpu_description();
                gpu_available = true;
            }
            catch (const std::exception &error) {
                gpu_info = error.what();
                std::cerr << "GPU indisponivel: " << gpu_info << '\n';
            }
        }
        else {
            gpu_info = "desativada por --cpu-only";
        }
        if (gpu_available) {
            self_test_gpu();
        }
#endif
        if (require_gpu && !gpu_available) {
            throw std::runtime_error("GPU required, but CUDA could not run.");
        }
        if (self_test) {
            std::cout << "CPU: 27 matrix multiplication tests passed. GPU: " << (gpu_available ? "54 matrix multiplication tests passed" : "not tested") << '\n';
            return 0;
        }

        const Model model = read_model(data);
        std::filesystem::create_directories(output);
        std::ofstream csv(output / "benchmark.csv");
        if (!csv) {
            throw std::runtime_error("Could not create report.");
        }
        csv << std::setprecision(12);
        csv << "batch,algorithm,metric,repetitions,median_ms,p10_ms,p90_ms,max_error,accuracy\n";
        std::ofstream metadata(output / "metadata.txt");
        metadata << "cpu=" << cpu_description() << "\ngpu=" << gpu_info << "\ngpu_executed=" << gpu_available << "\nrepetitions=" << repetitions << "\nwarmup=" << warmup << "\nprecision=FP32\n";

        for (int batch : batches) {
            Matrix input(batch, model.inputs.cols);
            Matrix reference(batch, model.probabilities.cols);
            for (int i = 0; i < batch; ++i) {
                const int source = i % model.inputs.rows;
                std::copy_n(model.inputs.values.begin() + source * input.cols, input.cols, input.values.begin() + i * input.cols);
                std::copy_n(model.probabilities.values.begin() + source * reference.cols, reference.cols, reference.values.begin() + i * reference.cols);
            }

            // Buffers and file reads are outside timed regions.
            for (CPUKernel kernel : {CPUKernel::Naive, CPUKernel::Blocked}) {
                CPUWorkspace workspace(input, model, kernel);
                for (int i = 0; i < warmup; ++i) {
                    workspace.run();
                }
                std::vector<double> samples;
                for (int i = 0; i < repetitions; ++i) {
                    const auto start = std::chrono::steady_clock::now();
                    workspace.run();
                    samples.push_back(elapsed_ms(start, std::chrono::steady_clock::now()));
                }
                const Matrix result = workspace.run();
                const double error = max_error(result, reference);
                if (error > 1e-4) {
                    throw std::runtime_error("CPU inference differs from the Python reference.");
                }
                write_row(csv, batch, kernel == CPUKernel::Naive ? "cpu_naive" : "cpu_optimized", "wall", samples, error, accuracy(result, model));
            }

#ifdef HAVE_CUDA
            if (gpu_available) {
                for (bool tiled : {false, true}) {
                    auto engine = create_gpu(model, input, tiled);
                    for (int i = 0; i < warmup; ++i) {
                        engine->measure();
                    }
                    std::vector<double> device, resident, transfers;
                    for (int i = 0; i < repetitions; ++i) {
                        const auto timing = engine->measure();
                        device.push_back(timing.device_ms);
                        resident.push_back(timing.resident_wall_ms);
                        transfers.push_back(timing.transfer_wall_ms);
                    }
                    const Matrix result = engine->result();
                    const double error = max_error(result, reference);
                    if (error > 1e-4) {
                        throw std::runtime_error("CUDA inference differs from the Python reference.");
                    }
                    const std::string name = tiled ? "gpu_tiled" : "gpu_naive";
                    write_row(csv, batch, name, "device_events", device, error, accuracy(result, model));
                    write_row(csv, batch, name, "resident_wall", resident, error, accuracy(result, model));
                    write_row(csv, batch, name, "transfer_wall", transfers, error, accuracy(result, model));
                }
            }
#endif
        }
        std::cout << "Measurement report saved to " << output.string() << '\n';
        return 0;
    }
    catch (const std::exception &error) {
        std::cerr << "Erro: " << error.what() << '\n';
        return 1;
    }
}
