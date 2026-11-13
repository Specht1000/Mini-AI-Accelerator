#include "native.hpp"

#include <random>
#include <sstream>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

void validate_product(const Matrix &a, const Matrix &b) {
    if (a.rows < 1 || a.cols < 1 || b.rows < 1 || b.cols < 1 || a.cols != b.rows) {
        throw std::runtime_error("Invalid matrix multiplication dimensions.");
    }
}

namespace {
void matmul_into(const Matrix &a, const Matrix &b, Matrix &output, CPUKernel kernel) {
    validate_product(a, b);
    std::fill(output.values.begin(), output.values.end(), 0.0f);

    if (kernel == CPUKernel::Naive) {
        // Sequential reference: one complete dot product for each output.
        for (int i = 0; i < a.rows; ++i) {
            for (int j = 0; j < b.cols; ++j) {
                float sum = 0;
                for (int k = 0; k < a.cols; ++k) {
                    sum += a.values[i * a.cols + k] * b.values[k * b.cols + j];
                }
                output.values[i * b.cols + j] = sum;
            }
        }
        return;
    }

    // The i-k-j order accesses B and output contiguously, enabling SIMD.
    // Optional OpenMP parallelism distributes independent rows.
    // Parallel region startup is included in the measurement.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < a.rows; ++i) {
        for (int kk = 0; kk < a.cols; kk += 32) {
            for (int k = kk; k < std::min(kk + 32, a.cols); ++k) {
                const float value = a.values[i * a.cols + k];
                for (int j = 0; j < b.cols; ++j) {
                    output.values[i * b.cols + j] += value * b.values[k * b.cols + j];
                }
            }
        }
    }
}
} // namespace

Matrix matmul_cpu(const Matrix &a, const Matrix &b, CPUKernel kernel) {
    Matrix output(a.rows, b.cols);
    matmul_into(a, b, output, kernel);
    return output;
}

Matrix infer_cpu(const Matrix &input, const Model &model, CPUKernel kernel) {
    CPUWorkspace workspace(input, model, kernel);
    return workspace.run();
}

CPUWorkspace::CPUWorkspace(const Matrix &input, const Model &model, CPUKernel kernel) : model(model), kernel(kernel) {
    activations.push_back(input);
    for (const auto &layer : model.layers) {
        activations.emplace_back(input.rows, layer.weights.cols);
    }
}

const Matrix &CPUWorkspace::run() {

    for (size_t layer_index = 0; layer_index < model.layers.size(); ++layer_index) {
        const auto &layer = model.layers[layer_index];
        auto &output = activations[layer_index + 1];
        matmul_into(activations[layer_index], layer.weights, output, kernel);

        for (int i = 0; i < output.rows; ++i) {
            for (int j = 0; j < output.cols; ++j) {
                output.values[i * output.cols + j] += layer.bias[j];
            }

            if (layer_index + 1 < model.layers.size()) {
                for (int j = 0; j < output.cols; ++j) {
                    auto &value = output.values[i * output.cols + j];
                    value = std::max(0.0f, value);
                }
            }
            else {
                // Stable softmax: subtract the largest logit before exponentiation.
                auto begin = output.values.begin() + i * output.cols;
                const float maximum = *std::max_element(begin, begin + output.cols);
                float sum = 0;
                for (int j = 0; j < output.cols; ++j) {
                    auto &value = output.values[i * output.cols + j];
                    value = std::exp(value - maximum);
                    sum += value;
                }
                for (int j = 0; j < output.cols; ++j) {
                    output.values[i * output.cols + j] /= sum;
                }
            }
        }
    }
    return activations.back();
}

namespace {
Matrix read_matrix(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);
    uint32_t shape[2] = {};
    file.read(reinterpret_cast<char *>(shape), sizeof(shape));
    if (!file || shape[0] < 1 || shape[1] < 1 || uint64_t(shape[0]) * shape[1] > 100000000) {
        throw std::runtime_error("Invalid matrix file: " + path.string());
    }
    Matrix matrix(static_cast<int>(shape[0]), static_cast<int>(shape[1]));
    file.read(reinterpret_cast<char *>(matrix.values.data()), static_cast<std::streamsize>(matrix.values.size() * sizeof(float)));
    if (!file || file.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("Incorrect file size: " + path.string());
    }
    for (float value : matrix.values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("Input contains a non-finite value.");
        }
    }
    return matrix;
}
} // namespace

Model read_model(const std::filesystem::path &folder) {
    std::ifstream manifest(folder / "model.txt");
    int count = 0;
    manifest >> count;
    if (!manifest || count < 1 || count > 32) {
        throw std::runtime_error("Missing or invalid model manifest.");
    }
    Model model;
    for (int i = 0; i < count; ++i) {
        auto weights = read_matrix(folder / ("W" + std::to_string(i) + ".bin"));
        auto bias = read_matrix(folder / ("b" + std::to_string(i) + ".bin"));
        if (bias.rows != 1 || bias.cols != weights.cols) {
            throw std::runtime_error("Incorrect bias dimensions.");
        }
        model.layers.push_back({std::move(weights), std::move(bias.values)});
    }
    model.inputs = read_matrix(folder / "inputs.bin");
    model.probabilities = read_matrix(folder / "probabilities.bin");
    Matrix labels = read_matrix(folder / "labels.bin");
    int features = model.inputs.cols;
    for (const auto &layer : model.layers) {
        if (layer.weights.rows != features) {
            throw std::runtime_error("Incompatible layer dimensions.");
        }
        features = layer.weights.cols;
    }
    if (model.probabilities.rows != model.inputs.rows || model.probabilities.cols != features || labels.rows != model.inputs.rows || labels.cols != 1) {
        throw std::runtime_error("Incorrect sample or reference dimensions.");
    }
    for (float label : labels.values) {
        if (label < 0 || label >= features || label != std::floor(label)) {
            throw std::runtime_error("Invalid label.");
        }
        model.labels.push_back(static_cast<int>(label));
    }
    return model;
}

void self_test_cpu() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-1, 1);
    for (int m : {1, 3, 17}) {
        for (int k : {1, 7, 33}) {
            for (int n : {1, 10, 35}) {
                Matrix a(m, k), b(k, n);
                for (auto &value : a.values) {
                    value = distribution(rng);
                }
                for (auto &value : b.values) {
                    value = distribution(rng);
                }
                if (max_error(matmul_cpu(a, b, CPUKernel::Naive), matmul_cpu(a, b, CPUKernel::Blocked)) > 1e-4) {
                    throw std::runtime_error("CPU matrix multiplication test failed.");
                }
            }
        }
    }
}

std::string cpu_description() {
    std::ostringstream result;
    result << "hardware_threads=" << std::thread::hardware_concurrency();
#ifdef _OPENMP
    result << ";openmp_threads=" << omp_get_max_threads();
#else
    result << ";openmp=off;optimized=single_thread_simd_compiler";
#endif
    return result.str();
}
