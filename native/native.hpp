#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Tensors are contiguous, row-major FP32 arrays.
struct Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<float> values;

    Matrix() = default;
    Matrix(int m, int n) : rows(m), cols(n), values(static_cast<size_t>(m) * n, 0) {}
};

struct Layer {
    Matrix weights;
    std::vector<float> bias;
};

struct Model {
    std::vector<Layer> layers;
    Matrix inputs;
    Matrix probabilities;
    std::vector<int> labels;
};

enum class CPUKernel { Naive, Blocked };

class CPUWorkspace {
    const Model &model;
    CPUKernel kernel;
    std::vector<Matrix> activations;

  public:
    CPUWorkspace(const Matrix &input, const Model &model, CPUKernel kernel);
    const Matrix &run();
};

void validate_product(const Matrix &a, const Matrix &b);
Matrix matmul_cpu(const Matrix &a, const Matrix &b, CPUKernel kernel);
Matrix infer_cpu(const Matrix &input, const Model &model, CPUKernel kernel);
Model read_model(const std::filesystem::path &folder);
void self_test_cpu();
std::string cpu_description();

struct GPUTiming {
    double device_ms = 0;
    double resident_wall_ms = 0;
    double transfer_wall_ms = 0;
};

// Common interface so the benchmark can compile without CUDA.
// CUDA weights and buffers are allocated once outside the benchmark.
class GPUEngine {
  public:
    virtual ~GPUEngine() = default;
    virtual GPUTiming measure() = 0;
    virtual Matrix result() const = 0;
};

#ifdef HAVE_CUDA
std::string gpu_description();
std::unique_ptr<GPUEngine> create_gpu(const Model &model, const Matrix &input, bool tiled);
void self_test_gpu();
#endif

inline double elapsed_ms(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

inline double max_error(const Matrix &a, const Matrix &b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        throw std::runtime_error("Mismatched dimensions in comparison.");
    }

    double maximum = 0;
    for (size_t i = 0; i < a.values.size(); ++i) {
        if (!std::isfinite(a.values[i]) || !std::isfinite(b.values[i])) {
            throw std::runtime_error("Non-finite numerical result.");
        }
        maximum = std::max(maximum, std::abs(double(a.values[i]) - b.values[i]));
    }
    return maximum;
}
