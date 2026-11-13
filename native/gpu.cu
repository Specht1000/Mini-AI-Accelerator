#include "native.hpp"

#include <cuda_runtime.h>
#include <random>
#include <sstream>

namespace {
constexpr int TILE = 16;

void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(status));
    }
}

// RAII releases resources even if a later operation throws an exception.
struct DeviceBuffer {
    float *pointer = nullptr;
    explicit DeviceBuffer(size_t count) {
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&pointer), count * sizeof(float)));
    }
    ~DeviceBuffer() {
        if (pointer) {
            cudaFree(pointer);
        }
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};

struct PinnedBuffer {
    float *pointer = nullptr;
    explicit PinnedBuffer(size_t count) {
        cuda_check(cudaMallocHost(reinterpret_cast<void **>(&pointer), count * sizeof(float)));
    }
    ~PinnedBuffer() {
        if (pointer) {
            cudaFreeHost(pointer);
        }
    }
};

struct Stream {
    cudaStream_t handle = nullptr;
    Stream() {
        cuda_check(cudaStreamCreate(&handle));
    }
    ~Stream() {
        if (handle) {
            cudaStreamDestroy(handle);
        }
    }
};

struct Event {
    cudaEvent_t handle = nullptr;
    Event() {
        cuda_check(cudaEventCreate(&handle));
    }
    ~Event() {
        if (handle) {
            cudaEventDestroy(handle);
        }
    }
};

__global__ void matmul_naive(const float *a, const float *b, float *c, int m, int k, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0;
        for (int reduction = 0; reduction < k; ++reduction) {
            sum += a[row * k + reduction] * b[reduction * n + col];
        }
        c[row * n + col] = sum;
    }
}

__global__ void matmul_tiled(const float *a, const float *b, float *c, int m, int k, int n) {
    __shared__ float local_a[TILE][TILE];
    __shared__ float local_b[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0;

    for (int offset = 0; offset < k; offset += TILE) {
        const int a_col = offset + threadIdx.x;
        const int b_row = offset + threadIdx.y;

        // Boundary threads load zeros and also participate in barriers.
        local_a[threadIdx.y][threadIdx.x] = row < m && a_col < k ? a[row * k + a_col] : 0;
        local_b[threadIdx.y][threadIdx.x] = b_row < k && col < n ? b[b_row * n + col] : 0;
        __syncthreads();

        for (int reduction = 0; reduction < TILE; ++reduction) {
            sum += local_a[threadIdx.y][reduction] * local_b[reduction][threadIdx.x];
        }

        // Do not overwrite tiles while other warps are still using them.
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

__global__ void bias_activation(float *values, const float *bias, int count, int columns, bool relu) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        float value = values[index] + bias[index % columns];
        values[index] = relu && value < 0 ? 0 : value;
    }
}

__global__ void softmax_rows(float *values, int rows, int columns) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    // One thread per sample is sufficient for the ten-class output.
    float maximum = values[row * columns];
    for (int j = 1; j < columns; ++j) {
        maximum = fmaxf(maximum, values[row * columns + j]);
    }
    float sum = 0;
    for (int j = 0; j < columns; ++j) {
        float value = expf(values[row * columns + j] - maximum);
        values[row * columns + j] = value;
        sum += value;
    }
    for (int j = 0; j < columns; ++j) {
        values[row * columns + j] /= sum;
    }
}

void launch_product(const float *a, const float *b, float *c, int m, int k, int n, bool tiled, cudaStream_t stream) {
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
    if (tiled) {
        matmul_tiled<<<grid, block, 0, stream>>>(a, b, c, m, k, n);
    }
    else {
        matmul_naive<<<grid, block, 0, stream>>>(a, b, c, m, k, n);
    }
    cuda_check(cudaGetLastError());
}

class CUDAEngine final : public GPUEngine {
    const Model &model;
    int batch;
    bool tiled;
    Stream stream;
    Event begin_event;
    Event end_event;
    std::vector<std::unique_ptr<DeviceBuffer>> weights;
    std::vector<std::unique_ptr<DeviceBuffer>> biases;
    std::vector<std::unique_ptr<DeviceBuffer>> activations;
    PinnedBuffer host_input;
    PinnedBuffer host_output;

    void network() {
        int inputs = model.inputs.cols;
        for (size_t index = 0; index < model.layers.size(); ++index) {
            const int outputs = model.layers[index].weights.cols;
            float *output = activations[index + 1]->pointer;
            launch_product(activations[index]->pointer, weights[index]->pointer, output, batch, inputs, outputs, tiled, stream.handle);
            const bool relu = index + 1 < model.layers.size();
            bias_activation<<<(batch * outputs + 255) / 256, 256, 0, stream.handle>>>(output, biases[index]->pointer, batch * outputs, outputs, relu);
            cuda_check(cudaGetLastError());
            if (!relu) {
                softmax_rows<<<(batch + 127) / 128, 128, 0, stream.handle>>>(output, batch, outputs);
                cuda_check(cudaGetLastError());
            }
            inputs = outputs;
        }
    }

  public:
    CUDAEngine(const Model &model, const Matrix &input, bool tiled) : model(model), batch(input.rows), tiled(tiled), host_input(input.values.size()), host_output(static_cast<size_t>(input.rows) * model.layers.back().weights.cols) {
        std::copy(input.values.begin(), input.values.end(), host_input.pointer);
        activations.push_back(std::make_unique<DeviceBuffer>(input.values.size()));

        for (const auto &layer : model.layers) {
            weights.push_back(std::make_unique<DeviceBuffer>(layer.weights.values.size()));
            biases.push_back(std::make_unique<DeviceBuffer>(layer.bias.size()));
            activations.push_back(std::make_unique<DeviceBuffer>(static_cast<size_t>(batch) * layer.weights.cols));
            cuda_check(cudaMemcpyAsync(weights.back()->pointer, layer.weights.values.data(), layer.weights.values.size() * sizeof(float), cudaMemcpyHostToDevice, stream.handle));
            cuda_check(cudaMemcpyAsync(biases.back()->pointer, layer.bias.data(), layer.bias.size() * sizeof(float), cudaMemcpyHostToDevice, stream.handle));
        }

        cuda_check(cudaMemcpyAsync(activations.front()->pointer, host_input.pointer, input.values.size() * sizeof(float), cudaMemcpyHostToDevice, stream.handle));
        cuda_check(cudaStreamSynchronize(stream.handle));
    }

    GPUTiming measure() override {
        GPUTiming timing;

        // First scenario: input, weights, and intermediates already reside on GPU.
        cuda_check(cudaEventRecord(begin_event.handle, stream.handle));
        const auto resident_start = std::chrono::steady_clock::now();
        network();
        cuda_check(cudaEventRecord(end_event.handle, stream.handle));
        cuda_check(cudaEventSynchronize(end_event.handle));
        const auto resident_end = std::chrono::steady_clock::now();
        float device_ms = 0;
        cuda_check(cudaEventElapsedTime(&device_ms, begin_event.handle, end_event.handle));
        timing.device_ms = device_ms;
        timing.resident_wall_ms = elapsed_ms(resident_start, resident_end);

        // Second scenario: wall time includes H2D, network, D2H, and synchronization.
        // Weights remain resident; cudaMalloc is excluded from this region.
        const auto transfer_start = std::chrono::steady_clock::now();
        cuda_check(cudaMemcpyAsync(activations.front()->pointer, host_input.pointer, static_cast<size_t>(batch) * model.inputs.cols * sizeof(float), cudaMemcpyHostToDevice, stream.handle));
        network();
        cuda_check(cudaMemcpyAsync(host_output.pointer, activations.back()->pointer, static_cast<size_t>(batch) * model.layers.back().weights.cols * sizeof(float), cudaMemcpyDeviceToHost, stream.handle));
        cuda_check(cudaStreamSynchronize(stream.handle));
        timing.transfer_wall_ms = elapsed_ms(transfer_start, std::chrono::steady_clock::now());
        return timing;
    }

    Matrix result() const override {
        Matrix output(batch, model.layers.back().weights.cols);
        std::copy(host_output.pointer, host_output.pointer + output.values.size(), output.values.begin());
        return output;
    }
};
} // namespace

std::string gpu_description() {
    int count = 0;
    cuda_check(cudaGetDeviceCount(&count));
    if (count < 1) {
        throw std::runtime_error("No CUDA GPU found.");
    }
    cudaDeviceProp properties = {};
    cuda_check(cudaGetDeviceProperties(&properties, 0));
    cuda_check(cudaSetDevice(0));
    std::ostringstream result;
    result << properties.name << ";cc=" << properties.major << '.' << properties.minor << ";SMs=" << properties.multiProcessorCount;
    return result.str();
}

std::unique_ptr<GPUEngine> create_gpu(const Model &model, const Matrix &input, bool tiled) {
    return std::make_unique<CUDAEngine>(model, input, tiled);
}

void self_test_gpu() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-1, 1);
    for (int m : {1, 3, 17}) {
        for (int k : {1, 7, 33}) {
            for (int n : {1, 10, 35}) {
                Matrix a(m, k), b(k, n), output(m, n);
                for (auto &value : a.values) {
                    value = distribution(rng);
                }
                for (auto &value : b.values) {
                    value = distribution(rng);
                }
                DeviceBuffer da(a.values.size()), db(b.values.size()), dc(output.values.size());
                cuda_check(cudaMemcpy(da.pointer, a.values.data(), a.values.size() * sizeof(float), cudaMemcpyHostToDevice));
                cuda_check(cudaMemcpy(db.pointer, b.values.data(), b.values.size() * sizeof(float), cudaMemcpyHostToDevice));
                const auto reference = matmul_cpu(a, b, CPUKernel::Naive);
                for (bool tiled : {false, true}) {
                    launch_product(da.pointer, db.pointer, dc.pointer, m, k, n, tiled, nullptr);
                    cuda_check(cudaMemcpy(output.values.data(), dc.pointer, output.values.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    if (max_error(output, reference) > 1e-4) {
                        throw std::runtime_error("CUDA numerical test failed.");
                    }
                }
            }
        }
    }
}
