#pragma once
#include <cstddef>
#include <vector>
#include <string>
#include <stdexcept>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SESIModelOpaque SESIModelOpaque;

/* compute_units: 0 = ALL (ANE+GPU+CPU), 1 = CPUAndGPU, 2 = CPUOnly */
SESIModelOpaque* sesi_create(const char* mlpackage_path,
                             int channels, int height, int width,
                             int compute_units,
                             char* error_out, size_t error_capacity);

int sesi_predict(SESIModelOpaque* model,
                 const float* input, size_t input_count,
                 float* output, size_t output_capacity,
                 size_t* output_count,
                 char* error_out, size_t error_capacity);

void sesi_destroy(SESIModelOpaque* model);

const char* sesia_last_error(void);

#ifdef __cplusplus
} // extern "C"

namespace sesi {

class InferenceError : public std::runtime_error {
public:
    explicit InferenceError(const std::string& m) : std::runtime_error(m) {}
};

class Model {
public:
    Model(const std::string& path, int C, int H, int W, int compute_units = 0)
        : C_(C), H_(H), W_(W), handle_(nullptr) {
        char err[512] = {0};
        handle_ = sesi_create(path.c_str(), C, H, W, compute_units, err, sizeof(err));
        if (!handle_) throw InferenceError(err[0] ? err : "sesi_create failed");
    }

    ~Model() { if (handle_) sesi_destroy(handle_); }

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&& o) noexcept
        : handle_(o.handle_), C_(o.C_), H_(o.H_), W_(o.W_) { o.handle_ = nullptr; }

    std::vector<float> predict(const std::vector<float>& input) {
        if (input.size() != static_cast<size_t>(C_ * H_ * W_))
            throw InferenceError("Input size mismatch");

        std::vector<float> output(static_cast<size_t>(C_ * H_ * W_));
        size_t n = 0;
        char err[512] = {0};

        int rc = sesi_predict(handle_, input.data(), input.size(),
                              output.data(), output.size(), &n, err, sizeof(err));
        if (rc != 0) throw InferenceError(err[0] ? err : "sesi_predict failed");

        output.resize(n);
        return output;
    }

private:
    SESIModelOpaque* handle_;
    int C_, H_, W_;
};

} // namespace sesi
#endif
