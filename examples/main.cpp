#include "sesi_inference.h"
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1]
        : "Sources/SESIInference/Resources/sesi_cfd.mlpackage";

    constexpr int C = 3, H = 64, W = 64;

    try {
        sesi::Model model(path, C, H, W, /*compute_units=*/0);

        std::vector<float> input(static_cast<size_t>(C * H * W), 0.0f);
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.f, 1.f);
        for (auto& v : input) v = dist(rng);

        auto out = model.predict(input);

        std::cout << "SESI CoreML C++ inference OK\n";
        std::cout << "Output elements: " << out.size() << "\n";
        std::cout << "First 8 values: ";
        for (size_t i = 0; i < std::min<size_t>(8, out.size()); ++i)
            std::cout << out[i] << " ";
        std::cout << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
