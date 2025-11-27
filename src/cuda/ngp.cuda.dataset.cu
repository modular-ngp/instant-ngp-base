#include "ngp.cuda.train.h"

#include <nlohmann/json.hpp>
#include <stb_image.h>
#include <fstream>
#include <iostream>

#include <tiny-cuda-nn/gpu_memory.h>

namespace ngp::cuda::hidden {
}

ngp::cuda::LoadDatasetResult ngp::cuda::load_dataset(const LoadDatasetParams& params) {
    std::vector<nlohmann::json> json_files;
    for (const auto& entry : std::filesystem::directory_iterator(params.dataset_path)) if (entry.path().extension() == ".json") json_files.push_back(nlohmann::json::parse(std::ifstream(entry.path())));

    LoadDatasetResult result;

    // Load NeRF Synthetic Dataset
    for (const auto& entry : json_files) {
        for (const auto& frame : entry["frames"]) {
            std::array<std::array<float, 3>, 4> mat{};
            for (int m = 0; m < 3; m++) for (int n = 0; n < 4; n++) mat[n][m] = static_cast<float>(frame["transform_matrix"][m][n]);
            result.xforms.emplace_back(mat);

            auto image_file_path = std::filesystem::canonical(std::filesystem::absolute((params.dataset_path / frame["file_path"].get<std::string>()).concat(".png")));
            int w, h, comp;
            uint8_t* image                     = stbi_load(image_file_path.string().c_str(), &w, &h, &comp, 4);
            const size_t n_pixels              = static_cast<size_t>(w) * static_cast<size_t>(h);
            const size_t img_size              = n_pixels * 4 * sizeof(uint8_t);
            constexpr size_t image_type_stride = sizeof(uint8_t);
            tcnn::GPUMemory<uint8_t> images_data_gpu_tmp;
            images_data_gpu_tmp.resize(img_size * image_type_stride);
            images_data_gpu_tmp.copy_from_host(image);

            auto metadata       = result.metadata.emplace_back();
            metadata.pixels     = images_data_gpu_tmp.data();
            metadata.resolution = {static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
            std::cout << "Loaded image: " << image_file_path << " (" << w << "x" << h << ")\n";
        }
    }

    result.success = true;
    result.message = "Found" + std::to_string(result.metadata.size()) + " images in dataset.";

    return result;
}

const ngp::cuda::LoadDatasetParams& ngp::cuda::LoadDatasetParams::check() const {
    if (!std::filesystem::exists(dataset_path)) throw std::runtime_error("[FATAL ERROR] - Invalid config path: " + dataset_path.string());
    if (!std::filesystem::is_directory(dataset_path)) throw std::runtime_error("[FATAL ERROR] - Config path is not a directory: " + dataset_path.string());
    {
        bool has_json = false;
        for (const auto& entry : std::filesystem::directory_iterator(dataset_path)) {
            if (entry.path().extension() == ".json") {
                has_json = true;
                break;
            }
        }
        if (!has_json) throw std::runtime_error("[FATAL ERROR] - Dataset directory does not contain any JSON files: " + dataset_path.string());
    }

    return *this;
}
