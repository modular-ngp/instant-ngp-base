#include "ngp.cuda.train.h"

#include <nlohmann/json.hpp>
#include <stb_image.h>
#include <fstream>
#include <iostream>

#include <tiny-cuda-nn/gpu_memory.h>

namespace ngp::cuda::hidden {
    LoadDatasetResult load_dataset_nerf_synthetic(const std::filesystem::path& dataset_path) {
        std::vector<nlohmann::json> json_files;
        for (const auto& entry : std::filesystem::directory_iterator(dataset_path)) if (entry.path().extension() == ".json") json_files.push_back(nlohmann::json::parse(std::ifstream(entry.path())));

        LoadDatasetResult result;
        for (const auto& entry : json_files) {
            for (const auto& frame : entry["frames"]) {
                std::array<std::array<float, 3>, 4>& _mat = result.xforms.emplace_back();
                for (int m = 0; m < 3; m++) for (int n = 0; n < 4; n++) _mat[n][m] = static_cast<float>(frame["transform_matrix"][m][n]);

                int _w, _h, _comp;
                const std::filesystem::path _image_file_path = std::filesystem::canonical(std::filesystem::absolute((dataset_path / frame["file_path"].get<std::string>()).concat(".png")));
                const uint8_t* _image_data_cpu               = stbi_load(_image_file_path.string().c_str(), &_w, &_h, &_comp, 4);

                tcnn::GPUMemory<uint8_t> images_data_gpu;
                images_data_gpu.resize(_w * _h * _comp * sizeof(uint8_t));
                images_data_gpu.copy_from_host(_image_data_cpu);

                LoadDatasetResult::TrainingImageMetadata& metadata = result.metadata.emplace_back();
                metadata.pixels                                    = images_data_gpu.data();
                metadata.resolution                                = {static_cast<uint32_t>(_w), static_cast<uint32_t>(_h)};

                std::array<float, 2>& focal_length = metadata.focal_length;
                focal_length[0]                    = static_cast<float>(_w) / (2.0f * tan(static_cast<float>(entry["camera_angle_x"]) * 0.5f));
                focal_length[1]                    = 0.f;
                std::cout << "Loaded image: " << _image_file_path << " (" << _w << "x" << _h << "), focal length: (" << focal_length[0] << ", " << focal_length[1] << ")\n";
            }
        }
        result.success = true;
        result.message = "Found" + std::to_string(result.metadata.size()) + " images in dataset.";
        return result;
    }
}

ngp::cuda::LoadDatasetResult ngp::cuda::load_dataset(const LoadDatasetParams& params) {
    switch (params.dataset_type) {
    case LoadDatasetParams::DatasetType::NeRfSynthetic: return hidden::load_dataset_nerf_synthetic(params.dataset_path);
    case LoadDatasetParams::DatasetType::LLFF:
    case LoadDatasetParams::DatasetType::NSVF:
    case LoadDatasetParams::DatasetType::Unknown: throw std::runtime_error("[FATAL ERROR] - 1");
    default: throw std::runtime_error("[FATAL ERROR] - 2");
    }
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
