#include "ngp.dataset.h"

#include <nlohmann/json.hpp>
#include <stb_image.h>
#include <fstream>

namespace ngp::hidden {
    LoadDatasetResult load_dataset_nerf_synthetic(const std::filesystem::path& dataset_path) {
        std::vector<nlohmann::json> json_files;
        for (const auto& entry : std::filesystem::directory_iterator(dataset_path)) if (entry.path().extension() == ".json") json_files.push_back(nlohmann::json::parse(std::ifstream(entry.path())));

        LoadDatasetResult result;
        for (const auto& entry : json_files) {
            for (const auto& frame : entry["frames"]) {
                LoadDatasetResult::Dataset& dataset = result.dataset.emplace_back();
                for (int m = 0; m < 3; m++) for (int n = 0; n < 4; n++) dataset.xform[n][m] = static_cast<float>(frame["transform_matrix"][m][n]);

                int _w, _h, _comp;
                const std::filesystem::path _image_file_path = std::filesystem::canonical(std::filesystem::absolute((dataset_path / frame["file_path"].get<std::string>()).concat(".png")));
                dataset.pixels                               = stbi_load(_image_file_path.string().c_str(), &_w, &_h, &_comp, 4);
                dataset.resolution                           = {static_cast<uint32_t>(_w), static_cast<uint32_t>(_h)};
                dataset.focal_length[0]                      = static_cast<float>(_w) / (2.0f * tan(static_cast<float>(entry["camera_angle_x"]) * 0.5f));
                dataset.focal_length[1]                      = 0.f;
            }
        }
        result.success = true;
        result.message = "Found" + std::to_string(result.dataset.size()) + " images in dataset.";
        return result;
    }
}

ngp::LoadDatasetResult ngp::load_dataset(const LoadDatasetParams& params) {
    switch (params.dataset_type) {
    case LoadDatasetParams::DatasetType::NeRfSynthetic: return hidden::load_dataset_nerf_synthetic(params.dataset_path);
    case LoadDatasetParams::DatasetType::LLFF:
    case LoadDatasetParams::DatasetType::NSVF:
    case LoadDatasetParams::DatasetType::Unknown: throw std::runtime_error("[FATAL ERROR] - 1");
    default: throw std::runtime_error("[FATAL ERROR] - 2");
    }
}
