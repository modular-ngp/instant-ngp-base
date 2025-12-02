#include "ngp.dataset.h"

#include <nlohmann/json.hpp>
#include <stb_image.h>
#include <fstream>
#include <ranges>
#include <execution>
#include <atomic>
#include <vector>

namespace ngp::hidden {
    std::shared_ptr<LoadDatasetResult> load_dataset_nerf_synthetic(const std::filesystem::path& dataset_path) {
        const auto json_paths = std::ranges::to<std::vector<std::filesystem::path>>(
            std::filesystem::directory_iterator(dataset_path)
            | std::views::filter([](auto const& e) {
                return e.path().extension() == ".json";
            })
            | std::views::transform([](auto const& e) {
                return e.path();
            })
            );

        std::vector<nlohmann::json> json_files;
        json_files.reserve(json_paths.size());
        for (auto const& p : json_paths) if (std::ifstream f(p); f) json_files.emplace_back(nlohmann::json::parse(f));

        auto frames = std::ranges::to<std::vector<nlohmann::json>>(
            json_files | std::views::transform([](auto const& j) {
                return j["frames"];
            })
            | std::views::join
            );

        auto result = std::make_shared<LoadDatasetResult>();
        result->images.resize(frames.size());

        const float angle = json_files.front()["camera_angle_x"];
        std::atomic_uint i{0};

        std::for_each(std::execution::par, frames.begin(), frames.end(),
            [&](auto const& f) {
                auto& [xform, pixels, resolution, focal, channel] = result->images[i.fetch_add(1)];
                for (int m = 0; m < 3; ++m) for (int n = 0; n < 4; ++n) xform[n][m] = static_cast<float>(f["transform_matrix"][m][n]);
                auto path = (dataset_path / f["file_path"].template get<std::string>()).concat(".png");
                int w{}, h{}, c{};
                pixels          = stbi_load(path.string().c_str(), &w, &h, &c, 4);
                resolution      = {static_cast<size_t>(w), static_cast<size_t>(h)};
                focal[0] = static_cast<float>(w) / (2.f * std::tan(angle * .5f));
                focal[1] = focal[0];
                channel         = static_cast<size_t>(c);
            }
            );

        result->success = true;
        result->message = std::format(
            "Modern loader OK\tJSON: {}\tFrames: {}\tThreads: {}\t",
            json_files.size(),
            result->images.size(),
            std::thread::hardware_concurrency()
            );
        return result;
    }
}

std::shared_ptr<ngp::LoadDatasetResult> ngp::load_dataset(const LoadDatasetParams& params) {
    switch (params.dataset_type) {
    case LoadDatasetParams::DatasetType::NeRfSynthetic: return hidden::load_dataset_nerf_synthetic(params.dataset_path);
    case LoadDatasetParams::DatasetType::LLFF:
    case LoadDatasetParams::DatasetType::NSVF:
    case LoadDatasetParams::DatasetType::Unknown: throw std::runtime_error("[FATAL ERROR] - 1");
    default: throw std::runtime_error("[FATAL ERROR] - 2");
    }
}
