#include "ngp.train.h"
#include "ngp.dataset.h"
#include "ngp.cuda.session.cuh"

#include <pcg32/pcg32.h>
#include <tiny-cuda-nn/gpu_memory.h>


namespace ngp::cuda::hidden {

#define CUDA_CHECK_THROW(expr)                                                                 \
do {                                                                                           \
cudaError_t _result = (expr);                                                              \
if (_result != cudaSuccess) {                                                              \
throw std::runtime_error(fmt::format(                                                  \
"[CUDA ERROR] {}:{}: '{}' failed with: {}",                                        \
__FILE__, __LINE__, #expr, cudaGetErrorString(_result)));                          \
}                                                                                          \
} while (0)

    constexpr __device__ uint32_t N_MAX_RANDOM_SAMPLES_PER_RAY() {
        return 16;
    }

    inline __device__ uint32_t image_idx(
        const uint32_t base_idx,
        const uint32_t n_rays,
        const uint32_t n_training_images
        ) {
        return base_idx * n_training_images / n_rays % n_training_images;
    }

    inline __device__ tcnn::vec2 nerf_random_image_pos_training(
        tcnn::pcg32& rng,
        const tcnn::ivec2& resolution,
        const bool snap_to_pixel_centers
        ) {
        tcnn::vec2 uv = {rng.next_float(), rng.next_float()};

        if (snap_to_pixel_centers) {
            uv = (tcnn::vec2(tcnn::clamp(tcnn::ivec2(uv * tcnn::vec2(resolution)), 0, resolution - 1)) + 0.5f) / tcnn::vec2(resolution);
        }

        return uv;
    }

    enum class EImageDataType {
        None,
        Byte,
        Half,
        Float,
    };

    inline __device__ tcnn::ivec2 image_pos(const tcnn::vec2& pos, const tcnn::ivec2& resolution) {
        return tcnn::clamp(tcnn::ivec2(pos * tcnn::vec2(resolution)), 0, resolution - 1);
    }

    inline __device__ uint64_t pixel_idx(const tcnn::vec2& uv, const tcnn::ivec2& resolution, const uint32_t img) {
        const tcnn::ivec2 px = tcnn::clamp(tcnn::ivec2(uv * tcnn::vec2(resolution)), 0, resolution - 1);
        return px.x + px.y * resolution.x + img * static_cast<uint64_t>(resolution.x * resolution.y);
    }

    inline __device__ float srgb_to_linear(float x) {
        return (x <= 0.04045f) ? (x * (1.f / 12.92f)) : powf((x + 0.055f) * (1.f / 1.055f), 2.4f);
    }

    inline __device__ tcnn::vec4 read_rgba(
        const tcnn::vec2& uv,
        const tcnn::ivec2& resolution,
        const void* pixels,
        const uint32_t img = 0
        ) {
        const uint64_t idx  = pixel_idx(uv, resolution, img);
        const uint32_t rgba = static_cast<const uint32_t*>(pixels)[idx];

        if (rgba == 0x00FF00FFu) return {-1.f, -1.f, -1.f, -1.f};

        const float r = static_cast<float>((rgba >> 0) & 0xFF) * (1.f / 255.f);
        const float g = static_cast<float>((rgba >> 8) & 0xFF) * (1.f / 255.f);
        const float b = static_cast<float>((rgba >> 16) & 0xFF) * (1.f / 255.f);
        const float a = static_cast<float>((rgba >> 24) & 0xFF) * (1.f / 255.f);

        return {srgb_to_linear(r) * a,
                srgb_to_linear(g) * a,
                srgb_to_linear(b) * a,
                a};
    }

    inline __device__ tcnn::Ray uv_to_ray(
        const tcnn::vec2& uv,
        const tcnn::ivec2& resolution,
        const tcnn::vec2& focal_length,
        const tcnn::vec2& principal_point,
        const tcnn::mat4x3& xform
        ) {
        const tcnn::vec3 dir_cam = {
            (uv.x - principal_point.x) * static_cast<float>(resolution.x) / focal_length.x,
            (uv.y - principal_point.y) * static_cast<float>(resolution.y) / focal_length.y,
            1.0f
        };

        const tcnn::vec3 origin_world = xform[3];
        const tcnn::vec3 dir_world    = tcnn::mat3(xform) * dir_cam;

        return {origin_world, tcnn::normalize(dir_world)};
    }


    __global__ void generate_training_samples_nerf(
        const uint32_t n_rays,
        const uint32_t n_training_images,
        const NGPSession::ImagesDataGPU* __restrict__ dataset,
        tcnn::pcg32 rng
        ) {
        const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_rays) return;
        rng.advance(i * N_MAX_RANDOM_SAMPLES_PER_RAY());
        const uint32_t img  = image_idx(i, n_rays, n_training_images);
        const tcnn::vec2 uv = nerf_random_image_pos_training(rng, dataset[img].resolution, true);

        if (read_rgba(uv, dataset[img].resolution, dataset[img].pixels, 0).x < 0.0f) {
            printf("uv=(%f,%f)\n", uv.x, uv.y);
            return;
        }

        const tcnn::Ray ray_normalized = uv_to_ray(
            uv,
            dataset[img].resolution,
            dataset[img].focal,
            tcnn::vec2(0.5f, 0.5f),
            dataset[img].xform
            );

        assert(ray_normalized.is_valid());
    }
}

ngp::TrainResult ngp::train_session(const TrainParams& params) {
    const auto& images_cpu = params.dataset_cpu->images;
    const size_t n_images  = images_cpu.size();

    auto& session = cuda::hidden::NGPSession::instance();
    std::vector<cuda::hidden::NGPSession::ImagesDataGPU> _tmp_image_cpu;
    _tmp_image_cpu.resize(n_images);
    session.pixels_gpu.resize(n_images);
    const auto rng = session.m_rng;

    for (size_t i = 0; i < n_images; ++i) {
        const auto& [xform_cpu, pixels_cpu, resolution_cpu, focal_cpu, ch_cpu] = images_cpu[i];
        auto& [_pixels, _resolution, _focal, _xform]                           = _tmp_image_cpu[i];

        session.pixels_gpu[i].resize(resolution_cpu[0] * resolution_cpu[1] * ch_cpu * sizeof(uint8_t));
        session.pixels_gpu[i].copy_from_host(pixels_cpu);

        _pixels     = session.pixels_gpu[i].data();
        _resolution = tcnn::ivec2(static_cast<int>(resolution_cpu[0]), static_cast<int>(resolution_cpu[1]));
        _focal      = tcnn::vec2(focal_cpu[0], focal_cpu[1]);
        _xform      = tcnn::mat4x3(
            xform_cpu[0][0], xform_cpu[0][1], xform_cpu[0][2],
            xform_cpu[1][0], xform_cpu[1][1], xform_cpu[1][2],
            xform_cpu[2][0], xform_cpu[2][1], xform_cpu[2][2],
            xform_cpu[3][0], xform_cpu[3][1], xform_cpu[3][2]
            );
    }

    session.dataset_gpu.resize(n_images);
    session.dataset_gpu.copy_from_host(_tmp_image_cpu.data());

    tcnn::linear_kernel(
        cuda::hidden::generate_training_samples_nerf,
        0,
        session.m_stream.get(),
        session.rays_per_batch,
        n_images,
        session.dataset_gpu.data(),
        rng
        );

    CUDA_CHECK_THROW(cudaDeviceSynchronize());

    return {
        true,
        "Training not yet implemented in CUDA backend."
    };
}
