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
        const uint32_t img = 0 // optional, default works same as before
        ) {
        // ---------------------------------------------
        // 1. Get pixel address from uv + resolution
        // ---------------------------------------------
        const uint64_t idx  = pixel_idx(uv, resolution, img);
        const uint32_t rgba = static_cast<const uint32_t*>(pixels)[idx]; // packed 0xAARRGGBB

        // ---------------------------------------------
        // 2. Masked pixel → skip (-1 = INVALID)
        // ---------------------------------------------
        if (rgba == 0x00FF00FFu) return {-1.f, -1.f, -1.f, -1.f};

        // ---------------------------------------------
        // 3. Extract channels [0–255] → float [0–1]
        // ---------------------------------------------
        const float r = static_cast<float>((rgba >> 0) & 0xFF) * (1.f / 255.f);
        const float g = static_cast<float>((rgba >> 8) & 0xFF) * (1.f / 255.f);
        const float b = static_cast<float>((rgba >> 16) & 0xFF) * (1.f / 255.f);
        const float a = static_cast<float>((rgba >> 24) & 0xFF) * (1.f / 255.f);

        return {srgb_to_linear(r) * a,
                srgb_to_linear(g) * a,
                srgb_to_linear(b) * a,
                a};
    }


    __global__ void generate_training_samples_nerf(
        const uint32_t n_rays,
        const uint32_t n_training_images,
        const NGPSession::ImagesDataGPU* __restrict__ dataset,
        tcnn::pcg32 rng
        ) {
        const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_rays) return;
        const uint32_t img           = image_idx(i, n_rays, n_training_images);
        const tcnn::ivec2 resolution = dataset[img].resolution;
        rng.advance(i * N_MAX_RANDOM_SAMPLES_PER_RAY());

        tcnn::vec2 uv = nerf_random_image_pos_training(rng, resolution, true);

        uint64_t pix_idx = pixel_idx(uv, resolution, 0);

        printf("thread=%u img=%u pix=%llu\n", i, img, (unsigned long long) pix_idx);
        // if (read_rgba(uv, resolution, dataset->metadata[img].pixels, 0).x < 0.0f) {
        //     return;
        // }
        //
        // constexpr float max_level = 1.0f;
        //
        // tcnn::mat4x3 xform = dataset->xforms[img];
    }
}

ngp::TrainResult ngp::train_session(const TrainParams& params) {
    const auto& images_cpu = params.dataset_cpu->images;
    const size_t n_images  = images_cpu.size();

    auto& session = cuda::hidden::NGPSession::instance();
    session.dataset_cpu.resize(n_images);
    session.pixel_buffers_gpu.resize(n_images);
    auto rng = session.m_rng; // copy rng

    for (size_t i = 0; i < n_images; ++i) {
        const auto& [xform, pixels, resolution, focal_length, channel] = images_cpu[i];

        // 1. Upload pixels to GPU
        session.pixel_buffers_gpu[i].resize(resolution[0] * resolution[1] * channel * sizeof(uint8_t));
        session.pixel_buffers_gpu[i].copy_from_host(pixels);

        // 2. Fill CPU-side metadata struct
        auto& dst        = session.dataset_cpu[i];
        dst.pixels       = session.pixel_buffers_gpu[i].data(); // device pointer
        dst.resolution   = tcnn::ivec2(resolution[0], resolution[1]);
        dst.focal_length = tcnn::vec2(focal_length[0], focal_length[1]);
        dst.xform        = tcnn::mat4x3(
            xform[0][0], xform[0][1], xform[0][2],
            xform[1][0], xform[1][1], xform[1][2],
            xform[2][0], xform[2][1], xform[2][2],
            xform[3][0], xform[3][1], xform[3][2]
            );
    }

    // 3. Copy the whole array-of-structs to device
    session.dataset_gpu.resize(n_images);
    session.dataset_gpu.copy_from_host(session.dataset_cpu.data());

    tcnn::linear_kernel(
        cuda::hidden::generate_training_samples_nerf,
        0,
        cuda::hidden::NGPSession::instance().m_stream.get(),
        4096,
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
