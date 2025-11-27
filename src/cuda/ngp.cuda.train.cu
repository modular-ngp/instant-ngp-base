#include "ngp.cuda.train.h"

namespace ngp::cuda::hidden {
    __device__ uint32_t image_idx(
        const uint32_t base_idx,
        const uint32_t n_rays,
        const uint32_t n_training_images
        ) {
        return base_idx * n_training_images / n_rays % n_training_images;
    }

    __global__ void generate_training_samples_nerf(
        const uint32_t n_rays,
        const uint32_t n_training_images
        ) {
        const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_rays) return;

        const uint32_t img = image_idx(i, n_rays, n_training_images);
    }
}

ngp::cuda::TrainResult ngp::cuda::train(const TrainParams& params) {
    return {
        true,
    };
}
