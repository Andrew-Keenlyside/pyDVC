// Host emulation of the CUDA features fused_gn.cu uses, for testing without a GPU.
//
// Every thread of a block is a std::thread; blocks run one after another.
// __syncthreads is a std::barrier across the block, __shfl_down_sync a
// barrier-fenced exchange within each 32-thread warp, and __shared__ is
// `static`, which is block-shared because blocks never overlap. Slow, but it
// runs the real kernel source, reductions included.
#pragma once
#include <barrier>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

struct dim3 {
    unsigned x = 1, y = 1, z = 1;
};

namespace emu {
inline thread_local dim3 thread_idx;
inline dim3 block_idx, block_dim, grid_dim;
inline std::barrier<>* block_barrier = nullptr;
inline std::vector<std::unique_ptr<std::barrier<>>> warp_barriers;
inline double warp_slots[64][32];

inline void launch(unsigned grid, unsigned block, const std::function<void()>& body) {
    if (block == 0 || block % 32 != 0 || block > 2048) throw std::runtime_error("blockDim must be a multiple of 32, <= 2048");
    grid_dim = {grid, 1, 1};
    block_dim = {block, 1, 1};
    std::barrier<> bar((std::ptrdiff_t)block);
    block_barrier = &bar;
    warp_barriers.clear();
    for (unsigned w = 0; w < block / 32; ++w) warp_barriers.emplace_back(new std::barrier<>(32));
    for (unsigned b = 0; b < grid; ++b) {
        block_idx = {b, 1, 1};
        std::vector<std::thread> threads;
        threads.reserve(block);
        for (unsigned t = 0; t < block; ++t)
            threads.emplace_back([t, &body] {
                thread_idx = {t, 1, 1};
                body();
            });
        for (auto& th : threads) th.join();
    }
    block_barrier = nullptr;
}
}  // namespace emu

#define threadIdx (emu::thread_idx)
#define blockIdx (emu::block_idx)
#define blockDim (emu::block_dim)
#define gridDim (emu::grid_dim)
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static

inline void __syncthreads() { emu::block_barrier->arrive_and_wait(); }

template <class T>
inline T __ldg(const T* p) { return *p; }

template <class T>
inline T __shfl_down_sync(unsigned, T v, unsigned delta, int = 32) {
    const unsigned t = threadIdx.x, w = t / 32, l = t % 32;
    emu::warp_slots[w][l] = (double)v;
    emu::warp_barriers[w]->arrive_and_wait();
    const T r = (l + delta < 32) ? (T)emu::warp_slots[w][l + delta] : v;
    emu::warp_barriers[w]->arrive_and_wait();
    return r;
}
