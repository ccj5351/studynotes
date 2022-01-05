# How to index a 4D tensor in your own cuda kernel function

## grid, block and thred in CUDA
- CUDA defines built-in 3D variables for threads and blocks. Threads are indexed using the built-in 3D variable `threadIdx`. Three-dimensional indexing provides a natural way to index elements in vectors, matrix, and volume and makes CUDA programming easier. 
- Similarly, blocks are also indexed using the in-built 3D variable called `blockIdx`.

## [One Low-level Example](https://github.com/princeton-vl/RAFT-Stereo/blob/main/sampler/sampler_kernel.cu)
- In this example, we can see how the `blockIdx.x (or .y, .z)` and `threadIdx.x (or .y , .z)` are directly used to access the 4D tensor (in size [N,C,H, W]) in PyTorch.
> const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.z;
  
- See the details in this kernel function defined as below:

```cpp
template <typename scalar_t>
__global__ void sampler_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> volume,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> coords,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> corr,
    int r){
  // --------------------------  
  // How to do batch index !!!
  // --------------------------
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.z;

  const int h1 = volume.size(1);
  const int w1 = volume.size(2);
  const int w2 = volume.size(3);

  if (!within_bounds(y, x, h1, w1)) {
    return;
  }

  float x0 = coords[n][0][y][x];
  //float y0 = coords[n][1][y][x];

  float dx = x0 - floor(x0);
  //float dy = y0 - floor(y0);

  int rd = 2*r + 1;
  for (int i=0; i<rd+1; i++) { // i is X
    int x1 = static_cast<int>(floor(x0)) - r + i;

    if (within_bounds(0, x1, 1, w2)) {
      scalar_t s = volume[n][y][x][x1];

      if (i > 0)
        corr[n][i-1][y][x] += s * scalar_t(dx);

      if (i < rd)
        corr[n][i][y][x] += s * scalar_t((1.0f-dx));

    }
  }
}

```

- The host function to call the kernel is defined below, and pay attention to syntex "<<<blocks, threads>>>" and how the `blocks` and `threads` dimensions are difined by `dim3`:

```cpp
std::vector<torch::Tensor> sampler_cuda_forward(
    torch::Tensor volume,
    torch::Tensor coords,
    int radius){
        
    const auto batch_size = volume.size(0);
    const auto ht = volume.size(1);
    const auto wd = volume.size(2);

    // --------------------------
    //CCJ: number of blocks per grid;
    // --------------------------
    const dim3 blocks((wd + BLOCK - 1) / BLOCK, 
                        (ht + BLOCK - 1) / BLOCK, 
                        batch_size);
    // --------------------------
    //CCJ: number of threads per block
    // --------------------------
    const dim3 threads(BLOCK, BLOCK);

    auto opts = volume.options();
    torch::Tensor corr = torch::zeros(
        {batch_size, 2*radius+1, ht, wd}, opts);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.type(), "sampler_forward_kernel", ([&] {
        sampler_forward_kernel<scalar_t><<<blocks, threads>>>(
        volume.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        coords.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        corr.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        radius);
    }));

    return {corr};
}
```

## [Another Higher-level Example](https://github.com/lizhihao6/Forward-Warp)

- In this example, we can see another way to define grids, blocks and threds using 1D indexing (i.e., only with `*.x` as in `gridDim.x`, `blockDim.x`, and `threadIdx.x`).

> const int B = im0.size(0);
  const int C = im0.size(1);
  const int H = im0.size(2);
  const int W = im0.size(3);
  const int total_step = B * H * W;
  
- See the details in the kernel defined below:

```cpp
// Define CUDA_NUM_THREAS and GET_BLOCKS
// Use 1024 threads per block, which requires cuda sm_2x or above;
const int CUDA_NUM_THREADS = 1024;
// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N){
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#endif

static __forceinline__ __device__ 
int get_im_index(
    const int b,
    const int c,
    const int h,
    const int w,
    const size_t C,
    const size_t H,
    const size_t W) {
  return b*C*H*W + c*H*W + h*W + w;
}

template <typename scalar_t>
__global__ void forward_warp_cuda_forward_kernel(
    const int total_step,
    const scalar_t* im0,
    const scalar_t* flow,
    scalar_t* im1,
    const int B,
    const int C,
    const int H,
    const int W,
    const GridSamplerInterpolation interpolation_mode) {
    CUDA_KERNEL_LOOP(index, total_step-1) {
    const int b = index / (H * W);
    const int h = (index-b*H*W) / W;
    const int w = index % W;
    const scalar_t x = (scalar_t)w + flow[index*2+0];
    const scalar_t y = (scalar_t)h + flow[index*2+1];
    if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
      const int x_f = static_cast<int>(::floor(x));
      const int y_f = static_cast<int>(::floor(y));
      const int x_c = x_f + 1;
      const int y_c = y_f + 1;
      if(x_f >= 0 && x_c < W && y_f >= 0 && y_c < H){
        const scalar_t nw_k = (x_c - x) * (y_c - y);
        const scalar_t ne_k = (x - x_f) * (y_c - y);
        const scalar_t sw_k = (x_c - x) * (y - y_f);
        const scalar_t se_k = (x - x_f) * (y - y_f);
        const scalar_t * im0_p = im0 + get_im_index(b, 0, h, w, C, H, W);
        scalar_t* im1_p = im1 + get_im_index(b, 0, y_f, x_f, C, H, W);
        for (int c = 0; c < C; ++c, im0_p += H*W, im1_p += H*W){// for Channel dimension;
          /* atomicAdd() reads a word at some address in global or shared memory, adds a number to it, 
           * and writes the result back to the same address. The operation is atomic in the sense 
           * that it is guaranteed to be performed without interference from other threads. 
           * In other words, no other thread can access this address until the operation is complete. 
           * Atomic functions do not act as memory fences and do not imply synchronization or ordering 
           * constraints for memory operations (see Memory Fence Functions for more details on memory fences). 
           * Atomic functions can only be used in device functions.
           */
            atomicAdd(im1_p,     nw_k*(*im0_p));
            atomicAdd(im1_p+1,   ne_k*(*im0_p));
            atomicAdd(im1_p+W,   sw_k*(*im0_p));
            atomicAdd(im1_p+W+1, se_k*(*im0_p));
        }
      }
    } 
    else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
      const int x_nearest = static_cast<int>(::round(x));
      const int y_nearest = static_cast<int>(::round(y));
      if(x_nearest >= 0 && x_nearest < W && y_nearest >= 0 && y_nearest < H){
        const scalar_t* im0_p = im0 + get_im_index(b, 0, h, w, C, H, W);
        scalar_t * im1_p = im1 + get_im_index(b, 0, y_nearest, x_nearest, C, H, W);
        for (int c = 0; c < C; ++c, im0_p += H*W, im1_p += H*W) {
            *im1_p = *im0_p;
        }
      }
    }
  }
}

```

- And the host function to call the kernel:

```cpp
at::Tensor forward_warp_cuda_forward(
    const at::Tensor im0, 
    const at::Tensor flow,
    const GridSamplerInterpolation interpolation_mode) {
  auto im1 = at::zeros_like(im0);
  const int B = im0.size(0);
  const int C = im0.size(1);
  const int H = im0.size(2);
  const int W = im0.size(3);
  const int total_step = B * H * W;
  /*CCJ: 
         [Warning - Deprecated Due to PyTorch&CUDA Version]
         passing at::DeprecatedTypeProperties to an AT_DISPATCH macro is deprecated, 
         pass an at::ScalarType instead [-Wdeprecated-declarations]
    Solution: The correct way to do this now is to use .scalar_type() instead of .type();
  */
  //AT_DISPATCH_FLOATING_TYPES(im0.type(), "forward_warp_forward_cuda", ([&] {
  AT_DISPATCH_FLOATING_TYPES(im0.scalar_type(), "forward_warp_forward_cuda", ([&] {
    forward_warp_cuda_forward_kernel<scalar_t>
    <<<GET_BLOCKS(total_step), CUDA_NUM_THREADS>>>(
      total_step,
      /*By CCJ: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead*/
      //im0.data<scalar_t>(),
      //flow.data<scalar_t>(),
      //im1.data<scalar_t>(), 
      im0.data_ptr<scalar_t>(),
      flow.data_ptr<scalar_t>(),
      im1.data_ptr<scalar_t>(), 
      B, C, H, W,
      interpolation_mode);
  }));

  return im1;
}
```
