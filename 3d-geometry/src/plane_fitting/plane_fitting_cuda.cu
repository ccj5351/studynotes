#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

/*
dx * x + dy * y = z(a, b) - z(0, 0);
*/
__global__ void plane_fitting_cuda_foward_kernel(
    float sigma, // e.g., == 0.1;
    float min_disp,
    float max_disp,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 4, torch::RestrictPtrTraits> random,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> output)
    {
    
    const int N = input.size(0);
    const int kernel_H = input.size(1); // e.g., kernel_H = 9
    const int kernel_W = input.size(2); // e.g., kernel_W = 9
    const int L = input.size(3);
    const int I = random.size(2); // iter, e.g., I=256

    const int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= N * L)
    {
        return;
    }
    const int n = Index / L;
    const int l = Index % L;

    int max_inlier = 0;
    float best_dx = 0.0f;
    float best_dy = 0.0f;
    
    // E.g., the center pixel is (9//2, 9//2)=(4,4);
    float z00 = input[n][kernel_H / 2][kernel_W / 2][l];

    if (z00 < min_disp || z00 > max_disp)
    {
        output[n][0][l] = best_dx;
        output[n][1][l] = best_dy;
        return;
    }

    for (int i = 0; i < I; ++i)
    {
        int ids0 = random[n][l][i][0];
        ids0 = (ids0 >= kernel_H * kernel_W / 2) ? ids0 + 1 : ids0;
        int x0 = ids0 % kernel_W;
        int y0 = ids0 / kernel_W;
        float z0 = input[n][y0][x0][l];
        if (z0 < min_disp || z0 > max_disp)
        {
            continue;
        }
        int ids1 = random[n][l][i][1];
        ids1 = (ids1 >= kernel_H * kernel_W / 2) ? ids1 + 1 : ids1;
        int x1 = ids1 % kernel_W;
        int y1 = ids1 / kernel_W;
        float z1 = input[n][y1][x1][l];
        if (z1 < min_disp || z1 > max_disp)
        {
            continue;
        }

        // NOTE: CCJ's note:
        // Sample 3 points P1, P2, and Pc to form a plane, and calculate the normal of the plane;
        // where P1 = (x1,y1,z1), P2 = (x2,y2,z2) and Pc =(xc, yc,zc), here xc=kernel_W//2, yc=kernel_H//2, (i.e., 
        // the center of the kernel widnow).
        // Given the vector vec(Pc,P1)=P1-Pc=(x1-xc, y1-yc,z1-zc) and the vector vec(Pc,P2)=P2-Pc=(x2-xc, y2-yc,z2-zc),
        // we can get the normal vector via `cross-product` between vec(Pc,P1) and vec(Pc,P2),
        // i.e., n=vec(Pc,P1) x vec(Pc,P1)
        //               i         j          k
        // vec(P1-Pc)  a=x1-xc  b=y1-yc  c=z1-zc
        // vec(P2-Pc)  d=x2-xc  e=y2-yc  f=z2-zc
        // Therefore, the normal n = i*(bf-ce) - j*(af-cd) + k*(ae-bd)
        // i.e., 
        // vector n = (bf-ce, cd-af, ae-bd) ... (1)
        // Given the normal and one point on the plane, say Pc, 
        // we can get the equation of the plane: Ax + By + Cz + D = 0
        // where the normal     n = (A,B,C) ... (2)
        // the derivative     dz/dx = - A/C ... (3)
        // and the derivative dz/dy = - B/C ... (3)
        // Submitting Eq. 1 to Eq. 2 ,3, and 4, we get
        //   1) dz/dx = -(bf-ce)/(ae-bd) = (ce-bf)/(ae-bd), and  
        //   2) dz/dY = -(cd-af)/(ae-bd) = (af-cd)/(ae-bd);
        //   Done!
        x0 -= (kernel_W / 2);
        y0 -= (kernel_H / 2);

        x1 -= (kernel_W / 2);
        y1 -= (kernel_H / 2);

        float c0 = z0 - z00;
        float c1 = z1 - z00;

        float dx = (c0 * y1 - y0 * c1) / (x0 * y1 - y0 * x1);
        float dy = (x0 * c1 - c0 * x1) / (x0 * y1 - y0 * x1);

        int inlier = 0;
        for (int h = 0; h < kernel_H; ++h)
        {
            for (int w = 0; w < kernel_W; ++w)
            {
                float zwh = input[n][h][w][l];
                if (zwh < min_disp || zwh > max_disp){
                    continue;
                }

                float err = dx * (w - kernel_W / 2) + dy * (h - kernel_H / 2) - zwh + z00;
                if (err < 0){
                    err = -err;
                }

                if (err < sigma){
                    ++inlier;
                }
            }
        }

        if (inlier > max_inlier)
        {
            max_inlier = inlier;
            best_dx = dx;
            best_dy = dy;
        }
    }

    output[n][0][l] = best_dx;
    output[n][1][l] = best_dy;
    return;
}

/* CCJ's note: 
 * Input is a disparity map D in shape of [N,C,H,W]=[N,1,H,W]
 * To fit a plane to disparity map D in a k x k= 9x9 window centered
 * at the pixel in D.
 * So here the input is actually a unfolded version of the original
 * disparity map D, e.g., `input` = torch.unfold(D, kernel_size=k, padding=k//2, stride=1);
 * resulting the `input` here in shape of [N*C, k, k, L]=[N,k,k,H*W] (with C=1);
 */
torch::Tensor plane_fitting_cuda_foward(
    torch::Tensor input, // [N*C, k, k, L]=[N,k,k,H*W] 
    int iter, 
    float sigma, 
    float min_disp, 
    float max_disp
    ){
    const auto N = input.size(0); // batch dimension
    const auto kernel_H = input.size(1); // kernel height
    const auto kernel_W = input.size(2); // kernel width
    const auto L = input.size(3); // number of pixels

    torch::Tensor random = torch::randint(
        0, kernel_H * kernel_W - 1, 
        {N, L, iter, 2}, 
        torch::dtype(torch::kInt32).device(input.device())
        );
    torch::Tensor output = torch::ones(
        {N, 2, L}, 
        torch::dtype(torch::kFloat32).device(input.device())
        );

    const at::cuda::OptionalCUDAGuard guard(device_of(input));
    plane_fitting_cuda_foward_kernel<<<(N * L + 1023) / 1024, 1024>>>(
        sigma,
        min_disp,
        max_disp,
        input.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        random.packed_accessor32<int, 4, torch::RestrictPtrTraits>(),
        output.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
    return output;
}