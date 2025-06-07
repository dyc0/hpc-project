#include "swe.hh"
#include "xdmf_writer.hh"

#include <iostream>
#include <cstddef>
#include <vector>
#include <string>
#include <cassert>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <cstdio>
#include <cmath>
#include <memory>


__constant__ int d_nx;
__constant__ int d_ny;
__constant__ double d_dx;
__constant__ double d_dy;
__constant__ double d_dt;
__constant__ double d_g;
__constant__ double d_reflect;

__forceinline__ __device__ double& at(double *array, int i, int j, int stride)
{
  return array[i + j * stride];
}

__device__ void compute_kernel(int i, int j, 
                               int gi, int gj, 
                               double *h0, double *hu0, double *hv0, 
                               double *h, double *hu, double *hv, 
                               double *zdx, double *zdy)
{
  double C1x = 0.5 * d_dt / d_dx;
  double C1y = 0.5 * d_dt / d_dy;
  double C2  = d_dt * d_g;
  double C3  = 0.5 * d_g;
  int ts = blockDim.x + 2; // Tile stride, include ghost cells

  double hij = 0.25 * (at(h0, i, j - 1, ts) + at(h0, i, j + 1, ts) + at(h0, i - 1, j, ts) + at(h0, i + 1, j, ts))
               + C1x * (at(hu0, i - 1, j, ts) - at(hu0, i + 1, j, ts)) + C1y * (at(hv0, i, j - 1, ts) - at(hv0, i, j + 1, ts));
  // Avoid if condition by boolean multiplication
  hij = hij*(hij >= 0.0) + 1.0e-5*(hij < 0.0); 

  at(h, gi, gj, d_nx) = hij;
  
  // Avoid branching and division by zero. We might get some idle
  // threads in a warp, but that should be better than double 
  // execution.
  hij = (hij > 0.0001) * hij + (hij <= 0.0001) * 1.0e-5; 
  at(hu, gi, gj, d_nx) = (hij > 0.0001) * (
    0.25 * (at(hu0, i, j - 1, ts) + at(hu0, i, j + 1, ts) + at(hu0, i - 1, j, ts) + at(hu0, i + 1, j, ts)) - C2 * hij * at(zdx, i, j, ts)
    + C1x
        * (at(hu0, i - 1, j, ts) * at(hu0, i - 1, j, ts) / at(h0, i - 1, j, ts) + C3 * at(h0, i - 1, j, ts) * at(h0, i - 1, j, ts)
            - at(hu0, i + 1, j, ts) * at(hu0, i + 1, j, ts) / at(h0, i + 1, j, ts) - C3 * at(h0, i + 1, j, ts) * at(h0, i + 1, j, ts))
    + C1y
        * (at(hu0, i, j - 1, ts) * at(hv0, i, j - 1, ts) / at(h0, i, j - 1, ts)
            - at(hu0, i, j + 1, ts) * at(hv0, i, j + 1, ts) / at(h0, i, j + 1, ts))
    );

  at(hv, gi, gj, d_nx) = (hij > 0.0001) * (
    0.25 * (at(hv0, i, j - 1, ts) + at(hv0, i, j + 1, ts) + at(hv0, i - 1, j, ts) + at(hv0, i + 1, j, ts)) - C2 * hij * at(zdy, i, j, ts)
    + C1x
        * (at(hu0, i - 1, j, ts) * at(hv0, i - 1, j, ts) / at(h0, i - 1, j, ts)
            - at(hu0, i + 1, j, ts) * at(hv0, i + 1, j, ts) / at(h0, i + 1, j,ts))
    + C1y
        * (at(hv0, i, j - 1, ts) * at(hv0, i, j - 1, ts) / at(h0, i, j - 1, ts) + C3 * at(h0, i, j - 1, ts) * at(h0, i, j - 1, ts)
            - at(hv0, i, j + 1, ts) * at(hv0, i, j + 1, ts) / at(h0, i, j + 1, ts) - C3 * at(h0, i, j + 1, ts) * at(h0, i, j + 1, ts))
    );
}


__global__ void compute_step(double *h0, double *hu0, double *hv0, 
                             double *h, double *hu, double *hv, 
                             double *zdx, double *zdy)
{
  extern __shared__ double shared_memory[];

  // Shared memory arrays with padding for ghost cells
  double *s_h0, *s_hu0, *s_hv0, *s_zdx, *s_zdy;
  s_h0  = &shared_memory[0];
  s_hu0 = &s_h0[(blockDim.x + 2) * (blockDim.y + 2)];
  s_hv0 = &s_hu0[(blockDim.x + 2) * (blockDim.y + 2)];
  s_zdx = &s_hv0[(blockDim.x + 2) * (blockDim.y + 2)];
  s_zdy = &s_zdx[(blockDim.x + 2) * (blockDim.y + 2)];

  // Thread Indices in shared memory (tiles), with padding
  int si = threadIdx.x + 1;
  int sj = threadIdx.y + 1; 

  // Thread Indices in global matrix, avoiding boundaries
  int gi = blockIdx.x * blockDim.x + si;
  int gj = blockIdx.y * blockDim.y + sj;

  // Copy local data into shared memory
  if (gi < d_nx - 1 && gj < d_ny - 1)
  {
    at(s_h0,  si, sj, blockDim.x + 2) = at(h0,  gi, gj, d_nx);
    at(s_hu0, si, sj, blockDim.x + 2) = at(hu0, gi, gj, d_nx);
    at(s_hv0, si, sj, blockDim.x + 2) = at(hv0, gi, gj, d_nx);
    at(s_zdx, si, sj, blockDim.x + 2) = at(zdx, gi, gj, d_nx);
    at(s_zdy, si, sj, blockDim.x + 2) = at(zdy, gi, gj, d_nx);
  }

  // Copy ghost cells. I don't think I can avoid branching here.
  // Copying left and to ghosts is easy, because we align blocks with the grid,
  // skipping boundary cells.
  if (si == 1 && gi < d_nx - 1 && gj < d_ny - 1) {
    // Left ghost cells
    at(s_h0,  0, sj, blockDim.x + 2) = at(h0,  gi - 1, gj, d_nx);
    at(s_hu0, 0, sj, blockDim.x + 2) = at(hu0, gi - 1, gj, d_nx);
    at(s_hv0, 0, sj, blockDim.x + 2) = at(hv0, gi - 1, gj, d_nx);
    at(s_zdx, 0, sj, blockDim.x + 2) = at(zdx, gi - 1, gj, d_nx);
    at(s_zdy, 0, sj, blockDim.x + 2) = at(zdy, gi - 1, gj, d_nx);
  }
  if (sj == 1 && gi < d_nx - 1 && gj < d_ny - 1) {
    // Top ghost cells
    at(s_h0,  si, 0, blockDim.x + 2) = at(h0,  gi, gj - 1, d_nx);
    at(s_hu0, si, 0, blockDim.x + 2) = at(hu0, gi, gj - 1, d_nx);
    at(s_hv0, si, 0, blockDim.x + 2) = at(hv0, gi, gj - 1, d_nx);
    at(s_zdx, si, 0, blockDim.x + 2) = at(zdx, gi, gj - 1, d_nx);
    at(s_zdy, si, 0, blockDim.x + 2) = at(zdy, gi, gj - 1, d_nx);
  }

  // Copying bottom and right ghost cells is a bit more tricky, because a block
  // can spill over the matrix boundaries. We need to copy the data into appropriate
  // cells that will be used in the caluclations.

  // Here we calculate the left boundary. If the block is completely inside the matrix,
  // a thread should copy the ghost cells iff it is at the edge of the block.
  // Otherwise, the thread that corresponds to the last interior cell of the full matrix
  // should copy the ghost. The first mask corresponds to the interior case. Last term
  // in its boolean expression avoids overlapping in case the blocks are totally aligned
  // with the matrix. Second mask corresponds to the case that the block covers exterior
  // of the mesh. The values of ghosts are not neccessarily at the block boundary.
  //  |-  block is fully inside the matrix                 |- block is partially outside
  if ((si == blockDim.x && gi < d_nx-1 && gi != d_nx-2) || (gi == d_nx-2) && gj < d_ny - 1) {
    // Right ghost cells
    at(s_h0,  si + 1, sj, blockDim.x + 2) = at(h0,  gi + 1, gj, d_nx);
    at(s_hu0, si + 1, sj, blockDim.x + 2) = at(hu0, gi + 1, gj, d_nx);
    at(s_hv0, si + 1, sj, blockDim.x + 2) = at(hv0, gi + 1, gj, d_nx);
    at(s_zdx, si + 1, sj, blockDim.x + 2) = at(zdx, gi + 1, gj, d_nx);
    at(s_zdy, si + 1, sj, blockDim.x + 2) = at(zdy, gi + 1, gj, d_nx);
  }
  // The previous reasoning is repeated for the bottom ghost cells.
  if (((sj == blockDim.y && gj < d_ny-1 && gj != d_ny-2) || (gj == d_ny-2)) && gi < d_nx - 1) {
    // Bottom ghost cells
    at(s_h0,  si, sj + 1, blockDim.x + 2) = at(h0,  gi, gj + 1, d_nx);
    at(s_hu0, si, sj + 1, blockDim.x + 2) = at(hu0, gi, gj + 1, d_nx);
    at(s_hv0, si, sj + 1, blockDim.x + 2) = at(hv0, gi, gj + 1, d_nx);
    at(s_zdx, si, sj + 1, blockDim.x + 2) = at(zdx, gi, gj + 1, d_nx);
    at(s_zdy, si, sj + 1, blockDim.x + 2) = at(zdy, gi, gj + 1, d_nx);
  }

  __syncthreads();

  // Don't do boundaries. gi, gj > 0 by design, so we only check the upper bounds.
  if (gi < d_nx-1 && gj < d_ny-1)
    compute_kernel(si, sj, gi, gj, s_h0, s_hu0, s_hv0, h, hu, hv, s_zdx, s_zdy);
}


__global__ void update_bcs(double *h0, double *hu0, double *hv0, 
                             double *h, double *hu, double *hv)
{
  // Update boundary conditions for h, hu, hv
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // Used for indexing the boundary cells, either row or column,
  // depending on the context.
  int idx;

  // There are a lot of ifs here, but with good block size, I think we
  // can avoid branching in the kernel by forcing warps to mostly work on
  // the same part of the boundary.
  if (tid < d_nx)    // Top boundary
  {
    idx = tid;
    at(h,  idx, 0, d_nx) = at(h0, idx, 1, d_nx);
    at(hu, idx, 0, d_nx) = at(hu0, idx, 1, d_nx);
    at(hv, idx, 0, d_nx) = d_reflect * at(hv0, idx, 1, d_nx);
  }
  else if (tid > d_nx && tid < d_nx + d_ny)    // Right boundary
  {
    idx = tid - d_nx;
    at(h,  d_nx - 1, idx, d_nx) = at(h0, d_nx - 2, idx, d_nx);
    at(hu, d_nx - 1, idx, d_nx) = d_reflect * at(hu0, d_nx - 2, idx, d_nx);
    at(hv, d_nx - 1, idx, d_nx) = at(hv0, d_nx - 2, idx, d_nx);
  }
  else if (tid > d_nx + d_ny && tid < 2*d_nx + d_ny)    // Bottom boundary
  {
    idx = tid - (d_nx + d_ny);
    // It doesn't matter we don't go clockwise.
    at(h,  idx, d_ny - 1, d_nx) = at(h0,  idx, d_ny - 2, d_nx);
    at(hu, idx, d_ny - 1, d_nx) = at(hu0, idx, d_ny - 2, d_nx);
    at(hv, idx, d_ny - 1, d_nx) = d_reflect * at(hv0, idx, d_ny - 2, d_nx);
  }
  else if (tid > 2*d_nx + d_ny && tid < 2*(d_nx + d_ny))    // Left boundary
  {
    idx = tid - (2 * d_nx + d_ny);
    at(h,  0, idx, d_nx) = at(h0, 1, idx, d_nx);
    at(hu, 0, idx, d_nx) = d_reflect * at(hu0, 1, idx, d_nx);
    at(hv, 0, idx, d_nx) = at(hv0, 1, idx, d_nx);
  }
}


void log_cuda_error(cudaError_t err, std::string additional_info) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << " " << additional_info << std::endl;
    throw std::runtime_error("CUDA error occurred.");
  }
}


void
SWESolver::solve(const double Tend, const bool full_log, const std::size_t output_n, const std::string &fname_prefix)
{
  std::shared_ptr<XDMFWriter> writer;
  if (output_n > 0)
  {
    writer = std::make_shared<XDMFWriter>(fname_prefix, this->nx_, this->ny_, this->size_x_, this->size_y_, this->z_);
    writer->add_h(h0_, 0.0);
  }

  double T = 0.0;

  std::vector<double> &h = h1_;
  std::vector<double> &hu = hu1_;
  std::vector<double> &hv = hv1_;

  std::vector<double> &h0 = h0_;
  std::vector<double> &hu0 = hu0_;
  std::vector<double> &hv0 = hv0_;

  // CUDA variables
  cudaError_t err;

  // Dimensions used for the computation of interior values
  dim3 stencil_block_size(16, 16); // Define block size
  dim3 stencil_grid_size((nx_ + stencil_block_size.x - 1) / stencil_block_size.x, (ny_ + stencil_block_size.y - 1) / stencil_block_size.y);
  // Pad the shared memory of a tile, 5 padded shared arrays
  int shared_memory_size = 5 * (stencil_block_size.x + 2) * (stencil_block_size.y + 2) * sizeof(double);
  
  // Dimensions used for computation of boundary values
  dim3 bdry_block_size(128);
  dim3 bdry_grid_size(2 * (nx_ + ny_ + bdry_block_size.x - 1) / bdry_block_size.x);

  // Initialize CUDA arrays and constants
  initialize_cuda_constants();
  initialize_cuda_arrays();
  copy_to_device(h0, hu0, hv0, h, hu, hv);
  err = cudaMemcpy(d_zdx, zdx_.data(), zdx_.size() * sizeof(double), cudaMemcpyHostToDevice);
  log_cuda_error(err, "Failed to copy zdx_ to device");
  err = cudaMemcpy(d_zdy, zdy_.data(), zdy_.size() * sizeof(double), cudaMemcpyHostToDevice);
  log_cuda_error(err, "Failed to copy zdy_ to device");
  double *tmp;      // For buffer swapping

  std::cout << "Solving SWE..." << std::endl;
  

  std::size_t nt = 1;
  while (T < Tend)
  {
    const double dt = this->compute_time_step(h0, hu0, hv0, T, Tend);
    err = cudaMemcpyToSymbol(d_dt, &dt, sizeof(double), 0, cudaMemcpyHostToDevice);
    log_cuda_error(err, "Failed to copy dt to constant memory");

    const double T1 = T + dt;

    printf("Computing T: %2.4f hr  (dt = %.2e s) -- %3.3f%%", T1, dt * 3600, 100 * T1 / Tend);
    std::cout << (full_log ? "\n" : "\r") << std::flush;

    copy_to_device(h0, hu0, hv0, h, hu, hv);
    update_bcs<<<bdry_grid_size, bdry_block_size>>>(d_h0, d_hu0, d_hv0, d_h, d_hu, d_hv);

    compute_step<<<stencil_grid_size, stencil_block_size, shared_memory_size>>>(d_h0, d_hu0, d_hv0, d_h, d_hu, d_hv, d_zdx, d_zdy);
    err = cudaGetLastError();
    log_cuda_error(err, "Failed to launch compute_step kernel");

    cudaDeviceSynchronize();
    copy_from_device(h0, hu0, hv0, h, hu, hv);

    if (output_n > 0 && nt % output_n == 0)
    {
      writer->add_h(h, T1);
    }
    ++nt;

    // Swap the old and new solutions
    std::swap(h, h0);
    std::swap(hu, hu0);
    std::swap(hv, hv0);

    T = T1;
  }

  // Copying last computed values to h1_, hu1_, hv1_ (if needed)
  if (&h0 != &h1_)
  {
    h1_ = h0;
    hu1_ = hu0;
    hv1_ = hv0;
  }

  if (output_n > 0)
  {
    writer->add_h(h1_, T);
  }

  std::cout << "Finished solving SWE." << std::endl;
}



void SWESolver::initialize_cuda_constants() {
  cudaError_t err;
  double tmp;
  err = cudaMemcpyToSymbol(d_nx, &nx_, sizeof(int));
  log_cuda_error(err, "Failed to copy nx_ to constant memory");
  err = cudaMemcpyToSymbol(d_ny, &ny_, sizeof(int));
  log_cuda_error(err, "Failed to copy ny_ to constant memory");
  tmp = size_x_ / static_cast<double>(nx_);
  err = cudaMemcpyToSymbol(d_dx, &tmp, sizeof(double));
  log_cuda_error(err, "Failed to copy dx to constant memory");
  tmp = size_y_ / static_cast<double>(ny_);
  err = cudaMemcpyToSymbol(d_dy, &tmp, sizeof(double));
  log_cuda_error(err, "Failed to copy dy to constant memory");
  tmp = g;
  err = cudaMemcpyToSymbol(d_g, &tmp, sizeof(double));
  log_cuda_error(err, "Failed to copy g to constant memory");
  tmp = reflective_ ? -1.0 :1.0;
  err = cudaMemcpyToSymbol(d_reflect, &tmp, sizeof(double));
  log_cuda_error(err, "Failed to copy reflect to constant memory");
};


void SWESolver::initialize_cuda_arrays() 
{
  cudaError_t err;

  err = cudaMalloc((void**)&d_h0, h0_.size() * sizeof(double));
  log_cuda_error(err, "Failed to allocate d_h0 on device");
  err = cudaMalloc((void**)&d_hu0, hu0_.size() * sizeof(double));
  log_cuda_error(err, "Failed to allocate d_hu0 on device");
  err = cudaMalloc((void**)&d_hv0, hv0_.size() * sizeof(double));
  log_cuda_error(err, "Failed to allocate d_hv0 on device");
  err = cudaMalloc((void**)&d_h, h1_.size() * sizeof(double));
  log_cuda_error(err, "Failed to allocate d_h on device");
  err = cudaMalloc((void**)&d_hu, hu1_.size() * sizeof(double));
  log_cuda_error(err, "Failed to allocate d_hu on device");
  err = cudaMalloc((void**)&d_hv, hv1_.size() * sizeof(double));
  log_cuda_error(err, "Failed to allocate d_hv on device");
  err = cudaMalloc((void**)&d_zdx, zdx_.size() * sizeof(double));
  log_cuda_error(err, "Failed to allocate d_zdx on device");
  err = cudaMalloc((void**)&d_zdy, zdy_.size() * sizeof(double));
  log_cuda_error(err, "Failed to allocate d_zdy on device");
}


void SWESolver::copy_to_device(std::vector<double> &h0,
                               std::vector<double> &hu0,
                               std::vector<double> &hv0,
                               std::vector<double> &h1,
                               std::vector<double> &hu1,
                               std::vector<double> &hv1)
{
  cudaError_t err;

  err = cudaMemcpy(d_h0, h0.data(), h0.size()*sizeof(double), cudaMemcpyHostToDevice);
  log_cuda_error(err, "Failed to copy h0_ to device");
  err = cudaMemcpy(d_hu0, hu0.data(), hu0.size()*sizeof(double), cudaMemcpyHostToDevice);
  log_cuda_error(err, "Failed to copy hu0_ to device");
  err = cudaMemcpy(d_hv0, hv0.data(), hv0.size()*sizeof(double), cudaMemcpyHostToDevice);
  log_cuda_error(err, "Failed to copy hv0_ to device");
  err = cudaMemcpy(d_h, h1.data(), h1.size()*sizeof(double), cudaMemcpyHostToDevice);
  log_cuda_error(err, "Failed to copy h1_ to device");
  err = cudaMemcpy(d_hu, hu1.data(), hu1.size()*sizeof(double), cudaMemcpyHostToDevice);
  log_cuda_error(err, "Failed to copy hu1_ to device");
  err = cudaMemcpy(d_hv, hv1.data(), hv1.size()*sizeof(double), cudaMemcpyHostToDevice);
  log_cuda_error(err, "Failed to copy hv1_ to device");
}

void SWESolver::copy_from_device(std::vector<double>  &h0,
                                  std::vector<double> &hu0,
                                  std::vector<double> &hv0,
                                  std::vector<double> &h1,
                                  std::vector<double> &hu1,
                                  std::vector<double> &hv1)
{
  cudaError_t err;

  err = cudaMemcpy(h0.data(), d_h0, h0.size()*sizeof(double), cudaMemcpyDeviceToHost);
  log_cuda_error(err, "Failed to copy d_h0 to h0_");
  err = cudaMemcpy(hu0.data(), d_hu0, hu0.size()*sizeof(double), cudaMemcpyDeviceToHost); 
  log_cuda_error(err, "Failed to copy d_hu0 to hu0_");
  err = cudaMemcpy(hv0.data(), d_hv0, hv0.size()*sizeof(double), cudaMemcpyDeviceToHost);
  log_cuda_error(err, "Failed to copy d_hv0 to hv0_");
  err = cudaMemcpy(h1.data(), d_h, h1.size()*sizeof(double), cudaMemcpyDeviceToHost);
  log_cuda_error(err, "Failed to copy d_h to h1_");
  err = cudaMemcpy(hu1.data(), d_hu, hu1.size()*sizeof(double), cudaMemcpyDeviceToHost);
  log_cuda_error(err, "Failed to copy d_hu to hu1_");
  err = cudaMemcpy(hv1.data(), d_hv, hv1.size()*sizeof(double), cudaMemcpyDeviceToHost);
  log_cuda_error(err, "Failed to copy d_hv to hv1_");
}