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

__forceinline__ __device__ double& at(double *array, int i, int j)
{
  return array[i + j * d_nx];
}

__device__ void compute_kernel(int i,
                              int j,
                              double *h0,
                              double *hu0,
                              double *hv0,
                              double *h,
                              double *hu,
                              double *hv,
                              double *zdx,
                              double *zdy)
{
  double C1x = 0.5 * d_dt / d_dx;
  double C1y = 0.5 * d_dt / d_dy;
  double C2 = d_dt * d_g;
  double C3 = 0.5 * d_g;

  double hij = 0.25 * (at(h0, i, j - 1) + at(h0, i, j + 1) + at(h0, i - 1, j) + at(h0, i + 1, j))
               + C1x * (at(hu0, i - 1, j) - at(hu0, i + 1, j)) + C1y * (at(hv0, i, j - 1) - at(hv0, i, j + 1));
  // Avoid if condition by boolean multiplication
  hij = hij*(hij >= 0.0) + 1.0e-5*(hij < 0.0); 

  at(h, i, j) = hij;
  
  // Avoid branching and division by zero. We might get some idle
  // threads in a warp, but that should be better than double 
  // execution.
  hij = (hij > 0.0001) * hij + (hij <= 0.0001) * 1.0e-5; 
  at(hu, i, j) = (hij > 0.0001) * (
    0.25 * (at(hu0, i, j - 1) + at(hu0, i, j + 1) + at(hu0, i - 1, j) + at(hu0, i + 1, j)) - C2 * hij * at(zdx, i, j)
    + C1x
        * (at(hu0, i - 1, j) * at(hu0, i - 1, j) / at(h0, i - 1, j) + C3 * at(h0, i - 1, j) * at(h0, i - 1, j)
            - at(hu0, i + 1, j) * at(hu0, i + 1, j) / at(h0, i + 1, j) - C3 * at(h0, i + 1, j) * at(h0, i + 1, j))
    + C1y
        * (at(hu0, i, j - 1) * at(hv0, i, j - 1) / at(h0, i, j - 1)
            - at(hu0, i, j + 1) * at(hv0, i, j + 1) / at(h0, i, j + 1))
    );

  at(hv, i, j) = (hij > 0.0001) * (
    0.25 * (at(hv0, i, j - 1) + at(hv0, i, j + 1) + at(hv0, i - 1, j) + at(hv0, i + 1, j)) - C2 * hij * at(zdy, i, j)
    + C1x
        * (at(hu0, i - 1, j) * at(hv0, i - 1, j) / at(h0, i - 1, j)
            - at(hu0, i + 1, j) * at(hv0, i + 1, j) / at(h0, i + 1, j))
    + C1y
        * (at(hv0, i, j - 1) * at(hv0, i, j - 1) / at(h0, i, j - 1) + C3 * at(h0, i, j - 1) * at(h0, i, j - 1)
            - at(hv0, i, j + 1) * at(hv0, i, j + 1) / at(h0, i, j + 1) - C3 * at(h0, i, j + 1) * at(h0, i, j + 1))
    );
}


__global__ void compute_step(double *h0,
                             double *hu0,
                             double *hv0,
                             double *h,
                             double *hu,
                             double *hv,
                             double *zdx,
                             double *zdy)
{
  extern __shared__ double shared_memory[];

  // Shared memory arrays with padding for ghost cells
  // double *s_h0, *s_hu0, *s_hv0, *s_h, *s_hu, *s_hv, *s_zdx, *s_zdy;
  // s_h0  = &shared_memory[0];
  // s_hu0 = &s_h0[(blockDim.x + 2) * (blockDim.y + 2)];
  // s_hv0 = &s_hu0[(blockDim.x + 2) * (blockDim.y + 2)];
  // s_h   = &s_hv0[(blockDim.x + 2) * (blockDim.y + 2)];
  // s_hu  = &s_h[(blockDim.x + 2) * (blockDim.y + 2)];
  // s_hv  = &s_hu[(blockDim.x + 2) * (blockDim.y + 2)];
  // s_zdx = &s_hv[(blockDim.x + 2) * (blockDim.y + 2)];
  // s_zdy = &s_zdx[(blockDim.x + 2) * (blockDim.y + 2)];

  // // Thread Indices in global matrix
  // int gi = blockIdx.x * blockDim.x + threadIdx.x;
  // int gj = blockIdx.y * blockDim.y + threadIdx.y;

  // // Thread Indices in shared memory, with padding
  // int i = threadIdx.x + 1;
  // int j = threadIdx.y + 1; 

  // // Copy data local into shared memory
  // if (gi > 0 && gi < d_nx - 1 && gj > 0 && gj < d_ny - 1)
  // {
  //   at(s_h0, i, j, blockDim.x + 2) = at(h0, gi, gj, d_nx);
  //   at(s_hu0, i, j, blockDim.x + 2) = at(hu0, gi, gj, d_nx);
  //   at(s_hv0, i, j, blockDim.x + 2) = at(hv0, gi, gj, d_nx);
  //   at(s_zdx, i, j, blockDim.x + 2) = at(zdx, gi, gj, d_nx);
  //   at(s_zdy, i, j, blockDim.x + 2) = at(zdy, gi, gj, d_nx);
  // }

  // // Copy ghost cells

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // Don't do boundaries
  if (i > 0 && i < d_nx-1 && j > 0 && j < d_ny-1)
    compute_kernel(i, j, h0, hu0, hv0, h, hu, hv, zdx, zdy);
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
  dim3 block_size(16, 16); // Define block size
  dim3 grid_size((nx_ + block_size.x - 1) / block_size.x, (ny_ + block_size.y - 1) / block_size.y);
  // Pad the shared memory of a tile, 5 padded shared arrays
  int shared_memory_size = 5 * (block_size.x + 2) * (block_size.y + 2) * sizeof(double);
  
  
  // Initialize CUDA arrays and constants
  initialize_cuda_constants();
  initialize_cuda_arrays();
  copy_to_device(h0, hu0, hv0, h, hu, hv);
  err = cudaMemcpy(d_zdx, zdx_.data(), zdx_.size() * sizeof(double), cudaMemcpyHostToDevice);
  log_cuda_error(err, "Failed to copy zdx_ to device");
  err = cudaMemcpy(d_zdy, zdy_.data(), zdy_.size() * sizeof(double), cudaMemcpyHostToDevice);
  log_cuda_error(err, "Failed to copy zdy_ to device");

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

    this->update_bcs(h0, hu0, hv0, h, hu, hv);
    copy_to_device(h0, hu0, hv0, h, hu, hv);

    compute_step<<<grid_size, block_size>>>(d_h0, d_hu0, d_hv0, d_h, d_hu, d_hv, d_zdx, d_zdy);
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