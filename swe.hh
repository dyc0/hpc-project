#include <cstddef>
#include <vector>
#include <string>

class SWESolver
{
public:
  /**
   * @brief Constructor for the SWESolver class.
   * @warning Not allowed to be used.
   */
  SWESolver() = delete;

  /// Gravity 9.82 * (3.6)^2 * 1000 in[km / hour^2]
  static constexpr double g = 127267.20000000;

  /**
   * @brief Construtor.
   * @param test_case_id It can be 1 (water drops in a box) or 2 (analytical tsunami).
   * @param nx  Number of cells along the x direction.
   * @param ny  Number of cells along the y direction.
   */
  SWESolver(const int test_case_id, const std::size_t nx, const std::size_t ny);

  /**
   * @brief Constructor for the SWESolver class.
   *
   * This constructor corresponds to the case in which the initial conditions
   * and topography are read from a HDF5 file.
   *
   * @param h5_file HDF5 file name containing the initial conditions and topography.
   * @param size_x  Size in km along the x direction.
   * @param size_y  Size in km along the y direction.
   */
  SWESolver(const std::string &h5_file, const double size_x, const double size_y);

  /**
   * @brief Solve the shallow water equations.
   * @brief Tend Total simulation time.
   * @brief full_log If true, the simulation will log the time step
   * and the time step size at each time step. Otherwise, only
   * the progress of the simulation will be logged.
   * @brief output_n If different from 0, the simulation will write
   * a solution each output_n time steps. E.g., if set to 10,
   * a solution file will be written each 10 steps.
   * @brief fname_prefix If @p output_n is different from 0, the generated
   * files will use this file name prefix.
   */
  void solve(const double Tend,
             const bool full_log = false,
             const std::size_t output_n = 0,
             const std::string &fname_prefix = "test",
             int block_dim_x = 16,
             int block_dim_y = 16);

private:
  /**
   * @brief Initializes the initial conditions and topography using
   * the provided HDF5 file.
   *
   * @param h5_file HDF5 file name containing the initial conditions and topography.
   */
  void init_from_HDF5_file(const std::string &h5_file);

  /**
   * @brief Initializes the initial conditions and topography using
   * a Gaussian function.
   *
   * The water height is initialized with two separated Gaussian peaks.
   * The initial water velocity is set to zero and the topography is set to zero.
   */
  void init_gaussian();

  /**
   * @brief Initializes the initial conditions and topography using
   * a dummy tsunami function.
   */
  void init_dummy_tsunami();

  /**
   * @brief Initializes the initial conditions and topography using
   * a slope function.
   */
  void init_dummy_slope();

  /**
   * @brief Initializes the derivatives dx and dy from the topography.
   */
  void init_dx_dy();

  std::size_t nx_;
  std::size_t ny_;
  double size_x_;
  double size_y_;
  bool reflective_;
  std::vector<double> h_;
  std::vector<double> hu_;
  std::vector<double> hv_;
  std::vector<double> z_;
  std::vector<double> zdx_;
  std::vector<double> zdy_;

  double *d_h0, *d_h;
  double *d_hu0, *d_hu;
  double *d_hv0, *d_hv;
  double *d_zdx, *d_zdy;
  double *d_local_dt;

  /**
   * @brief Accessor for 2D vector elements.
   */
  inline double &at(std::vector<double> &vec, const std::size_t i, const std::size_t j) const
  {
    return vec[j * nx_ + i];
  }

  /**
   * @brief Accessor for 2D vector elements.
   * @note Constant vector version.
   */
  inline const double &at(const std::vector<double> &vec, const std::size_t i, const std::size_t j) const
  {
    return vec[j * nx_ + i];
  }

  /**
   * @brief Copies global constants to the device.
   */
  void initialize_cuda_constants();

  /**
   * @brief Initializes h0, hu0, hv0, h1, hu1, hv1 arrays on the device.
   */
  void initialize_cuda_arrays();

  /**
   * @brief Copies the initial state h0, hu0, hv0 to the device. Also fills
   * h1, hu1, hv1 with zeros.
   */
  void copy_to_device(std::vector<double> &h,
                      std::vector<double> &hu,
                      std::vector<double> &hv);

  /** 
   * @brief Copies the current state h, hu, hv from the device to the host.
   */
  void copy_from_device(std::vector<double> &h,
                        std::vector<double> &hu,
                        std::vector<double> &hv);

  /**
   * @brief Deallocates device memory for the arrays.
   */
  void deallocate_device_arrays();
                    
  /**
   * @brief Swaps two buffers on the device.
   */
  inline void swap_buffers(double* &buf1, double* &buf2) {
    double* temp = buf1;
    buf1 = buf2;
    buf2 = temp;
  }
  
};