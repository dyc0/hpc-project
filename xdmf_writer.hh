
#include <vector>
#include <string>

/**
 * @brief This class is used to write the mesh and the solution of the SWE to an XDMF file that read HDF5 files.
 */
class XDMFWriter
{
public:
  /**
   * @brief Constructor for the XDMFWriter class.
   * @param filename_prefix The prefix of the filename to write.
   * @param nx The number of cells in the x direction.
   * @param ny The number of cells in the y direction.
   * @param size_x The size of the domain in the x direction.
   * @param size_y The size of the domain in the y direction.
   * @param topography Topography field (one value per point).
   */
  XDMFWriter( const std::string& filename_prefix,
              const std::size_t nx,
              const std::size_t ny,
              const std::size_t m_nx,
              const std::size_t m_ny,
              const std::size_t size_x,
              const std::size_t size_y,
              const std::size_t m_start_coords[2],
              const int c_coords[2],
              const int c_dims[2],
              const int w_rank,
              const std::vector<double>& topography);

  /**
   * @brief Adds a new solution step to the XDMF file.
   *
   * It updates the XDMF file to include the new solution step and
   * writes the solution to an HDF5 file.
   *
   * @brief h The water height solution to write.
   * @brief t The solution's instant.
   */
  void add_h(const std::vector<double>& h, const double t);

private:
  /**
   * @brief Write the root XDMF file that calls the HDF5 files for the
   * mesh and the solution at the time steps.
   *
   * This function may be called after each time step to update the XDMF file
   * for including the last generated solution.
   */
  void write_root_xdmf() const;

  /**
   * @brief Write the topography to an HDF5 file.
   * @param topography The topography field to write.
   */
  void write_topography_hdf5(const std::vector<double>& topography);

  /**
   * @brief Create the vertices of the mesh.
   * @param vertices The vector to store the vertices.
   */
  void create_vertices(std::vector<double>& vertices) const;

  /**
   * @brief Create the cells of the mesh.
   * @param cells The vector to store the cells.
   */
  void create_cells(std::vector<int>& cells) const;

  /**
   * @brief Write the mesh (vertices + cells) to an HDF5 file.
   */
  void write_mesh_hdf5() const;

  /**
   * @brief Write a 1D array to an HDF5 file.
   * @param filename The name of the HDF5 file.
   * @param dataset_name The name of the dataset to write.
   * @param data The data to write.
   */
  void write_array_to_hdf5( const std::string& filename,
                            const std::string& dataset_name,
                            const std::vector<double>& data);

  
  void prepare_write_buffer(const std::vector<double>& data);

  /// @brief The prefix of the filename to write.
  const std::string filename_prefix_;
  /// @brief The number of cells per direction in the full mesh.
  std::size_t m_nx_, m_ny_;
  /// @brief The number of cells per direction in the local mesh.
  const std::size_t nx_, ny_;
  /// @brief The start of the real data in the local mesh.
  std::size_t start_x_, start_y_;
  /// @brief The size of the domain per direction in km.
  const std::size_t size_x_, size_y_;
  /// @brief The number of real cells in the local mesh.
  // std::size_t real_nx_, real_ny_;
  /// @brief The time steps of the solution.
  std::vector<double> time_steps_;
  /// @brief Coordinates of the process in the Cartesian grid.
  int c_coords_[2], c_dims_[2];
  /// @brief Coordinates of the top-left corner of the local grid in the full matrix.
  size_t m_start_coords_[2];
  /// @brief The rank of the process in the World communicator.
  int w_rank_;
  /// @brief The buffer from which the data will be written to the HDF5 file.
  std::vector<double> write_buffer_;
};
