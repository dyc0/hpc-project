#include "swe.hh"

#include <string>
#include <cstddef>

#include <algorithm>
#include <cmath>
#include <iostream>

#include <mpi.h>

int
main(int argc, char ** argv)
{
  MPI_Init(&argc, &argv);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // **************************************************************
  // ********************** SIMULATION PARAMETERS *****************
  // **************************************************************
  const double Tend = 1.0;     // Simulation time in hours
  const std::size_t nx = 1000; // Number of cells per direction.
  const std::size_t ny = 1000; // Number of cells per direction.
  const std::size_t output_n = 20; // For profiling, I use 0
                                  // TODO: Come back to this and profile IO

                                  
  // **************************************************************
  // ********************** MPI INITIALIZATION *********************
  // **************************************************************
  // Note: for nx, ny undefined, we can leave dims = {0, 0} and let MPI decide the dimensions.
  int dims[2] = {0, 0};
  dims[0] = std::max(1, static_cast<int>(std::sqrt(world_size * ny / static_cast<double>(nx))));
  dims[1] = world_size / dims[0];
  MPI_Dims_create(world_size, 2, dims);

  int periods[2] = {0, 0}; // Non-periodic
  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);


  // **************************************************************
  // ********************** TEST CASE 1 ***************************
  // **************************************************************
  // Uncomment the option you want to run.
  // Option 1 - Solving simple problem: water drops in a box
  const int test_case_id = 1;  // Water drops in a box
  const std::string output_fname = "parallel_tests/water_drops";
  const bool full_log = true;

  SWESolver solver(test_case_id, nx, ny, cart_comm, dims);
  solver.solve(Tend, full_log, output_n, output_fname);
  


  // // Option 2 - Solving analytical (dummy) tsunami example.
  // const int test_case_id = 2;  // Analytical tsunami test case
  // const std::string output_fname = "analytical_tsunami";
  // const bool full_log = false;

  // SWESolver solver(test_case_id, nx, ny, cart_comm, dims);
  // solver.solve(Tend, full_log, output_n, output_fname);

  // // Option 3 - Solving tsunami problem with data loaded from file.
  // const double Tend = 0.2;   // Simulation time in hours
  // const double size = 500.0; // Size of the domain in km

  // // const std::string fname = "Data_nx501_500km.h5"; // File containg initial data (501x501 mesh).
  // const std::string fname = "Data_nx1001_500km.h5"; // File containg initial data (1001x1001 mesh).
  // // const std::string fname = "Data_nx2001_500km.h5"; // File containg initial data (2001x2001 mesh).
  // // const std::string fname = "Data_nx4001_500km.h5"; // File containg initial data (4001x4001 mesh).
  // // const std::string fname = "Data_nx8001_500km.h5"; // File containg initial data (8001x8001 mesh).

  // const std::size_t output_n = 0;
  // const std::string output_fname = "tsunami";
  // const bool full_log = false;

  // SWESolver solver(fname, size, size);
  // solver.solve(Tend, full_log, output_n, output_fname);

  MPI_Comm_free(&cart_comm);
  MPI_Finalize();

  return 0;
}
