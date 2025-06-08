#include "swe.hh"

#include <string>
#include <cstddef>
#include <iostream>

#include <sys/stat.h>
#include <sys/types.h>

#include <chrono>

int
main(int argc, char *argv[])
{
  // Uncomment the option you want to run.
  
  // ./swe <block-dim-x> <block-dim-y> [<output-n>] [<full-log>] [<case> <grid-dim-x> <grid-dim-y>]
  // ./swe <block-dim-x> <block-dim-y> [<output-n>] [<full-log>] [<input-file> <size>]

  if (argc < 3 || argc > 8)
  {
    std::cout << "Usage: " << argv[0] << " <block-dim-x> <block-dim-y> [<output-n>] [<full-log>] [<case> <grid-dim-x> <grid-dim-y> | input-file]" << std::endl;
    return 1;
  }

  const int block_dim_x = std::stoi(argv[1]);
  const int block_dim_y = std::stoi(argv[2]);

  int output_n = 20; // If compiled with profiling, nothing is written no matter the value.
  if (argc > 3)
    output_n = std::stoi(argv[3]);

  bool full_log = false;
  if (argc > 4)
    full_log = std::stoi(argv[4]) != 0;

  double Tend = 1.0;     // Simulation time in hours

  std::size_t nx = 4096;
  std::size_t ny = 4096;
  int test_case_id = 1;
  if (argc == 8)
  {
    test_case_id = std::stoi(argv[5]);
    if (test_case_id < 1 || test_case_id > 2)
    {
      std::cerr << "Invalid test case ID. It should be 1 (water drops in a box) or 2 (analytical tsunami)." << std::endl;
      return 1;
    }
    nx = std::stoi(argv[6]);
    ny = std::stoi(argv[7]);
  }

  std::string input_file;
  double size = 500.0;
  if (argc == 6)
  {
    input_file = argv[5];
    test_case_id = 3;
    Tend = 0.2; // Default simulation time for tsunami case
  }
  else if (argc == 7)
  {
    input_file = argv[5];
    size = std::stod(argv[6]);
    test_case_id = 3;
    Tend = 0.2; // Default simulation time for tsunami case
  }
  

  std::string output_fname = "";
  if (test_case_id == 1)
    output_fname += "water_drops";
  else if (test_case_id == 2)
    output_fname += "analytical_tsunami";
  else if (test_case_id == 3)
    output_fname += "tsunami_" + input_file;

  output_fname += "_" + std::to_string(nx) + "x" + std::to_string(ny);
  output_fname += "_" + std::to_string(block_dim_x) + "x" + std::to_string(block_dim_y);

  struct stat st = {0};
  if (stat(output_fname.c_str(), &st) == -1) {
    mkdir(output_fname.c_str(), 0755);
  }
  
  output_fname += "/output";
  
  std::cout << "Running test case " << test_case_id << " with grid size " << nx << "x" << ny
            << ", and block size " << block_dim_x << "x" << block_dim_y << std::endl;

  std::chrono::_V2::system_clock::time_point start_time, end_time;
  if (test_case_id == 1 || test_case_id == 2)
  {
    SWESolver solver(test_case_id, nx, ny);
    start_time = std::chrono::system_clock::now();
    solver.solve(Tend, full_log, output_n, output_fname, block_dim_x, block_dim_y);
    end_time = std::chrono::system_clock::now();
  }
  else if (test_case_id == 3)
  {
    SWESolver solver(input_file, size, size);
    start_time = std::chrono::system_clock::now();
    solver.solve(Tend, full_log, output_n, output_fname, block_dim_x, block_dim_y);
    end_time = std::chrono::system_clock::now();
  }

  std::cout << "Total solve time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
            << " ms" << std::endl;

  return 0;
}
