#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>
#include <iterator>
#include <ctime>
#include <cmath>
#include <string>
#include <sstream>
#include <random>
#include <chrono>

#include <filesystem>
namespace fs = std::filesystem;

#include "../utility.h"
#include "../CSR.h"
#include "../multiply.h"

#include "../hash_mult.h"
#include "sample_common.hpp"

using namespace std;

#define VALUETYPE double
#define INDEXTYPE int64_t

int main(int argc, char *argv[]) {
  bool save_to_file = false;
  INDEXTYPE cluster_size = 8;
  CSR<INDEXTYPE, VALUETYPE> A_csr, B_csr;

  if (argc < 3) {
    cout << "Normal usage: ./shuffle {gen|binary|text} matrix1.txt" << endl;
    return -1;
  }

  if(argc == 4) save_to_file = true;

  char* output_path = std::getenv("SHUFFLED_DATA_PATH");
  if (!output_path) throw std::runtime_error("SHUFFLED_DATA_PATH is not set");

  /* Generating input matrices based on argument */
  SetInputMatricesAsCSR(A_csr, argv, cluster_size);
  A_csr.Sorted();

  vector<INDEXTYPE> row_id(A_csr.rows);
  for (int i = 0; i < A_csr.rows; i += 1) row_id[i] = i;

  double start, end, sec;
  start = omp_get_wtime();
  mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
  shuffle(row_id.begin(), row_id.end(), rng);
  end = omp_get_wtime();
  sec = (end - start);

  string filename = parseFileNameFromPath(argv[2]);
  cout << "Dataset " << filename << " takes: " << sec << " seconds." << endl;

  if(save_to_file) {
    string filename_suffix = create_filename_suffix(argv);
    string outfile = output_path + '/' + filename_suffix;

    std::fstream file(outfile, std::ios::out);
    for (INDEXTYPE i = 0; i < A_csr.rows; ++i) {
      file << row_id[i] << endl;
    }
    file.close();
    cout << "[DONE] Writing to file!" << endl;
  }

  A_csr.make_empty();

  return 0;
}
