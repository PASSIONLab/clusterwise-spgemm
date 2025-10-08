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

#include <filesystem>
namespace fs = std::filesystem;

#include "../utility.h"
#include "../CSR.h"
#include "../multiply.h"

#include "../hash_mult.h"
#include "sample_common.hpp"

using namespace std;

#define VALUETYPE float
#define INDEXTYPE int64_t

int main(int argc, char *argv[]) {
  const bool sortOutput = false;

  const char* output_path = std::getenv("CLOSE_PAIR_DATA_PATH");
  if (!output_path) throw std::runtime_error("CLOSE_PAIR_DATA_PATH is not set");

  string filename = parseFileNamePrefixFromPath(argv[2]);
  string filename_suffix = create_filename_suffix(argv);
  string topk_outfile = output_path + '/' + filename_suffix;

  bool save_output = false;
  int tnums = 64;
  INDEXTYPE topk = 7;
  CSR<INDEXTYPE, VALUETYPE> A_csr, B_csr, C_csr;
  C_csr.is_value_sorted = sortOutput;

  if (argc < 5) {
    cout
        << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|matrix1.txt} {scale|matrix2.txt} <topk>"
        << endl;
    return -1;
  }
  topk = {atol(argv[4])};
  save_output = (argc > 5);
  INDEXTYPE cluster_size = 8;

  cout << "topk_outfile: " << topk_outfile << endl;

  /* Generating input matrices based on argument */
  CSC<INDEXTYPE, VALUETYPE> A_csc;
  SetInputMatricesAsCSC(A_csc, argv, cluster_size);
  A_csc.calculate_size();
  A_csr = *(new CSR<INDEXTYPE, VALUETYPE>(A_csc));

  // making B as A-transpose
  B_csr = *(new CSR<INDEXTYPE, VALUETYPE>(A_csc, true));

  A_csr.Sorted();
  B_csr.Sorted();

  A_csr.ResetValues();
  B_csr.ResetValues();

  A_csc.make_empty();
  A_csc.calculate_size();

  A_csr.calculate_size();
  B_csr.calculate_size();

  /* Count total number of floating-point operations */
  long long int nfop = get_flop(A_csr, B_csr);
  cout << "Total number of floating-point operations including addition and multiplication in SpGEMM (A * B): " << nfop << endl << endl;

  double start, end, msec;
  omp_set_num_threads(tnums);
  start = omp_get_wtime();

  HashSpGEMMTopK<sortOutput>(A_csr, B_csr, C_csr, multiplies<VALUETYPE>(), plus<VALUETYPE>(), topk, "");
  end = omp_get_wtime();
  msec = (end - start);
  printf("Dataset: %s done C = A * A_T in %f [seconds]\n", filename.c_str(), msec);

  if(save_output) {
    if (!sortOutput) C_csr.sortValues();
    C_csr.print_topK(topk_outfile, topk);
  }

  C_csr.make_empty();
  A_csr.make_empty();
  B_csr.make_empty();

//  A_csr.calculate_size();
//  B_csr.calculate_size();
//  C_csr.calculate_size();

  return 0;
}
