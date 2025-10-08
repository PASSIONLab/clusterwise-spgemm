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

#define VALUETYPE double
#define INDEXTYPE int64_t

inline string get_bc_frontier_file_name(string directory, int fidx) {
  return directory + "/batch_0_forwarditer_" + to_string(fidx) + ".mtx";
}

int main(int argc, char* argv[])
{
  const bool sortOutput = false;
  vector<int> tnums;
  CSR<INDEXTYPE,VALUETYPE> A_csr, B_csr, C_csr;

  if (argc < 4) {
    cout << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|matrix1.txt} {scale|matrix2.txt} {edgefactor|product.txt} <numthreads>" << endl;
    return -1;
  }
  else if (argc < 6) {
    cout << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|matrix1.txt} {scale|matrix2.txt} {edgefactor|product.txt} <numthreads>" << endl;

    cout << "Running on 64 threads" << endl;
    tnums = {64};
  }
  else {
    cout << "Running on " << argv[6] << " processors" << endl << endl;
    tnums = {atoi(argv[5])};
  }

  /* Generating input matrices based on argument */
  SetInputMatricesAsCSR(A_csr, argv);
  A_csr.Sorted();

  string dataset = parseFileNamePrefixFromPathWithSanitization(argv[2]);
  string directory = string(argv[3]) + "/" + dataset;
  cout << "Reading tall skinny matrix from directory: " << directory << endl;
  int num_files = get_file_count(directory);
  cout << "Number of tall-skinny matrix: " << num_files << endl;

  /* Execute Hash-SpGEMM */
  cout << "Evaluation of RowSpGEMM" << endl;
  for (int tnum: tnums) {
    omp_set_num_threads(tnum);

    for(int fidx=0; fidx<num_files; fidx+=1) {
      string ts_file = get_bc_frontier_file_name(directory, fidx);
      cout << "Reading tall skinny matrix from file: " << ts_file << endl;
      ReadMatrixFromMtxFile(B_csr, ts_file);
      B_csr.Sorted();

//      cout << "B_csr: " << B_csr.rows << " x " << B_csr.cols << " nnz: " << B_csr.nnz << endl;

      /* Count total number of floating-point operations */
      long long int nfop = get_flop(A_csr, B_csr);
      cout << "Tall Skinny File #" << fidx << ": Total number of floating-point operations including addition and multiplication in SpGEMM (A * B): "
           << nfop << endl;

      double start, end, msec, ave_msec, mflops;

      /* First execution is excluded from evaluation */
      RowSpGEMM<false, sortOutput>(A_csr, B_csr, C_csr, multiplies<VALUETYPE>(), plus<VALUETYPE>(), "");
      C_csr.make_empty();

      ave_msec = 0;
      for (int i = 0; i < ITERS; ++i) {
        start = omp_get_wtime();
        RowSpGEMM<false, sortOutput>(A_csr, B_csr, C_csr, multiplies<VALUETYPE>(), plus<VALUETYPE>(), "");
        end = omp_get_wtime();
        msec = (end - start) * 1000;
        ave_msec += msec;
        if (i < ITERS - 1) {
          C_csr.make_empty();
        }
      }
      ave_msec /= ITERS;
      mflops = (double) nfop / ave_msec / 1000;

//        printf("RowSpGEMM returned with %d nonzeros. Compression ratio is %f\n", C_csr.nnz, (double)(nfop / 2) / (double)(C_csr.nnz));
      cout << "Tall Skinny File #" << fidx << ": RowSpGEMM returned with " << C_csr.nnz << " nonzeros. Compression ratio is "
           << ((double) (nfop / 2) / (double) (C_csr.nnz)) << endl;
      printf("Tall Skinny File #%d: RowSpGEMM with %3d threads computes C = A * B in %f [milli seconds] (%f [MFLOPS])\n\n", fidx, tnum, ave_msec,
             mflops);

      C_csr.make_empty();
      B_csr.make_empty();
    }
  }

  A_csr.make_empty();
  return 0;
}
