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
  INDEXTYPE cluster_size = 8;
  const bool sortOutput = false;
  vector<int> tnums;
  CSR<INDEXTYPE, VALUETYPE> A_csr_original, B_csr, C_csr;

  if (argc < 5) {
    cout << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|matrix1.txt} {shuffle.txt} {tall-skinny.dir}" << endl;
    return -1;
  }
  else if (argc < 7) {
    cout << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|matrix1.txt} {shuffle.txt} {tall-skinny.dir} <numthreads>" << endl;

    cout << "Running on 64 threads" << endl;
    tnums = {64}; // for perlmutter
  }
  else {
    cout << "Running on " << argv[5] << " processors" << endl << endl;
    tnums = {atoi(argv[5])};
  }

  /* Generating input matrices based on argument */
  SetInputMatricesAsCSR(A_csr_original, argv, cluster_size);
  A_csr_original.sortIds();

  std::ifstream infile(argv[3]);
  if (!infile.is_open()) {
    std::cout << "Couldn't open file " << argv[3] << std::endl;
    std::exit(-2);
  }

  vector<INDEXTYPE> shuffled_rows;
  unordered_map<INDEXTYPE, INDEXTYPE> col_map;
  INDEXTYPE tmp;
  INDEXTYPE col_id = 0;
  while(infile >> tmp) {
    shuffled_rows.push_back(tmp);
    col_map[tmp] = col_id;
    col_id += 1;
  }

  if(shuffled_rows.size() != A_csr_original.rows) {
    if(shuffled_rows.size()%cluster_size != 0) {
      tmp = ((shuffled_rows.size() / cluster_size) + 1) * cluster_size;
      assert(tmp == A_csr_original.rows && "shuffle list should match with the matrix-row size after adjustment!");
      for(tmp = shuffled_rows.size(); tmp<A_csr_original.rows; tmp+=1) {
        shuffled_rows.push_back(tmp);
      }
    }
  }
  assert(shuffled_rows.size() == A_csr_original.rows && "shuffle list does not match with the matrix-row size!");

  // reconstruct A_csr/B_csr by reordering the rows of A_csr_original
  Triple<INDEXTYPE, VALUETYPE> *triples = new Triple<INDEXTYPE, VALUETYPE>[A_csr_original.nnz];
  INDEXTYPE triplet_id = 0;
  INDEXTYPE row_id = 0;
  for (auto t: shuffled_rows) {
    for (INDEXTYPE idx = A_csr_original.rowptr[t]; idx < A_csr_original.rowptr[t + 1]; idx += 1) {
//      triples[triplet_id] = Triple<INDEXTYPE, VALUETYPE>(row_id, A_csr_original.colids[idx], A_csr_original.values[idx]);
      triples[triplet_id] = Triple<INDEXTYPE, VALUETYPE>(col_map[t], A_csr_original.colids[idx], A_csr_original.values[idx]);
      triplet_id += 1;
    }
    row_id += 1;
  }

  CSR<INDEXTYPE, VALUETYPE> A_csr(triples, A_csr_original.nnz, A_csr_original.rows, A_csr_original.cols);
  A_csr.sortIds();
  A_csr_original.make_empty();

//  CSR<INDEXTYPE, VALUETYPE> B_csr(triples, A_csr_original.nnz, A_csr_original.rows, A_csr_original.cols);
//  B_csr.sortIds();

  string dataset = parseFileNamePrefixFromPath(argv[2]);
  string directory = string(argv[4]) + "/" + dataset;
  cout << "Reading tall skinny matrix from directory: " << directory << endl;
  int num_files = get_file_count(directory);
  cout << "Number of tall-skinny matrix: " << num_files << endl;

  /* Execute Hash-SpGEMM */
  cout << "Evaluation of RowSpGEMM" << endl;
  for (int tnum : tnums) {
    omp_set_num_threads(tnum);

    for(int fidx=0; fidx<num_files; fidx+=1) {
      string ts_file = get_bc_frontier_file_name(directory, fidx);
      cout << "Reading tall skinny matrix from file: " << ts_file << endl;
      ReadMatrixFromMtxFile(B_csr, ts_file, cluster_size);
      B_csr.sortIds();

      /* Count total number of floating-point operations */
      long long int nfop = get_flop(A_csr, B_csr);
      cout << "Total number of floating-point operations including addition and multiplication in SpGEMM (A * B): " << nfop << endl;

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
      mflops = (double)nfop / ave_msec / 1000;

      cout << "RowSpGEMM returned with " << C_csr.nnz << " nonzeros. Compression ratio is " << ((double)(nfop / 2) / (double)(C_csr.nnz)) << endl;
      printf("RowSpGEMM with %3d threads computes C = A * B in %f [milli seconds] (%f [MFLOPS])\n\n", tnum, ave_msec, mflops);

      C_csr.make_empty();
      B_csr.make_empty();
    }
  }

  A_csr.make_empty();

  return 0;
}
