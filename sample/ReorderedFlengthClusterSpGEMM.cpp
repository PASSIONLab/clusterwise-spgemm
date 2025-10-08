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
#include "../cluster_utility.h"
#include "../CSR.h"
#include "../CSR_FlengthCluster.h"
#include "../multiply.h"

#include "../hash_mult.h"
#include "../hash_mult_flengthcluster.h"
#include "sample_common.hpp"

using namespace std;

#define VALUETYPE double
#define INDEXTYPE int64_t

#define VALIDATE 0

int main(int argc, char *argv[]) {
  string filename_suffix = create_filename_suffix(argv);

  INDEXTYPE cluster_size = 8;
  const bool sortOutput = false;
  vector<int> tnums = {64};
  CSR<INDEXTYPE, VALUETYPE> A_csr_original;

  if (argc < 4) {
    cout
        << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|matrix1.txt} {shuffle.txt} {edgefactor|product.txt} <numthreads>"
        << endl;
    return -1;
  }
  if (argc >= 5) {
    cout
        << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|matrix1.txt} {shuffle.txt} <cluster_size>"
        << endl;
    cluster_size = atoi(argv[4]);
  }
  if (argc >= 6) {
    cout
        << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|matrix1.txt} {shuffle.txt} <cluster_size> <signature_length> <band_size> <numthreads>"
        << endl;
    tnums = {atoi(argv[5])};
  }

  cout << "cluster_size: " << cluster_size << endl;

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

  CSR<INDEXTYPE, VALUETYPE> B_csr(A_csr_original);
  B_csr.sortIds();

  A_csr_original.make_empty();

  /* Count total number of floating-point operations */
  long long int nfop = get_flop(A_csr, B_csr);
  cout << "Total number of floating-point operations including addition and multiplication in SpGEMM (A * B): " << nfop
       << endl << endl;

  double start, end, msec, ave_msec, mflops;

  // create A_csr_flength_cluster from A_csr
  CSR_FlengthCluster<INDEXTYPE, VALUETYPE> A_csr_flength_cluster(A_csr, cluster_size);

  /* Execute HashSpGEMMCluster */
  cout << "Evaluation of HashSpGEMMCluster" << endl;
  for (int tnum: tnums) {
    omp_set_num_threads(tnum);

    CSR_FlengthCluster<INDEXTYPE, VALUETYPE> C_csr_flength_cluster;

    /* First execution is excluded from evaluation */
    HashSpGEMMCluster<sortOutput>(A_csr_flength_cluster, B_csr, C_csr_flength_cluster, multiplies<VALUETYPE>(), plus<VALUETYPE>());
    C_csr_flength_cluster.make_empty();

    ave_msec = 0;
    for (int i = 0; i < ITERS; ++i) {
      start = omp_get_wtime();
      HashSpGEMMCluster<sortOutput>(A_csr_flength_cluster, B_csr, C_csr_flength_cluster, multiplies<VALUETYPE>(), plus<VALUETYPE>());
      end = omp_get_wtime();
      msec = (end - start) * 1000;
      ave_msec += msec;
      if (i < ITERS - 1) {
        C_csr_flength_cluster.make_empty();
      }
    }
    ave_msec /= ITERS;
    mflops = (double) nfop / ave_msec / 1000;

    printf("HashSpGEMMNonReorderCluster with %3d threads computes C = A * B in %f [milli seconds] (%f [MFLOPS])\n",
           tnum, ave_msec, mflops);

#if VALIDATE
    start = omp_get_wtime();
    CSR<INDEXTYPE,VALUETYPE> C_csr;
    csr_flength_cluster2csr(C_csr, C_csr_flength_cluster);
    end = omp_get_wtime();
    cout << "Reconstruct output CSR time: " << (end - start) * 1000 << " [milli seconds]" << endl;

    C_csr.sortIds();
    CSR<INDEXTYPE,VALUETYPE> C_csr_hash;
    RowSpGEMM<false, sortOutput>(A_csr, B_csr, C_csr_hash, multiplies<VALUETYPE>(), plus<VALUETYPE>(), "");
    if(!sortOutput) C_csr_hash.sortIds();
    cout << "RowSpGEMM == HashSpGEMMCluster ? " << (C_csr_hash==C_csr) << endl << endl;
    C_csr_hash.make_empty();
    C_csr.make_empty();
#endif
    C_csr_flength_cluster.make_empty();
  }

  A_csr.make_empty();
  B_csr.make_empty();

  return 0;
}
