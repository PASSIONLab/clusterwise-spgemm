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
#include "../CSR_VlengthCluster.h"
#include "../multiply.h"

#include "../hash_mult.h"
#include "../hash_mult_vlengthcluster.h"
#include "sample_common.hpp"

using namespace std;

#define VALUETYPE double
#define INDEXTYPE int64_t

#define VALIDATE 0
#define MAX_CLUSTER_SIZE 8


int main(int argc, char *argv[]) {
  VALUETYPE similarity_th = 0.3;
  VALUETYPE eps = 0.000001f;
  INDEXTYPE max_cluster_size = 8;
  INDEXTYPE cluster_size_used_in_shuffle_input = 8;
  const bool sortOutput = false;
  vector<int> tnums;
  CSR<INDEXTYPE, VALUETYPE> A_csr_original;

  if (argc < 4) {
    cout
        << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|matrix1.txt} {shuffle.txt} <max_cluster_size> <numthreads>"
        << endl;
    return -1;
  } else if (argc < 6) {
    cout
        << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|matrix1.txt} {shuffle.txt} <max_cluster_size> <numthreads>"
        << endl;

    cout << "Running on 1, 16, 32, 64 threads" << endl;
    tnums = {64};
    if (argc == 5) max_cluster_size = atoi(argv[4]);
  } else {
    cout << "Running on " << argv[5] << " processors" << endl << endl;
    max_cluster_size = atoi(argv[4]);
    tnums = {atoi(argv[5])};
  }

  cout << "max_cluster_size: " << max_cluster_size << endl;

  SetInputMatricesAsCSR(A_csr_original, argv, cluster_size_used_in_shuffle_input);
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
//    assert(max_cluster_size != -1);
    if(shuffled_rows.size()%cluster_size_used_in_shuffle_input != 0) {
      tmp = ((shuffled_rows.size() / cluster_size_used_in_shuffle_input) + 1) * cluster_size_used_in_shuffle_input;
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

  vector<INDEXTYPE> offset;
  INDEXTYPE curr_off = 0;
  offset.push_back(curr_off);
  VALUETYPE sim_score;

  INDEXTYPE real_max_cluster_size = 0;
  while(curr_off < A_csr.rows) {
    INDEXTYPE next_off = curr_off + 1;
    while (next_off < A_csr.rows) {
      sim_score = A_csr.jaccard_similarity(curr_off, next_off);
      if(sim_score < similarity_th - eps) break;
      if(max_cluster_size != -1 && next_off - curr_off == max_cluster_size) break;
      next_off += 1;
    }
    real_max_cluster_size = max(real_max_cluster_size, (next_off - curr_off));
    offset.push_back(next_off);
    curr_off = next_off;
  }

  cout << "# of clusters: " << offset.size() << endl;
  cout << "max_cluster_size for SpGEMM: " << real_max_cluster_size << endl;

  // create A_csr_vlength_cluster from A_csr and reconstructed_clusters
  CSR_VlengthCluster<INDEXTYPE, VALUETYPE> A_csr_vlength_cluster(A_csr, offset);

  /* Execute HashSpGEMMVLCluster */
  cout << "Evaluation of HashSpGEMMVLCluster" << endl;
  for (int tnum: tnums) {
    omp_set_num_threads(tnum);

    CSR_VlengthCluster<INDEXTYPE, VALUETYPE> C_csr_vlength_cluster;

    /* First execution is excluded from evaluation */
    HashSpGEMMVLCluster<sortOutput>(A_csr_vlength_cluster, B_csr, C_csr_vlength_cluster, multiplies<VALUETYPE>(), plus<VALUETYPE>());
    C_csr_vlength_cluster.make_empty();

    ave_msec = 0;
    for (int i = 0; i < ITERS; ++i) {
      start = omp_get_wtime();
      HashSpGEMMVLCluster<sortOutput>(A_csr_vlength_cluster, B_csr, C_csr_vlength_cluster, multiplies<VALUETYPE>(), plus<VALUETYPE>());
      end = omp_get_wtime();
      msec = (end - start) * 1000;
      ave_msec += msec;
      if (i < ITERS - 1) {
        C_csr_vlength_cluster.make_empty();
      }
    }
    ave_msec /= ITERS;
    mflops = (double) nfop / ave_msec / 1000;

    printf("HashSpGEMMVLCluster with %3d threads computes C = A * B in %f [milli seconds] (%f [MFLOPS])\n",
           tnum, ave_msec, mflops);

#if VALIDATE
    // convert C_csr_vlength_cluster to CSR format and save in C_csr
    start = omp_get_wtime();
    CSR<INDEXTYPE,VALUETYPE> C_csr;
    csr_vlength_cluster2csr(C_csr, C_csr_vlength_cluster);
    end = omp_get_wtime();
    cout << "Reconstruct output CSR time: " << (end - start) * 1000 << " [milli seconds]" << endl;
    C_csr.sortIds();

    // reconstruct A_csr by reordering the rows and save in A_csr_new
    // we have to do this because we did HashSpGEMMVLCluster on the modified A_csr
    //    - by creating CSR_VlengthCluster by accessing rows of A_csr in pp.final_clusters order
    // so to compare HashSpGEMMVLCluster result with RowSpGEMM, we need to perform (A_csr_new * B_csr)
    Triple<INDEXTYPE, VALUETYPE> *triples = new Triple<INDEXTYPE, VALUETYPE>[A_csr.nnz];
    INDEXTYPE triplet_id = 0;
    INDEXTYPE row_id = 0;
    for (auto t: pp.final_reordered_rows) {
      for (INDEXTYPE idx = A_csr.rowptr[t]; idx < A_csr.rowptr[t + 1]; idx += 1) {
        triples[triplet_id] = Triple<INDEXTYPE, VALUETYPE>(row_id, A_csr.colids[idx], A_csr.values[idx]);
        triplet_id += 1;
      }
      row_id += 1;
    }

    CSR<INDEXTYPE, VALUETYPE> A_csr_new(triples, A_csr.nnz, A_csr.rows, A_csr.cols);
    A_csr_new.sortIds();

    CSR<INDEXTYPE,VALUETYPE> C_csr_hash;
    RowSpGEMM<false, sortOutput>(A_csr_new, B_csr, C_csr_hash, multiplies<VALUETYPE>(), plus<VALUETYPE>(), "");
    if(!sortOutput) C_csr_hash.sortIds();

    cout << "RowSpGEMM == HashSpGEMMVLCluster ? " << (C_csr_hash==C_csr) << endl << endl;

    C_csr.make_empty();
    A_csr_new.make_empty();
    C_csr_hash.make_empty();
#endif
    C_csr_vlength_cluster.make_empty();
  }

  A_csr.make_empty();
  B_csr.make_empty();

  return 0;
}
