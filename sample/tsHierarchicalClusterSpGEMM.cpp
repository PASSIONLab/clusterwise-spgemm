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
#include <queue>

#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <filesystem>
namespace fs = std::filesystem;

#include "../utility.h"
#include "../cluster_utility.h"
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

inline string get_bc_frontier_file_name(string directory, int fidx) {
  return directory + "/batch_0_forwarditer_" + to_string(fidx) + ".mtx";
}

using item_t = pair<VALUETYPE, pair<INDEXTYPE, INDEXTYPE>>;
auto cmp = [](const item_t &a, const item_t &b){ return a.first < b.first; };
priority_queue<item_t, vector<item_t>, decltype(cmp)> sims(cmp);

static map<INDEXTYPE, vector<INDEXTYPE>> hierachical_clustering_v0(CSR<INDEXTYPE, VALUETYPE> &sp,
                                                                   map<pair<INDEXTYPE, INDEXTYPE>, VALUETYPE> &close_pairs,
                                                                   int cluster_size) {


  for (auto &p: close_pairs) {
    sims.push(make_pair(p.second, p.first));
  }
  vector<INDEXTYPE> clusters(sp.rows);
  vector<INDEXTYPE> sz(sp.rows);
  vector<INDEXTYPE> valid(sp.rows);
  int nclusters = sp.rows;
  map<int, int> row_to_cluster;
  for (int i=0; i<sp.rows; i++) {
    clusters[i] = i;
    sz[i] = 1;
    valid[i] = 1;
  }

  while (!sims.empty() && nclusters != 0) {
    item_t s = sims.top();
    sims.pop();
    int i = s.second.first;
    int j = s.second.second;
    if (clusters[i] == i && clusters[j] == j) {
      if (!valid[i] || !valid[j]) continue;
      nclusters--;
      if (sz[i] < sz[j]) {
        clusters[i] = j;
        sz[j] += sz[i];
        if (sz[j] >= cluster_size) {
          valid[j] = 0;
          nclusters--;
        }
      } else {
        clusters[j] = i;
        sz[i] += sz[j];
        if (sz[i] >= cluster_size) {
          valid[i] = 0;
          nclusters--;
        }
      }
    } else {
      while (i != clusters[i]) {
        clusters[i] = clusters[clusters[i]];
        i = clusters[i];
      }
      while (j != clusters[j]) {
        clusters[j] = clusters[clusters[j]];
        j = clusters[j];
      }
      if (!valid[i] || !valid[j]) continue;
      if (i != j) {
        auto p = make_pair(i, j);
        if (close_pairs.find(p) == close_pairs.end()) {
          VALUETYPE s = sp.jaccard_similarity(i, j);// LSH::jaccard_similarity_v1(LSH::getkeys(sp[i].second), LSH::getkeys(sp[j].second));
          sims.push(make_pair(s, p));
          close_pairs[p] = s;
        }
      }
    }
  }

  map<INDEXTYPE, vector<INDEXTYPE>> reordered_dict;

  for (int i=0; i<clusters.size(); i++) {
    int j = i;
    while (j != clusters[j]) j = clusters[j];
    if (reordered_dict.find(j) == reordered_dict.end()) reordered_dict[j] = vector<INDEXTYPE>();
    reordered_dict[j].push_back(i);
  }

  return reordered_dict;
}

int main(int argc, char *argv[]) {
  INDEXTYPE cluster_size = 8;
  const bool sortOutput = false;
  vector<int> tnums = {64};
  CSR<INDEXTYPE, VALUETYPE> A_csr, B_csr;
  CSR_VlengthCluster<INDEXTYPE, VALUETYPE> C_csr_vlength_cluster;

  if (argc < 5) {
    cout
        << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|matrix1.txt} {ts_directory} {edgefactor|candidate_pairs.txt}"
        << endl;
    return -1;
  }
  if (argc >= 6) {
    cout
        << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|matrix1.txt} {ts_directory} {edgefactor|candidate_pairs.txt} <cluster_size>"
        << endl;
    cluster_size = atoi(argv[5]);
  }
  if (argc >= 7) {
    cout
        << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|matrix1.txt} {ts_directory} {edgefactor|candidate_pairs.txt} <cluster_size> <thread_num>"
        << endl;
    tnums = {atoi(argv[6])};
  }

  /* Generating input matrices based on argument */
  SetInputMatricesAsCSR(A_csr, argv);
  A_csr.Sorted();

  map<pair<INDEXTYPE, INDEXTYPE>, VALUETYPE> close_pairs;
  std::ifstream closepair_file(argv[4]);
  if (!closepair_file.is_open()) {
    std::cout << "Couldn't open file " << argv[3] << std::endl;
    std::exit(-2);
  }

  INDEXTYPE u, v;
  VALUETYPE common;
  while(closepair_file >> u >> v >> common) {
    close_pairs.insert(make_pair(make_pair(u, v), common));
  }
  closepair_file.close();

  // set the highest number of allowed concurrent threads to run clustering/reordering algorithm
  omp_set_num_threads(tnums[tnums.size() - 1]);
  map<INDEXTYPE, vector<INDEXTYPE>> clusters = hierachical_clustering_v0(A_csr, close_pairs, cluster_size);

  // create A_csr_flength_cluster from A_csr and reconstructed_clusters
  CSR_VlengthCluster<INDEXTYPE, VALUETYPE> A_csr_vlength_cluster(A_csr, clusters);

#if !VALIDATE
  A_csr.make_empty();
#endif

  string dataset = parseFileNamePrefixFromPathWithSanitization(argv[2]);
  string directory = string(argv[3]) + "/" + dataset;
  cout << "Reading tall skinny matrix from directory: " << directory << endl;
  int num_files = get_file_count(directory);
  cout << "Number of tall-skinny matrix: " << num_files << endl;

  /* Execute HashSpGEMMCluster */
  cout << "Evaluation of HashSpGEMMCluster" << endl;
  for (int tnum: tnums) {
    omp_set_num_threads(tnum);

    for(int fidx=0; fidx<num_files; fidx+=1) {
      string ts_file = get_bc_frontier_file_name(directory, fidx);
      cout << "Reading tall skinny matrix from file: " << ts_file << endl;
      ReadMatrixFromMtxFile(B_csr, ts_file);
      B_csr.Sorted();

      double start, end, msec, ave_msec;

      /* First execution is excluded from evaluation */
      HashSpGEMMVLCluster<sortOutput>(A_csr_vlength_cluster, B_csr, C_csr_vlength_cluster, multiplies<VALUETYPE>(), plus<VALUETYPE>());
      C_csr_vlength_cluster.make_empty();

      ave_msec = 0;
      for (int i = 0; i < ITERS; ++i) {
        start = omp_get_wtime();
        HashSpGEMMVLCluster<sortOutput>(A_csr_vlength_cluster, B_csr, C_csr_vlength_cluster, multiplies<VALUETYPE>(),
                                        plus<VALUETYPE>());
        end = omp_get_wtime();
        msec = (end - start) * 1000;
        ave_msec += msec;
        if (i < ITERS - 1) {
          C_csr_vlength_cluster.make_empty();
        }
      }

      ave_msec /= ITERS;
      printf("Tall Skinny File #%d: HashSpGEMMVLCluster with %3d threads computes C = A * B in %f [milli seconds]\n", fidx, tnum, ave_msec);

#if VALIDATE
      start = omp_get_wtime();
      CSR<INDEXTYPE,VALUETYPE> C_csr;
      csr_vlength_cluster2csr(C_csr, C_csr_vlength_cluster);
      end = omp_get_wtime();
      cout << "Reconstruct output CSR time: " << (end - start) * 1000 << " [milli seconds]" << endl;

      C_csr.sortIds();
      CSR<INDEXTYPE,VALUETYPE> C_csr_hash;
      RowSpGEMM<false, sortOutput>(A_csr, B_csr, C_csr_hash, multiplies<VALUETYPE>(), plus<VALUETYPE>(), "");
      if(!sortOutput) C_csr_hash.sortIds();
      cout << "RowSpGEMM == HashSpGEMMVLCluster ? " << (C_csr_hash==C_csr) << endl << endl;
      C_csr_hash.make_empty();
      C_csr.make_empty();
#endif
      C_csr_vlength_cluster.make_empty();
      B_csr.make_empty();
    }
  }

#if VALIDATE
  A_csr.make_empty();
#endif
  A_csr_vlength_cluster.make_empty();

  return 0;
}
