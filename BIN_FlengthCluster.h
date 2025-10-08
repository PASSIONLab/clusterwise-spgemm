#ifndef BIN_FLENGTH_CLUSTER_H
#define BIN_FLENGTH_CLUSTER_H

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>
#include <algorithm>

#include "utility.h"

/*
 * Manage the group
 * prepare hash table for each group
 */
template<class IT, class NT>
class BIN_FlengthCluster {
public:
  BIN_FlengthCluster() : total_intprod(0), thread_num(omp_get_max_threads()) {
  }

  BIN_FlengthCluster(IT clusters, IT ht_size, IT cluster_size) : total_intprod(0), thread_num(omp_get_max_threads()),
                                                          min_ht_size(ht_size), cluster_sz(cluster_size) {
    num_clusters = clusters;
    cluster_nz = my_malloc<IT>(num_clusters);
    clusters_offset = my_malloc<IT>(thread_num + 1);
    bin_id_cluster = my_malloc<char>(num_clusters);

    local_hash_table_id = my_malloc<IT *>(thread_num);
    for (IT i = 0; i < thread_num; i += 1) local_hash_table_id[i] = nullptr;

    local_hash_table_val = my_malloc<NT *>(thread_num);

    total_size = 0;
    total_size += (num_clusters * sizeof(IT));             // cluster_nz
    total_size += ((thread_num + 1) * sizeof(IT));         // clusters_offset
    total_size += (num_clusters * sizeof(char));           // bin_id_cluster
  }

  ~BIN_FlengthCluster() {
    my_free<IT>(cluster_nz);
    my_free<IT>(clusters_offset);
    my_free<char>(bin_id_cluster);
    if (local_hash_table_id != nullptr) {
#pragma omp parallel
      {
        int tid = omp_get_thread_num();
        if (local_hash_table_id[tid] != nullptr) {
          my_free<IT>(local_hash_table_id[tid]);
          my_free<NT>(local_hash_table_val[tid]);
        }
      }
      my_free<IT *>(local_hash_table_id);
      my_free<NT *>(local_hash_table_val);
    }
  }

  size_t calculate_size();
  double calculate_size_in_gb();

  void set_max_bin(const IT *arpt, const IT *acol, const IT *brpt, const IT cols);

  void set_min_bin(const IT cols);

  void create_local_hash_table(const IT cols);

  void set_intprod_num(const IT *arpt, const IT *acol, const IT *brpt);

  void set_clusters_offset();

  void set_bin_id(const IT cols, const IT min);

  int64_t total_intprod;
  IT thread_num;
  IT min_ht_size;
  IT num_clusters;
  IT cluster_sz;

  IT *cluster_nz;                       // the number of flop or non-zero elements of output cluster-matrix
  IT *clusters_offset;                  // offset for cluster_nz

  char *bin_id_cluster;                 // hash table size for cluster[i] will be 2^bin_id_cluster[i]

  IT **local_hash_table_id = nullptr;   // col-ids in the output matrix
  NT **local_hash_table_val = nullptr;  // values in the output matrix

  size_t total_size;
};

template<class IT, class NT>
double BIN_FlengthCluster<IT, NT>::calculate_size_in_gb() {
  size_t size_bytes = 0;

  // cluster_nz
  size_bytes += num_clusters * sizeof(IT);

  // clusters_offset
  size_bytes += (thread_num + 1) * sizeof(IT);

  // bin_id_cluster
  size_bytes += num_clusters * sizeof(char);

  // local_hash_table_id pointers
  size_bytes += thread_num * sizeof(IT *);

  // local_hash_table_val pointers
  size_bytes += thread_num * sizeof(NT *);

  // Add per-thread allocations (if they exist)
  for (IT i = 0; i < thread_num; ++i) {
    if (local_hash_table_id && local_hash_table_id[i]) {
      size_bytes += min_ht_size * sizeof(IT);  // assuming same size across threads
    }
    if (local_hash_table_val && local_hash_table_val[i]) {
      size_bytes += min_ht_size * sizeof(NT);  // assuming same size across threads
    }
  }

//  // Store total
//  total_size = size_bytes;

  // Convert to GB (1 GB = 2^30 bytes)
//  return static_cast<double>(size_bytes) / (1024.0 * 1024.0 * 1024.0);
  return static_cast<double>(size_bytes);
}

//! calculate size in Bytes
template <class IT, class NT>
size_t BIN_FlengthCluster<IT, NT>::calculate_size() {
  return total_size;
}

/* Count the number of intermediate products per cluster (= flop / 2) */
template<class IT, class NT>
inline void BIN_FlengthCluster<IT, NT>::set_intprod_num(const IT *arpt, const IT *acol, const IT *brpt) {
#pragma omp parallel
  {
    IT each_int_prod = 0;
#pragma omp for
    for (IT i = 0; i < num_clusters; ++i) {                       // loop over clusters
      IT nz_per_cluster = 0;
      for (IT j = arpt[i]; j < arpt[i + 1]; ++j) {                // col-ids of i-th cluster
        nz_per_cluster += (brpt[acol[j] + 1] - brpt[acol[j]]);    // summing up rows of B
      }
      cluster_nz[i] = nz_per_cluster;                             // cluster_nz is populated by flops
      each_int_prod += (nz_per_cluster * cluster_sz);
    }
#pragma omp atomic
    total_intprod += each_int_prod;
  }
}

/* Get total number of floating operations and average
 * then, use it for assigning clusters to thread as the amount of work is equally distributed
 */
template<class IT, class NT>
inline void BIN_FlengthCluster<IT, NT>::set_clusters_offset() {
  // cluster_nz only counts the sum of rows of B
  // so, prefix sum of intermediate products performed on: (cluster_nz[i] * cluster_sz)
  IT *ps_cluster_nz = my_malloc<IT>(num_clusters + 1);
  scan(cluster_nz, ps_cluster_nz, cluster_sz, num_clusters + 1);

  IT average_intprod = (total_intprod + thread_num - 1) / thread_num;

  /* Search end point of each range */
  clusters_offset[0] = 0;
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    long long int end_itr =
        (lower_bound(ps_cluster_nz, ps_cluster_nz + num_clusters + 1, average_intprod * (tid + 1))) - ps_cluster_nz;
    clusters_offset[tid + 1] = end_itr;
  }
  clusters_offset[thread_num] = num_clusters;
  my_free<IT>(ps_cluster_nz);
}

/*
 * Prepare hash table for each thread_num
 * once allocate memory space for hash table, the thread reuse it for each cluster
 * hash table size is determined by the cluster_nz values
 */
template<class IT, class NT>
inline void BIN_FlengthCluster<IT, NT>::create_local_hash_table(const IT cols) {
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    IT ht_size = 0;

    /* Get max size of hash table */
    for (IT j = clusters_offset[tid]; j < clusters_offset[tid + 1]; ++j) {
      if (ht_size < cluster_nz[j]) ht_size = cluster_nz[j];
    }
    /* the size of hash table is aligned as 2^n */
    if (ht_size > 0) {
      if (ht_size > cols) ht_size = cols;
      int k = min_ht_size;
      while (k < ht_size) {
        k <<= 1;
      }
      ht_size = k;
    }

    local_hash_table_id[tid] = my_malloc<IT>(ht_size);
    local_hash_table_val[tid] = my_malloc<NT>(ht_size * cluster_sz);

    total_size += (ht_size * sizeof(IT));                   // local_hash_table_id
    total_size += ((ht_size * cluster_sz) * sizeof(NT));    // local_hash_table_val
  }
}

/*
 * Precompute how many entries each row requires for the hash table
 * the size is 2^bin_id_cluster
 */
template<class IT, class NT>
inline void BIN_FlengthCluster<IT, NT>::set_bin_id(const IT cols, const IT min) {
  IT i;
#pragma omp parallel for
  for (i = 0; i < num_clusters; ++i) {
    IT j;
    IT nz_per_cluster = cluster_nz[i];
    if (nz_per_cluster > cols) nz_per_cluster = cols;
    if (nz_per_cluster == 0) {
      bin_id_cluster[i] = 0;
    } else {
      j = 0;
      while (nz_per_cluster > (min << j)) {
        j++;
      }
      bin_id_cluster[i] = j + 1;
    }
  }
}

/* grouping and preparing hash table based on the number of floating operations */
template<class IT, class NT>
inline void BIN_FlengthCluster<IT, NT>::set_max_bin(const IT *arpt, const IT *acol, const IT *brpt, const IT cols) {
  set_intprod_num(arpt, acol, brpt);
  set_clusters_offset();
  set_bin_id(cols, min_ht_size);
}

/* Reset the size of hash table which each row requires */
template<class IT, class NT>
inline void BIN_FlengthCluster<IT, NT>::set_min_bin(const IT cols) {
  set_bin_id(cols, min_ht_size);
}

#endif  //BIN_FLENGTH_CLUSTER_H