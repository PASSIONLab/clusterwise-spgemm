#ifndef BIN_VLENGTH_CLUSTER_H
#define BIN_VLENGTH_CLUSTER_H

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
class BIN_VlengthCluster {
public:
  BIN_VlengthCluster() : total_intprod(0), max_intprod(0), max_nz(0), thread_num(omp_get_max_threads()) {
  }

  BIN_VlengthCluster(IT clusters, IT ht_size, const IT *cluster_size) : total_intprod(0), max_intprod(0), max_nz(0),
                                                                   thread_num(omp_get_max_threads()),
                                                                   min_ht_size(ht_size) {
    num_clusters = clusters;

    cluster_sz = my_malloc<IT>(num_clusters);
    memcpy(cluster_sz, cluster_size, sizeof(IT) * num_clusters);

    cluster_nz = my_malloc<IT>(num_clusters);
    clusters_offset = my_malloc<IT>(thread_num + 1);
    bin_id_cluster = my_malloc<char>(num_clusters);

    local_hash_table_id = my_malloc<IT *>(thread_num);
    for (IT i = 0; i < thread_num; i += 1) local_hash_table_id[i] = nullptr;

    local_hash_table_val = my_malloc<NT *>(thread_num);

//    ht_sz = my_malloc<IT>(thread_num);
//    ht_val_sz = my_malloc<IT>(thread_num);
  }

  ~BIN_VlengthCluster() {
    my_free<IT>(cluster_sz);
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
//    my_free<IT>(ht_sz);
//    my_free<IT>(ht_val_sz);
  }

  void set_max_bin(const IT *arpt, const IT *acol, const IT *brpt, const IT cols);

  void set_min_bin(const IT cols);

  void create_local_hash_table(const IT cols);

  void set_intprod_num(const IT *arpt, const IT *acol, const IT *brpt);

  void set_clusters_offset();

  void set_bin_id(const IT cols, const IT min);

  double calculate_size_in_gb();

  long long int total_intprod;
  IT max_intprod;
  IT max_nz;
  IT thread_num;
  IT min_ht_size;
  IT num_clusters;
  IT *cluster_sz;

  IT *cluster_nz;                       // the number of flop or non-zero elements of output cluster-matrix
  IT *clusters_offset;                  // offset for cluster_nz

  char *bin_id_cluster;                 // hash table size for cluster[i] will be 2^bin_id_cluster[i]

  IT **local_hash_table_id = nullptr;   // col-ids in the output matrix
  NT **local_hash_table_val = nullptr;  // values in the output matrix

  // temporarily added for debugging; can be removed later
//  IT *ht_sz;                            // size of the local_hash_table_id per thread
//  IT *ht_val_sz;                        // size of the local_hash_table_val per thread
};

template<class IT, class NT>
double BIN_VlengthCluster<IT, NT>::calculate_size_in_gb() {
  size_t size_bytes = 0;

  // Primary arrays
  if (cluster_sz)       size_bytes += num_clusters * sizeof(IT);
  if (cluster_nz)       size_bytes += num_clusters * sizeof(IT);
  if (clusters_offset)  size_bytes += (thread_num + 1) * sizeof(IT);
  if (bin_id_cluster)   size_bytes += num_clusters * sizeof(char);

  // Pointer arrays
  if (local_hash_table_id)  size_bytes += thread_num * sizeof(IT*);
  if (local_hash_table_val) size_bytes += thread_num * sizeof(NT*);

  // Per-thread hash tables (if allocated)
  for (IT i = 0; i < thread_num; ++i) {
    if (local_hash_table_id && local_hash_table_id[i]) {
      size_bytes += min_ht_size * sizeof(IT); // todo: this is a bug
    }
    if (local_hash_table_val && local_hash_table_val[i]) {
      size_bytes += min_ht_size * sizeof(NT); // todo: this is a bug
    }
  }

//  return static_cast<double>(size_bytes) / (1L << 30); // Convert bytes to GB
  return static_cast<double>(size_bytes);
}

/* Count the number of intermediate products per row (= flop / 2) */
template<class IT, class NT>
inline void BIN_VlengthCluster<IT, NT>::set_intprod_num(const IT *arpt, const IT *acol, const IT *brpt) {
#pragma omp parallel
  {
    int64_t each_int_prod = 0;
#pragma omp for
    for (IT i = 0; i < num_clusters; ++i) {                                 // loop over clusters
      IT nz_per_cluster = 0;
      for (IT j = arpt[i]; j < arpt[i + 1]; ++j) {                          // col-ids of i-th cluster
        nz_per_cluster += ((brpt[acol[j] + 1] - brpt[acol[j]]));            // summing up rows of B
      }
      cluster_nz[i] = nz_per_cluster;                                       // cluster_nz is populated by flops
      each_int_prod += ((int64_t) nz_per_cluster * (int64_t) cluster_sz[i]);
    }
#pragma omp atomic
    total_intprod += each_int_prod;
  }
}

/* Get total number of floating operations and average
 * then, use it for assigning clusters to thread as the amount of work is equally distributed
 */
template<class IT, class NT>
inline void BIN_VlengthCluster<IT, NT>::set_clusters_offset() {
  // cluster_nz only counts the sum of rows of B
  // so, prefix sum of intermediate products performed on: (cluster_nz[i] * cluster_sz[i])
  int64_t *ps_cluster_nz = my_malloc<int64_t>(num_clusters + 1);
  scan<int64_t, IT>(cluster_nz, ps_cluster_nz, cluster_sz, num_clusters + 1);

  int64_t average_intprod = (total_intprod + thread_num - 1) / thread_num;

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
  my_free<int64_t>(ps_cluster_nz);
}

/*
 * Prepare hash table for each thread_num
 * once allocate memory space for hash table, the thread reuse it for each row
 * hash table size is determined by the cluster_nz values
 */
template<class IT, class NT>
inline void BIN_VlengthCluster<IT, NT>::create_local_hash_table(const IT cols) {
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    IT ht_size = 0;                     // max cluster_nz among all the clusters
    IT max_cluster_sz = 0;              // max cluster_sz among all the clusters
    IT ht_val_size = 0;

    /* Get max size of hash table */
    for (IT j = clusters_offset[tid]; j < clusters_offset[tid + 1]; ++j) {
      if (ht_size < cluster_nz[j]) ht_size = cluster_nz[j];
      if (max_cluster_sz < cluster_sz[j]) max_cluster_sz = cluster_sz[j];

//      if (ht_val_size < (cluster_nz[j] * cluster_sz[j])) {
//        ht_val_size = (cluster_nz[j] * cluster_sz[j]);
//        max_cluster_sz = cluster_sz[j];
//      }
//      assert(cluster_nz[j] * cluster_sz[j] < INT32_MAX && "ht_val_size goes beyond INT32_MAX limit!");
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

    // todo: can we calculate ht_val_size in a more efficient way???
    // todo: this problem happen for web-edu dataset
//    assert(((int64_t) ht_size * max_cluster_sz) < (int64_t) INT32_MAX && "ht_val_size will go beyond INT32_MAX limit!");
    ht_val_size = ht_size * max_cluster_sz;

    local_hash_table_id[tid] = my_malloc<IT>(ht_size);
    local_hash_table_val[tid] = my_malloc<NT>(ht_val_size);

//    ht_sz[tid] = ht_size;
//    ht_val_sz[tid] = ht_val_size;
  }
}

/*
 * Precompute how many entries each row requires for the hash table
 * the size is 2^bin_id_cluster
 */
template<class IT, class NT>
inline void BIN_VlengthCluster<IT, NT>::set_bin_id(const IT cols, const IT min) {
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
inline void BIN_VlengthCluster<IT, NT>::set_max_bin(const IT *arpt, const IT *acol, const IT *brpt, const IT cols) {
  set_intprod_num(arpt, acol, brpt);
  set_clusters_offset();
  set_bin_id(cols, min_ht_size);
}

/* Reset the size of hash table which each row requires */
template<class IT, class NT>
inline void BIN_VlengthCluster<IT, NT>::set_min_bin(const IT cols) {
  set_bin_id(cols, min_ht_size);
}

#endif  //BIN_VLENGTH_CLUSTER_H