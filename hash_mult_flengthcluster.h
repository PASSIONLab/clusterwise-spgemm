#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>
#include <algorithm>
#include <fstream>
#include <x86intrin.h>

#include "utility.h"
#include "CSR.h"
#include "CSR_FlengthCluster.h"
#include "BIN_FlengthCluster.h"

/* SpGEMM Specific Parameters */
#define HASH_SCAL 107 // Set disjoint number to hash table size (=2^n)
#define SMALL_THRESHOLD 100

#define MIN_HT_S 8 // minimum hash table size per row in symbolic phase
#define MIN_HT_N 8 // minimum hash table size per row in numeric phase

/*
 * Symbolic phase for HashSpGEMMCluster.
 */
template<class IT, class NT>
inline void hash_symbolic_kernel_cluster(const IT *arpt, const IT *acol,
                                         const IT *brpt, const IT *bcol,
                                         BIN_FlengthCluster<IT, NT> &bin) {
#pragma omp parallel
  {
    IT tid = omp_get_thread_num();
    IT start_row = bin.clusters_offset[tid];      // start cluster
    IT end_row = bin.clusters_offset[tid + 1];    // end cluster

    IT *check = bin.local_hash_table_id[tid];
    IT t_acol, key, hash;

    for (IT i = start_row; i < end_row; ++i) {                      // loop over clusters
      IT nz = 0;
      IT bid = bin.bin_id_cluster[i];

      if (bid > 0) {
        IT ht_size = MIN_HT_S << (bid - 1);                         // determine hash table size for i-th cluster
        for (IT j = 0; j < ht_size; ++j) {                          // initialize hash table
          check[j] = -1;
        }

        for (IT j = arpt[i]; j < arpt[i + 1]; ++j) {                // loop over the columns of cluster[i]
          t_acol = acol[j];
          for (IT k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {    // loop over columns of B (B.row selected by columns of cluster[i])
            key = bcol[k];
            hash = (key * HASH_SCAL) & (ht_size - 1);
            while (1) {                                             // loop for hash probing
              if (check[hash] == key) {                             // if the key is already inserted, it's ok
                break;
              } else if (check[hash] == -1) {                       // if the key has not been inserted yet, then it's added.
                check[hash] = key;
                nz++;
                break;
              } else {                                              // linear probing: check next entry
                hash = (hash + 1) & (ht_size - 1);                  // hash = (hash + 1) % ht_size
              }
            }
          }
        }
      }
      bin.cluster_nz[i] = nz;                                       // updating cluster_nz by row[i] nnz; previously it was set to flops of row[i]
    }
  }
}

// Reference function for Symbolic phase of HashSpGEMMCluster
template<class IT, class NT>
inline void hash_symbolic_cluster(const IT *arpt, const IT *acol, const IT *brpt, const IT *bcol,
                                  IT *crpt, BIN_FlengthCluster<IT, NT> &bin, const IT nrow, IT *nnz) {
  hash_symbolic_kernel_cluster(arpt, acol, brpt, bcol, bin);

  /* Set row pointer of matrix C */
  scan(bin.cluster_nz, crpt, nrow + 1);
  *nnz = crpt[nrow];
}

/*
 * Used for sort function.
 * Elements are sorted in ascending order.
 */
template<typename IT, typename NT>
bool sort_less_V1(const pair<IT, NT> &left, const pair<IT, NT> &right) {
  return left.first < right.first;
}

/*
 * After calculating on each hash table, sort them in ascending order if necessary, and then store them as output matrix
 * This function is used in hash_numeric* function.
 * the actual indices of colids and values of output matrix are rpt[rowid];
 */
template<bool sortOutput, typename IT, typename NT>
inline void sort_and_store_table2mat_cluster(IT *ht_check, NT *ht_value,
                                             IT *colids, NT *values,
                                             IT nz, IT ht_size, IT offset, IT cluster_sz,
                                             const IT cluster_id, const IT csr_rows) {
  IT index = 0;
  IT val_idx, ht_idx;
  // Sort elements in ascending order if necessary, and store them as output matrix
  if (sortOutput) {
    vector <pair<IT, IT>> p_vec(nz);                              // <col-id, position-in-hashtable>
    for (IT j = 0; j < ht_size; ++j) {                            // accumulate non-zero entry from hash table
      if (ht_check[j] != -1) {
        p_vec[index++] = make_pair(ht_check[j], j);
      }
    }

//    assert(index <= nz && "Index goes beyond p_vector limit");
//    assert(index < offset && "Index goes beyond output limit");

    sort(p_vec.begin(), p_vec.end(), sort_less_V1<IT, IT>);       // sort only non-zero elements

    // store the results
    for (IT j = 0; j < index; ++j) {
      colids[j] = p_vec[j].first;
      val_idx = (j * cluster_sz);
      ht_idx = (p_vec[j].second * cluster_sz);
      for (IT l = 0; l < cluster_sz; l += 1) {
        values[val_idx + l] = ht_value[ht_idx + l];
      }
    }
  } else {
    // store the results
    for (IT j = 0; j < ht_size; ++j) {
      if (ht_check[j] != -1) {
        colids[index] = ht_check[j];
        val_idx = (index * cluster_sz);
        ht_idx = (j * cluster_sz);
        for (IT l = 0; l < cluster_sz; l += 1) {
          values[val_idx + l] = ht_value[ht_idx + l];
        }
        index++;
      }
    }
  }
}

/*
 * Numeric phase in HashSpGEMMCluster.
 */
template<bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
inline void hash_numeric_cluster(const IT *arpt, const IT *acol, const NT *aval,
                                 const IT *brpt, const IT *bcol, const NT *bval,
                                 const IT *crpt, IT *ccol, NT *cval, const BIN_FlengthCluster<IT, NT> &bin,
                                 const MultiplyOperation multop, const AddOperation addop, IT cnnz,
                                 const int csr_rows, const IT cluster_sz, const NT eps = 0.000001f) {
#pragma omp parallel
  {
    IT tid = omp_get_thread_num();
    IT start_row = bin.clusters_offset[tid];
    IT end_row = bin.clusters_offset[tid + 1];

    IT *ht_check = bin.local_hash_table_id[tid];
    NT *ht_value = bin.local_hash_table_val[tid];

    IT t_acol;
    NT t_aval, t_val;
    for (IT i = start_row; i < end_row; ++i) {                            // A.clusters
      // note: BIN class is based on the cluster (not in row order of the original CSR)
      IT bid = bin.bin_id_cluster[i];
      if (bid > 0) {
        IT offset = crpt[i];
        IT ht_size = MIN_HT_N << (bid - 1);
        for (IT j = 0; j < ht_size; ++j) {
          ht_check[j] = -1;
        }

        for (IT j = arpt[i]; j < arpt[i + 1]; ++j) {                      // union of col-ids of cluster A.cluster[i]
          t_acol = acol[j];
          for (IT k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {          // B.cols
            IT key = bcol[k];
            IT hash = (key * HASH_SCAL) & (ht_size - 1);
            while (1) {                                                   // loop for hash probing
              if (ht_check[hash] == key) {                                // key is already inserted
                for (IT l = 0; l < cluster_sz; l += 1) {                  // loop over all the rows of A.cluster[i].col[j]
                  t_aval = aval[(j * cluster_sz) + l];                    // value from A
                  // note: maybe we can use a bitmap to find whether [(j * cluster_sz) + l]-pos is valid
                  if (fabs(t_aval - 0.0f) >= eps) {                       // avoid flop when (A.value[] == 0.0)
                    t_val = multop(t_aval, bval[k]);                        // value for C
                    ht_value[(hash * cluster_sz) + l] = addop(t_val, ht_value[(hash * cluster_sz) + l]);
                  }
                }
                break;
              } else if (ht_check[hash] == -1) {                          // insert new entry
                ht_check[hash] = key;
                for (IT l = 0; l < cluster_sz; l += 1) {                  // loop over all the rows of A.cluster[i].col[j]
                  t_aval = aval[(j * cluster_sz) + l];                    // value from A
                  t_val = multop(t_aval, bval[k]);                        // value for C
                  // ht_value will be automatically initialized by 0.0 if t_val is zero
                  ht_value[(hash * cluster_sz) + l] = t_val;
                }
                break;
              } else {
                hash = (hash + 1) & (ht_size - 1);                        // (hash + 1) % ht_size
              }
            }
          }
        }
        // copy results from ht to the C_csr
        sort_and_store_table2mat_cluster<sortOutput, IT, NT>(ht_check, ht_value,
                                                             ccol + offset, cval + (offset * cluster_sz),
                                                             (crpt[i + 1] - offset), ht_size, cnnz - offset,
                                                             cluster_sz, i, csr_rows);
      }
    }
  }
}


/*
 * Executing HashSpGEMMCluster
 * The function starts with initialization of hash table followed by symbolic phase and numeric phase with hash table.
 * Hash table also stores data in clustered format
 */
template<bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void HashSpGEMMCluster(const CSR_FlengthCluster<IT, NT> &a, const CSR<IT, NT> &b, CSR_FlengthCluster<IT, NT> &c,
                       MultiplyOperation multop, AddOperation addop) {
  // initialize bin
  BIN_FlengthCluster<IT, NT> bin(a.rows, MIN_HT_S, a.cluster_sz);

  c.csr_rows = a.csr_rows;
  c.rows = a.rows;
  c.cols = b.cols;
  c.cluster_sz = a.cluster_sz;

  /* Set max bin */
  bin.set_max_bin(a.rowptr, a.colids, b.rowptr, c.cols);

  /* Create hash table (thread local) */
  bin.create_local_hash_table(c.cols);

  /* Symbolic Phase */
  c.rowptr = my_malloc<IT>(c.rows + 1);
  hash_symbolic_cluster(a.rowptr, a.colids, b.rowptr, b.colids,
                        c.rowptr, bin, c.rows, &(c.nnzc));

  // resetting (reduce) hash table size
  // hash table size is initialized by flops count
  // after the symbolic phase, we can reset it by nnz count
  bin.set_bin_id(c.cols, bin.min_ht_size);

  c.colids = my_malloc<IT>(c.nnzc);
  c.values = my_malloc<NT>(c.nnzc * c.cluster_sz);

  /* Numeric Phase */
  hash_numeric_cluster<sortOutput>(a.rowptr, a.colids, a.values,
                                   b.rowptr, b.colids, b.values,
                                   c.rowptr, c.colids, c.values,
                                   bin, multop, addop, c.nnzc,
                                   c.csr_rows, c.cluster_sz);
}

/*
 * Executing HashSpGEMMCluster
 * The function starts with initialization of hash table followed by symbolic phase and numeric phase with hash table.
 * Hash table also stores data in clustered format
 */
template<bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void HashSpGEMMClustertoCheckMemoryRequirement(const CSR_FlengthCluster<IT, NT> &a, const CSR<IT, NT> &b, CSR_FlengthCluster<IT, NT> &c,
                       MultiplyOperation multop, AddOperation addop) {
  // initialize bin
  BIN_FlengthCluster<IT, NT> bin(a.rows, MIN_HT_S, a.cluster_sz);

  c.csr_rows = a.csr_rows;
  c.rows = a.rows;
  c.cols = b.cols;
  c.cluster_sz = a.cluster_sz;

  /* Set max bin */
  bin.set_max_bin(a.rowptr, a.colids, b.rowptr, c.cols);

  /* Create hash table (thread local) */
  bin.create_local_hash_table(c.cols);

  /* Symbolic Phase */
  c.rowptr = my_malloc<IT>(c.rows + 1);
  hash_symbolic_cluster(a.rowptr, a.colids, b.rowptr, b.colids,
                        c.rowptr, bin, c.rows, &(c.nnzc));

  // resetting (reduce) hash table size
  // hash table size is initialized by flops count
  // after the symbolic phase, we can reset it by nnz count
  bin.set_bin_id(c.cols, bin.min_ht_size);

  c.colids = my_malloc<IT>(c.nnzc);
  c.values = my_malloc<NT>(c.nnzc * c.cluster_sz);

  printf("BIN size: %.2f bytes\n", bin.calculate_size_in_gb());
  printf("C size: %.2f bytes\n", c.calculate_size());
}