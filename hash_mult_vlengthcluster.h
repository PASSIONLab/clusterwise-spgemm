#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>
#include <algorithm>
#include <fstream>
#include <x86intrin.h>

#include "utility.h"
#include "CSR.h"
#include "CSR_VlengthCluster.h"
#include "BIN_VlengthCluster.h"

/* SpGEMM Specific Parameters */
#define HASH_SCAL 107 // Set disjoint number to hash table size (=2^n)
#define SMALL_THRESHOLD 100

#define MIN_HT_S 8 // minimum hash table size per row in symbolic phase
#define MIN_HT_N 8 // minimum hash table size per row in numeric phase

/*
 * Symbolic phase for HashSpGEMMVLCluster.
 */
template<class IT, class NT>
inline void hash_symbolic_kernel_vlcluster(const IT *arpt, const IT *acol,
                                           const IT *brpt, const IT *bcol,
                                           BIN_VlengthCluster<IT, NT> &bin) {
#pragma omp parallel
  {
    IT tid = omp_get_thread_num();
    IT start_row = bin.clusters_offset[tid];        // start cluster
    IT end_row = bin.clusters_offset[tid + 1];      // end cluster

    IT *check = bin.local_hash_table_id[tid];

    for (IT i = start_row; i < end_row; ++i) {                            // loop over clusters
      IT nz = 0;
      IT bid = bin.bin_id_cluster[i];

      if (bid > 0) {
        IT ht_size = MIN_HT_S << (bid - 1);                               // determine hash table size for i-th row
        for (IT j = 0; j < ht_size; ++j) {                                // initialize hash table
          check[j] = -1;
        }

        for (IT j = arpt[i]; j < arpt[i + 1]; ++j) {                      // loop over the columns of cluster[i]
          IT t_acol = acol[j];
          for (IT k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {          // loop over columns of B (B.row selected by columns of cluster[i])
            IT key = bcol[k];
            IT hash = (key * HASH_SCAL) & (ht_size - 1);
            while (1) {                                                   // Loop for hash probing
              if (check[hash] == key) {                                   // if the key is already inserted, it's ok
                break;
              } else if (check[hash] == -1) {                             // if the key has not been inserted yet, then it's added.
                check[hash] = key;
                nz++;
                break;
              } else {                                                    // linear probing: check next entry
                hash = (hash + 1) & (ht_size - 1);                        // hash = (hash + 1) % ht_size
              }
            }
          }
        }
      }
      bin.cluster_nz[i] = nz;                                             // updating cluster_nz by row[i] nnz; previously it was set to flops of row[i]
    }
  }
}

// Reference function for Symbolic phase of HashSpGEMMVLCluster
template<class IT, class NT>
inline void hash_symbolic_vlcluster(const IT *arpt, const IT *acol, const IT *brpt, const IT *bcol,
                                    IT *crpt, IT *crpt_val, BIN_VlengthCluster<IT, NT> &bin,
                                    const IT nrow, IT *nnzc, IT *nnzv) {
  hash_symbolic_kernel_vlcluster(arpt, acol, brpt, bcol, bin);

  /* Set row/value pointer of matrix C */
  scan<IT>(bin.cluster_nz, crpt, crpt_val, bin.cluster_sz, nrow + 1);
  *nnzc = crpt[nrow];
  *nnzv = crpt_val[nrow];
}

/*
 * Used for sort function.
 * Elements are sorted in ascending order.
 */
// todo: uncomment this
//template<typename IT, typename NT>
//bool sort_less_V1(const pair<IT, NT> &left, const pair<IT, NT> &right) {
//  return left.first < right.first;
//}

/*
 * After calculating on each hash table, sort them in ascending order if necessary, and then store them as output matrix
 * This function is used in hash_numeric* function.
 * the actual indices of colids and values of output matrix are rpt[rowid];
 */
template<bool sortOutput, typename IT, typename NT>
inline void sort_and_store_table2mat_vlcluster(IT *ht_check, NT *ht_value,
                                               IT *colids, NT *values,
                                               IT nz, IT ht_size, IT offset, IT cluster_sz) {
  IT index = 0;
  IT val_idx, ht_idx;
  // Sort elements in ascending order if necessary, and store them as output matrix
  if (sortOutput) {
    // todo: uncomment this
//    vector <pair<IT, IT>> p_vec(nz);                            // <col-id, position-in-hashtable>
//    for (IT j = 0; j < ht_size; ++j) {                          // accumulate non-zero entry from hash table
//      if (ht_check[j] != -1) {
//        p_vec[index++] = make_pair(ht_check[j], j);
//      }
//    }
//
////      assert(index <= nz && "Index goes beyond p_vector limit");
////      assert(index < offset && "Index goes beyond output limit");
//
//    sort(p_vec.begin(), p_vec.end(), sort_less_V1<IT, IT>);     // sort only non-zero elements
//    // store the results
//    for (IT j = 0; j < index; ++j) {
//      colids[j] = p_vec[j].first;
//      val_idx = (j * cluster_sz);
//      ht_idx = (p_vec[j].second * cluster_sz);
//      for (IT l = 0; l < cluster_sz; l += 1) {
//        values[val_idx + l] = ht_value[ht_idx + l];
//      }
//    }
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

// todo: here is the update that I was trying to make to improve the performance of variable length cluster SpGEMM
// todo: I think this thinking is still valied as we observed that single-length cluster computation is not that efficient compared to the row-wise spgemm
// todo: I will try to implement this in the future
///*
// * Numeric phase in HashSpGEMMCluster.
// */
//template<bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
//inline void hash_numeric_vlcluster(const IT *arpt, const IT *arpt_val, const IT *acol, const NT *aval,
//                                   const IT *brpt, const IT *bcol, const NT *bval,
//                                   const IT *crpt, const IT *crpt_val, IT *ccol, NT *cval,
//                                   const BIN_VlengthCluster<IT, NT> &bin,
//                                   const MultiplyOperation multop, const AddOperation addop, IT cnnz,
//                                   const IT *cluster_sz, const NT eps = 0.000001f) {
//#pragma omp parallel
//  {
//    IT tid = omp_get_thread_num();
//    IT start_row = bin.clusters_offset[tid];
//    IT end_row = bin.clusters_offset[tid + 1];
//
//    IT *ht_check = bin.local_hash_table_id[tid];
//    NT *ht_value = bin.local_hash_table_val[tid];
//
//    IT t_acol;
//    NT t_aval, t_val;
//    for (IT i = start_row; i < end_row; ++i) {                            // A.clusters
//// note: BIN class is based on the cluster (not in row order of the original CSR)
//      IT bid = bin.bin_id_cluster[i];
//      if (bid > 0) {
//        IT offset = crpt[i];
//        IT cval_offset = crpt_val[i];
////        IT aval_offset = arpt_val[i];
//        IT aval_offset = arpt_val[i] + ((j - arpt[i]) * cluster_sz[i]);
//        IT ht_size = MIN_HT_N << (bid - 1);
//        for (IT j = 0; j < ht_size; ++j) {
//          ht_check[j] = -1;
//        }
//
//        for (IT j = arpt[i]; j < arpt[i + 1]; ++j) {                      // union of col-ids of cluster A.cluster[i]
//          t_acol = acol[j];
//          for (IT k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {          // B.cols
//            IT key = bcol[k];
//            IT hash = (key * HASH_SCAL) & (ht_size - 1);
//            while (1) {                                                   // Loop for hash probing
//              if (ht_check[hash] == key) {                                // key is already inserted
//                IT htval_offset = (hash * cluster_sz[i]);
//                for (IT l = 0; l <
//                               cluster_sz[i]; l += 1, htval_offset += 1) {               // loop over all the rows of A.cluster[i].col[j]
//                  t_aval = aval[aval_offset + l];   // value from A
//// note: maybe we can use a bitmap to find whether [(j * cluster_sz) + l]-pos is valid
//                  if (fabs(t_aval - 0.0f) >= eps) {                       // avoid flop when (A.value[] == 0.0)
//                    t_val = multop(t_aval, bval[k]);                                    // value for C
//                    ht_value[htval_offset] = addop(t_val, ht_value[htval_offset]);
//                  }
//                }
//                break;
//              } else if (ht_check[hash] == -1) {                          // insert new entry
//                ht_check[hash] = key;
//                for (IT l = 0; l <
//                               cluster_sz[i]; l += 1, htval_offset += 1) {                           // loop over all the rows of A.cluster[i].col[j]
//                  t_aval = aval[aval_offset + l];   // value from A
//                  t_val = multop(t_aval, bval[k]);                                    // value for C
//// ht_value will be automatically initialized by 0.0 if t_val is zero
//// assert((htval_offset + l) < bin.ht_val_sz[tid] && "Trying to access beyond Hashtable size!");
//                  ht_value[htval_offset] = t_val;
//                }
//                break;
//              } else {
//                hash = (hash + 1) & (ht_size - 1);                        // (hash + 1) % ht_size
//              }
//            }
//          }
//        }
//// copy results from ht to the C_csr
//        sort_and_store_table2mat_vlcluster<sortOutput, IT, NT>(ht_check, ht_value,
//                                                               ccol + offset, cval + cval_offset,
//                                                               (crpt[i + 1] - offset), ht_size, (cnnz - offset),
//                                                               cluster_sz[i]);
//      }
//    }
//  }
//}

/*
 * Numeric phase in HashSpGEMMCluster.
 */
template<bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
inline void hash_numeric_vlcluster(const IT *arpt, const IT *arpt_val, const IT *acol, const NT *aval,
                                   const IT *brpt, const IT *bcol, const NT *bval,
                                   const IT *crpt, const IT *crpt_val, IT *ccol, NT *cval,
                                   const BIN_VlengthCluster<IT, NT> &bin,
                                   const MultiplyOperation multop, const AddOperation addop, IT cnnz,
                                   const IT *cluster_sz, const NT eps = 0.000001f) {
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
        IT cval_offset = crpt_val[i];
        IT aval_offset = arpt_val[i];
        IT ht_size = MIN_HT_N << (bid - 1);
        for (IT j = 0; j < ht_size; ++j) {
          ht_check[j] = -1;
        }

        for (IT j = arpt[i]; j < arpt[i + 1]; ++j) {                      // union of col-ids of cluster A.cluster[i]
          t_acol = acol[j];
          for (IT k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {          // B.cols
            IT key = bcol[k];
            IT hash = (key * HASH_SCAL) & (ht_size - 1);
            while (1) {                                                   // Loop for hash probing
              if (ht_check[hash] == key) {                                // key is already inserted
                for (IT l = 0; l < cluster_sz[i]; l += 1) {               // loop over all the rows of A.cluster[i].col[j]
                  t_aval = aval[aval_offset + ((j - arpt[i]) * cluster_sz[i]) + l];   // value from A
                  t_val = multop(t_aval, bval[k]);                                    // value for C
                  // note: maybe we can use a bitmap to find whether [(j * cluster_sz) + l]-pos is valid
                  if (fabs(t_aval - 0.0f) >= eps) {                       // avoid flop when (A.value[] == 0.0)
                    ht_value[(hash * cluster_sz[i]) + l] = addop(t_val, ht_value[(hash * cluster_sz[i]) + l]);
                  }
                }
                break;
              } else if (ht_check[hash] == -1) {                          // insert new entry
                ht_check[hash] = key;
                for (IT l = 0; l < cluster_sz[i]; l += 1) {                           // loop over all the rows of A.cluster[i].col[j]
                  t_aval = aval[aval_offset + ((j - arpt[i]) * cluster_sz[i]) + l];   // value from A
                  t_val = multop(t_aval, bval[k]);                                    // value for C
                  // ht_value will be automatically initialized by 0.0 if t_val is zero
                  // assert(((hash * cluster_sz[i]) + l) < bin.ht_val_sz[tid] && "Trying to access beyond Hashtable size!");
                  ht_value[(hash * cluster_sz[i]) + l] = t_val;
                }
                break;
              } else {
                hash = (hash + 1) & (ht_size - 1);                        // (hash + 1) % ht_size
              }
            }
          }
        }
        // copy results from ht to the C_csr
        sort_and_store_table2mat_vlcluster<sortOutput, IT, NT>(ht_check, ht_value,
                                                               ccol + offset, cval + cval_offset,
                                                               (crpt[i + 1] - offset), ht_size, (cnnz - offset),
                                                               cluster_sz[i]);
      }
    }
  }
}

/*
 * After calculating on each hash table, sort them in ascending order if necessary, and then store them as output matrix
 * This function is used in hash_numeric* function.
 * the actual indices of colids and values of output matrix are rpt[rowid];
 */
template<bool sortOutput, typename IT, typename NT>
inline void sort_and_store_table2mat_vlcluster_V1(IT *ht_check, NT *ht_value,
                                               IT *colids, NT *values,
                                               IT nz, IT ht_size) {
  IT index = 0;
  IT val_idx, ht_idx;
  // Sort elements in ascending order if necessary, and store them as output matrix
  if (sortOutput) {
    // todo: uncomment this
//    vector <pair<IT, IT>> p_vec(nz);                            // <col-id, position-in-hashtable>
//    for (IT j = 0; j < ht_size; ++j) {                          // accumulate non-zero entry from hash table
//      if (ht_check[j] != -1) {
//        p_vec[index++] = make_pair(ht_check[j], j);
//      }
//    }
//
////      assert(index <= nz && "Index goes beyond p_vector limit");
////      assert(index < offset && "Index goes beyond output limit");
//
//    sort(p_vec.begin(), p_vec.end(), sort_less_V1<IT, IT>);     // sort only non-zero elements
//    // store the results
//    for (IT j = 0; j < index; ++j) {
//      colids[j] = p_vec[j].first;
////      val_idx = (j * cluster_sz);
//      ht_idx = p_vec[j].second;
////      for (IT l = 0; l < cluster_sz; l += 1) {
//        values[j] = ht_value[ht_idx];
////      }
//    }
  } else {
    // store the results
    for (IT j = 0; j < ht_size; ++j) {
      if (ht_check[j] != -1) {
        colids[index] = ht_check[j];
//        val_idx = (index * cluster_sz);
//        ht_idx = (j * cluster_sz);
//        for (IT l = 0; l < cluster_sz; l += 1) {
          values[index] = ht_value[j];
//        }
        index++;
      }
    }
  }
}

/*
 * Numeric phase in HashSpGEMMCluster.
 */
template<bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
inline void hash_numeric_vlcluster_V1(const IT *arpt, const IT *arpt_val, const IT *acol, const NT *aval,
                                   const IT *brpt, const IT *bcol, const NT *bval,
                                   const IT *crpt, const IT *crpt_val, IT *ccol, NT *cval,
                                   const BIN_VlengthCluster<IT, NT> &bin,
                                   const MultiplyOperation multop, const AddOperation addop, IT cnnz,
                                   const IT *cluster_sz, const NT eps = 0.000001f) {
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
        IT csz = cluster_sz[i];
        IT offset = crpt[i];
        IT cval_offset = crpt_val[i];
        IT aval_offset = arpt_val[i];
        IT ht_size = MIN_HT_N << (bid - 1);
        for (IT j = 0; j < ht_size; ++j) {
          ht_check[j] = -1;
        }

        IT tmp1, tmp2;
        for (IT j = arpt[i]; j < arpt[i + 1]; ++j) {                      // union of col-ids of cluster A.cluster[i]
          t_acol = acol[j];
          tmp1 = aval_offset + ((j - arpt[i]) * csz);
          for (IT k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {          // B.cols
            IT key = bcol[k];
            IT hash = (key * HASH_SCAL) & (ht_size - 1);
            while (1) {                                                   // Loop for hash probing
              tmp2 = (hash * csz);
              if (ht_check[hash] == key) {                                // key is already inserted
                for (IT l = 0; l < csz; l += 1) {               // loop over all the rows of A.cluster[i].col[j]
                  t_aval = aval[tmp1 + l];   // value from A
                  t_val = multop(t_aval, bval[k]);                                    // value for C
                  // note: maybe we can use a bitmap to find whether [(j * cluster_sz) + l]-pos is valid
                  if (fabs(t_aval - 0.0f) >= eps) {                       // avoid flop when (A.value[] == 0.0)
//                    ht_value[tmp2 + l] = addop(t_val, ht_value[tmp2 + l]);
                    ht_value[tmp2] = addop(t_val, ht_value[tmp2]);
                  }
                  tmp2++;
                }
                break;
              } else if (ht_check[hash] == -1) {                          // insert new entry
                ht_check[hash] = key;
                for (IT l = 0; l < csz; l += 1) {                           // loop over all the rows of A.cluster[i].col[j]
                  t_aval = aval[tmp1 + l];   // value from A
                  t_val = multop(t_aval, bval[k]);                                    // value for C
                  // ht_value will be automatically initialized by 0.0 if t_val is zero
                  // assert(((hash * csz) + l) < bin.ht_val_sz[tid] && "Trying to access beyond Hashtable size!");
                  ht_value[tmp2] = t_val;
                  tmp2++;
                }
                break;
              } else {
                hash = (hash + 1) & (ht_size - 1);                        // (hash + 1) % ht_size
              }
            }
          }
        }
        // copy results from ht to the C_csr
        sort_and_store_table2mat_vlcluster_V1<sortOutput, IT, NT>(ht_check, ht_value,
                                                               ccol + offset, cval + cval_offset,
                                                               (crpt[i + 1] - offset), ht_size);
      }
    }
  }
}


/*
 * Executing HashSpGEMMVLCluster
 * The function starts with initialization of hash table followed by symbolic phase and numeric phase with hash table.
 * Hash table also stores data in clustered format
 */
template<bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void
HashSpGEMMVLCluster(const CSR_VlengthCluster<IT, NT> &a, const CSR<IT, NT> &b, CSR_VlengthCluster<IT, NT> &c,
                    MultiplyOperation multop, AddOperation addop) {
  // initialize bin
//  double start, end, msec;
//  start = omp_get_wtime();
  BIN_VlengthCluster<IT, NT> bin(a.rows, MIN_HT_S, a.cluster_sz);

  c.csr_rows = a.csr_rows;
  c.rows = a.rows;
  c.cols = b.cols;
  c.cluster_sz = my_malloc<IT>(a.rows);
  memcpy(c.cluster_sz, a.cluster_sz, sizeof(IT) * a.rows);

  /* Set max bin */
  bin.set_max_bin(a.rowptr, a.colids, b.rowptr, c.cols);

  /* Create hash table (thread local) */
  bin.create_local_hash_table(c.cols);

  /* Symbolic Phase */
  c.rowptr = my_malloc<IT>(c.rows + 1);
  c.rowptr_val = my_malloc<IT>(c.rows + 1);
//  end = omp_get_wtime();
//  msec = (end - start) * 1000;
//  cout << "BIN initialization time: " << msec << " milli seconds." << endl;
//
//  start = omp_get_wtime();
  hash_symbolic_vlcluster(a.rowptr, a.colids, b.rowptr, b.colids,
                          c.rowptr, c.rowptr_val,
                          bin, c.rows, &(c.nnzc), &(c.nnzv));

  // resetting (reduce) hash table size
  // hash table size is initialized by flops count
  // after the symbolic phase, we can reset it by nnz count
  bin.set_bin_id(c.cols, bin.min_ht_size);

  c.colids = my_malloc<IT>(c.nnzc);
  c.values = my_malloc<NT>(c.nnzv);
//  end = omp_get_wtime();
//  msec = (end - start) * 1000;
//  cout << "Symbolic time: " << msec << " milli seconds." << endl;
//
//  /* Numeric Phase */
//  start = omp_get_wtime();

//  hash_numeric_vlcluster<sortOutput>(a.rowptr, a.rowptr_val, a.colids, a.values,
//                                     b.rowptr, b.colids, b.values,
//                                     c.rowptr, c.rowptr_val, c.colids, c.values,
//                                     bin, multop, addop, c.nnzc, c.cluster_sz);

  hash_numeric_vlcluster_V1<sortOutput>(a.rowptr, a.rowptr_val, a.colids, a.values,
                                     b.rowptr, b.colids, b.values,
                                     c.rowptr, c.rowptr_val, c.colids, c.values,
                                     bin, multop, addop, c.nnzc, c.cluster_sz);
//  end = omp_get_wtime();
//  msec = (end - start) * 1000;
//  cout << "Numeric time: " << msec << " milli seconds." << endl;
}

template<bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void
HashSpGEMMVLClustertoCheckMemoryRequirement(const CSR_VlengthCluster<IT, NT> &a, const CSR<IT, NT> &b, CSR_VlengthCluster<IT, NT> &c,
                    MultiplyOperation multop, AddOperation addop) {
  // initialize bin
  BIN_VlengthCluster<IT, NT> bin(a.rows, MIN_HT_S, a.cluster_sz);

  c.csr_rows = a.csr_rows;
  c.rows = a.rows;
  c.cols = b.cols;
  c.cluster_sz = my_malloc<IT>(a.rows);
  memcpy(c.cluster_sz, a.cluster_sz, sizeof(IT) * a.rows);

  /* Set max bin */
  bin.set_max_bin(a.rowptr, a.colids, b.rowptr, c.cols);

  /* Create hash table (thread local) */
  bin.create_local_hash_table(c.cols);

  /* Symbolic Phase */
  c.rowptr = my_malloc<IT>(c.rows + 1);
  c.rowptr_val = my_malloc<IT>(c.rows + 1);

  hash_symbolic_vlcluster(a.rowptr, a.colids, b.rowptr, b.colids,
                          c.rowptr, c.rowptr_val,
                          bin, c.rows, &(c.nnzc), &(c.nnzv));

  // resetting (reduce) hash table size
  // hash table size is initialized by flops count
  // after the symbolic phase, we can reset it by nnz count
  bin.set_bin_id(c.cols, bin.min_ht_size);

  c.colids = my_malloc<IT>(c.nnzc);
  c.values = my_malloc<NT>(c.nnzv);

  printf("BIN size: %.2f bytes\n", bin.calculate_size_in_gb());
  printf("C size: %.2f bytes\n", c.calculate_size());
}