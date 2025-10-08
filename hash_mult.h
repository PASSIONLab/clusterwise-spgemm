#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>
#include <algorithm>
#include <fstream>

#include <x86intrin.h>

#include "utility.h"
#include "CSR.h"
#include "BIN.h"

#define VECTORIZE

/* SpGEMM Specific Parameters */
#define HASH_SCAL 107 // Set disjoint number to hash table size (=2^n)
#define SMALL_THRESHOLD 100

#define MIN_HT_S 8 // minimum hash table size per row in symbolic phase
#define MIN_HT_N 8 // minimum hash table size per row in numeric phase
#define VEC_LENGTH 8
#define VEC_LENGTH_BIT 3
#define VEC_LENGTH_LONG 4
#define VEC_LENGTH_LONG_BIT 2

#ifdef PRINT_METADATA_TO_FILE
/*
 * Symbolic phase to Calculate NNZ Entries in per-row of SpGEMM output.
 */
template <class IT, class NT>
inline void symbolic_maxnnz_kernel(const IT *arpt, const IT *acol, const IT *brpt, const IT *bcol, BIN<IT, NT> &bin)
{
  IT *row_nz_mx = my_malloc<IT>(bin.num_rows);
#pragma omp parallel
  {
    IT tid = omp_get_thread_num();
    IT start_row = bin.rows_offset[tid];
    IT end_row = bin.rows_offset[tid + 1];

    IT *check = bin.local_hash_table_id[tid];

    // rows of A
    for (IT i = start_row; i < end_row; ++i) {
      IT nz = 0;

      // columns of A
      for (IT j = arpt[i]; j < arpt[i + 1]; ++j) {
        IT t_acol = acol[j];
        // nnz columns in B for row t_acol
        nz += (brpt[t_acol + 1] - brpt[t_acol]);
      }

      row_nz_mx[i] = nz;
    }
  }

//  string filename = "/home/aislam6/passionlab/nnz_frequency_predict.txt";
//  std::fstream file(filename, std::ios::out);
//  for (IT i = 0; i < bin.num_rows; ++i) {
//    file << i << " " << row_nz_mx[i] << endl;
//  }
//  file.close();

  my_free<IT>(row_nz_mx);
}
#endif

/*
 * Symbolic phase for Hash SpGEMM.
 */
template <class IT, class NT>
inline void hash_symbolic_kernel(const IT *arpt, const IT *acol, const IT *brpt, const IT *bcol,
                                 BIN<IT, NT> &bin, const string& out_nnz_freq_outfile, bool print_to_file = false) {
#pragma omp parallel
    {
        IT tid = omp_get_thread_num();
        IT start_row = bin.rows_offset[tid];
        IT end_row = bin.rows_offset[tid + 1];
        
        IT *check = bin.local_hash_table_id[tid];
        
        for (IT i = start_row; i < end_row; ++i) {
            IT nz = 0;
            IT bid = bin.bin_id[i];
            
            if (bid > 0) {
                IT ht_size = MIN_HT_S << (bid - 1); // determine hash table size for i-th row
                for (IT j = 0; j < ht_size; ++j) { // initialize hash table
                    check[j] = -1;
                }

                for (IT j = arpt[i]; j < arpt[i + 1]; ++j) {
                    IT t_acol = acol[j];
                    for (IT k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        IT key = bcol[k];
                        IT hash = (key * HASH_SCAL) & (ht_size - 1);
                        while (1) { // Loop for hash probing
                            if (check[hash] == key) { // if the key is already inserted, it's ok
                                break;
                            }
                            else if (check[hash] == -1) { // if the key has not been inserted yet, then it's added.
                                check[hash] = key;
                                nz++;
                                break;
                            }
                            else { // linear probing: check next entry
                                hash = (hash + 1) & (ht_size - 1); //hash = (hash + 1) % ht_size
                            }
                        }
                    }
                }
            }
            bin.row_nz[i] = nz;
        }
    }

#ifdef PRINT_METADATA_TO_FILE
  if(print_to_file) {
    std::fstream file(out_nnz_freq_outfile, std::ios::out);
    for (IT i = 0; i < bin.num_rows; ++i) {
      file << i << " " << bin.row_nz[i] << endl;
    }
    file.close();
  }
#endif
}

/*
 * Symbolic phase for Hash Vector SpGEMM
 * This function is optimized for 32-bit integer with AVX2.
 */
template <class NT>
inline void hash_symbolic_vec_kernel(const int *arpt, const int *acol, const int *brpt, const int *bcol, BIN<int, NT> &bin)
{
#ifdef VECTORIZE
    const __m256i init_m = _mm256_set1_epi32(-1);
    const __m256i true_m = _mm256_set1_epi32(0xffffffff);
#endif
    
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int start_row = bin.rows_offset[tid];
        int end_row = bin.rows_offset[tid + 1];
        
        int *check = bin.local_hash_table_id[tid];
        
        for (int i = start_row; i < end_row; ++i) {
#ifdef VECTORIZE
            __m256i key_m, check_m;
            __m256i mask_m;
            int mask;
#endif        
            int nz = 0;
            int bid = bin.bin_id[i];
            
            if (bid > 0) {
                int table_size = MIN_HT_S << (bid - 1); // the number of entries per table
                int ht_size = table_size >> VEC_LENGTH_BIT; // the number of chunks (1 chunk = VEC_LENGTH elments)
                for (int j = 0; j < table_size; ++j) {
                    check[j] = -1; // initialize hash table
                }

                for (int j = arpt[i]; j < arpt[i + 1]; ++j) {
                    int t_acol = acol[j];
                    for (int k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        int key = bcol[k];
                        int hash = (key * HASH_SCAL) & (ht_size - 1);
#ifdef VECTORIZE
                        key_m = _mm256_set1_epi32(key);
#endif
                        while (1) { // Loop for hash probing
                            // check whether the key is in hash table.
#ifdef VECTORIZE
                            check_m = _mm256_maskload_epi32(check + (hash << VEC_LENGTH_BIT), true_m);
                            mask_m = _mm256_cmpeq_epi32(key_m, check_m);
                            mask = _mm256_movemask_epi8(mask_m);
                            if (mask != 0) {
                                break;
                            }
#else
                            bool flag = false;
#pragma simd
                            for (int l = 0; l < VEC_LENGTH; ++l) {
                                if (check[(hash << VEC_LENGTH_BIT) + l] == key) {
                                    flag = true;
                                }
                            }
                            if (flag) {
                                break;
                            }
#endif
                            else {
                                // If the entry with same key cannot be found, check whether the chunk is filled or not
                                int cur_nz;
#ifdef VECTORIZE
                                mask_m = _mm256_cmpeq_epi32(check_m, init_m);
                                mask = _mm256_movemask_epi8(mask_m);
                                cur_nz = (32 - _popcnt32(mask)) >> 2;
#else
                                cur_nz = VEC_LENGTH;
#pragma simd
                                for (int l = VEC_LENGTH - 1; l >= 0; --l) {
                                    if (check[(hash << VEC_LENGTH_BIT) + l] == -1) {
                                        cur_nz = l;
                                    }
                                }
#endif
                                if (cur_nz < VEC_LENGTH) { //if it is not filled, push the entry to the table
                                    check[(hash << VEC_LENGTH_BIT) + cur_nz] = key;
                                    nz++;
                                    break;
                                }
                                else { // if is filled, check next chunk (linear probing)
                                    hash = (hash + 1) & (ht_size - 1);
                                }
                            }
                        }
                    }
                }
            }
            bin.row_nz[i] = nz;
        }
    }
}

template <class NT>
inline void hash_symbolic_vec_kernel(const long long int *arpt, const long long int *acol, const long long int *brpt, const long long int *bcol, BIN<long long int, NT> &bin)
{
#ifdef VECTORIZE
    const __m256i init_m = _mm256_set1_epi64x(-1);
    const __m256i true_m = _mm256_set1_epi64x(0xffffffffffffffff);
#endif
    
#pragma omp parallel
    {
        long long int tid = omp_get_thread_num();
        long long int start_row = bin.rows_offset[tid];
        long long int end_row = bin.rows_offset[tid + 1];
        
        long long int *check = bin.local_hash_table_id[tid];
        
        for (long long int i = start_row; i < end_row; ++i) {
#ifdef VECTORIZE
            __m256i key_m, check_m;
            __m256i mask_m;
            int mask;
#endif        
            long long int nz = 0;
            long long int bid = bin.bin_id[i];
            
            if (bid > 0) {
                long long int table_size = MIN_HT_S << (bid - 1);
                long long int ht_size = table_size >> VEC_LENGTH_LONG_BIT;
                for (long long int j = 0; j < table_size; ++j) {
                    check[j] = -1;
                }
                
                for (long long int j = arpt[i]; j < arpt[i + 1]; ++j) {
                    long long int t_acol = acol[j];
                    for (long long int k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        long long int key = bcol[k];
                        long long int hash = (key * HASH_SCAL) & (ht_size - 1);
#ifdef VECTORIZE
                        key_m = _mm256_set1_epi64x(key);
#endif
                        while (1) {
#ifdef VECTORIZE
                            check_m = _mm256_maskload_epi64(check + (hash << VEC_LENGTH_LONG_BIT), true_m);
                            mask_m = _mm256_cmpeq_epi64(key_m, check_m);
                            mask = _mm256_movemask_epi8(mask_m);
                            if (mask != 0) {
                                break;
                            }
#else
                            bool flag = false;
#pragma simd
                            for (int l = 0; l < VEC_LENGTH_LONG; ++l) {
                                if (check[(hash << VEC_LENGTH_LONG_BIT) + l] == key) {
                                    flag = true;
                                }
                            }
                            if (flag) {
                                break;
                            }
#endif
                            else {
                                long long int cur_nz;
#ifdef VECTORIZE
                                mask_m = _mm256_cmpeq_epi64(check_m, init_m);
                                mask = _mm256_movemask_epi8(mask_m);
                                cur_nz = (32 - _popcnt32(mask)) >> 3;
#else
                                cur_nz = VEC_LENGTH_LONG;
#pragma simd
                                for (int l = VEC_LENGTH_LONG - 1; l >= 0; --l) {
                                    if (check[(hash << VEC_LENGTH_LONG_BIT) + l] == -1) {
                                        cur_nz = l;
                                    }
                                }
#endif
                                if (cur_nz < VEC_LENGTH_LONG) {
                                    check[(hash << VEC_LENGTH_LONG_BIT) + cur_nz] = key;
                                    nz++;
                                    break;
                                }
                                else {
                                    hash = (hash + 1) & (ht_size - 1);
                                }
                            }
                        }
                    }
                }
            }
            bin.row_nz[i] = nz;
        }
    }
}

// Reference function for Symbolic phase of Hash SpGEMM
template <bool vectorProbing, class IT, class NT>
inline void hash_symbolic(const IT *arpt, const IT *acol, const IT *brpt, const IT *bcol, IT *crpt, BIN<IT, NT> &bin, const IT nrow, IT *nnz, string out_nnz_freq_outfile, bool first_pass = false)
{
//    if (vectorProbing) {
//        hash_symbolic_vec_kernel(arpt, acol, brpt, bcol, bin);
//    }
//    else {
        hash_symbolic_kernel(arpt, acol, brpt, bcol, bin, out_nnz_freq_outfile, first_pass);
#ifdef PRINT_METADATA_TO_FILE
//        if(first_pass) symbolic_maxnnz_kernel(arpt, acol, brpt, bcol, bin);
#endif
//    }
    
    /* Set row pointer of matrix C */
//    cout << "bin.row_nz[nrow-1]: " << bin.row_nz[nrow - 1] << ", bin.row_nz[nrow]: " << bin.row_nz[nrow] << endl;
    scan(bin.row_nz, crpt, nrow + 1);
//    cout << "crpt[nrow]: " << crpt[nrow] << endl;
    *nnz = crpt[nrow];
}

// Reference function for Symbolic phase of Hash SpGEMM
template <class IT, class NT>
inline void hash_symbolic_topK(const IT *arpt, const IT *acol,
                               const IT *brpt, const IT *bcol, IT *crpt,
                               BIN<IT, NT> &bin, const IT nrow, IT *nnz, IT topK,
                               string out_nnz_freq_outfile, bool first_pass = false)
{
  hash_symbolic_kernel(arpt, acol, brpt, bcol, bin, out_nnz_freq_outfile, first_pass);

  /* Set row pointer of matrix C */
//  scan(topK, crpt, nrow + 1);
//  *nnz = nrow * topK;
  scan(bin.row_nz, crpt, nrow + 1);
  *nnz = crpt[nrow];
}

/*
 * Used for sort function.
 * Elements are sorted in ascending order.
 */
template <typename IT, typename NT>
bool sort_less(const pair<IT, NT> &left,const pair<IT, NT> &right)
{
    return left.first < right.first;
}

/*
 * Numeric phase in Hash SpGEMM.
 */
template <typename IT>
inline void sanity_check(const IT *crpt, IT cnnz, IT num_rows) {
//  for(IT i=1; i<num_rows+1; i+=1) {
//    if(crpt[i] < 0) {
//      cout << i << " " << crpt[i] << endl;
//      break;
//    }
//  }
//  cout << "cnnz: " << cnnz << endl;
  assert(cnnz > 0 && "NNZ should be a positive number!");
  for(IT i=1; i<num_rows; i+=1) {
    assert(crpt[i] >= crpt[i - 1] && "C_ptr is wrongly calculated!");
  }
  assert(crpt[num_rows - 1] < cnnz && "C_ptr should not go beyond C_nnz");
}

/*
 * After calculating on each hash table, sort them in ascending order if necessary, and then store them as output matrix
 * This function is used in hash_numeric* function.
 * the actual indices of colids and values of output matrix are rpt[rowid];
 */
//sort_and_store_table2mat<sortOutput, IT, NT>(ht_check, ht_value, ccol + offset, cval + offset, crpt[i + 1] - offset, ht_size, cnnz - offset);
template <bool sortOutput, typename IT, typename NT>
inline void sort_and_store_table2mat(IT *ht_check, NT *ht_value, IT *colids, NT * values, IT nz, IT ht_size, IT offset, IT row_id = -1)
{
//    assert(nz > 0 && "NZ can not be negative!");
    IT index = 0;
    // Sort elements in ascending order if necessary, and store them as output matrix
    if (sortOutput) {
        vector<pair<IT, NT>> p_vec(nz);
        for (IT j = 0; j < ht_size; ++j) { // accumulate non-zero entry from hash table
            if (ht_check[j] != -1) {
                p_vec[index++] = make_pair(ht_check[j], ht_value[j]);
            }
        }
//      assert(index <= nz && "Index goes beyond p_vector limit");
//      assert(index < offset && "Index goes beyond output limit");
        sort(p_vec.begin(), p_vec.end(), sort_less<IT, NT>); // sort only non-zero elements
        for (IT j = 0; j < index; ++j) { // store the results
//          IT first = p_vec[j].first;
//          NT second = p_vec[j].second;
            colids[j] = p_vec[j].first;
            values[j] = p_vec[j].second;
        }
    }
    else {
        for (IT j = 0; j < ht_size; ++j) {
            if (ht_check[j] != -1) {
//              if(row_id == 0) {
//                cout << "j: " << j << ", index: " << index << endl;
//              }
                colids[index] = ht_check[j];
                values[index] = ht_value[j];
                index++;
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
inline void
sort_and_store_table2mat_topK(IT *ht_check, NT *ht_value, IT *colids, NT *values, IT nz, IT ht_size, IT topK) {
//    assert(nz > 0 && "NZ can not be negative!");
  IT index = 0;
  if (sortOutput) {
    // Sort elements in ascending order if necessary, and store them as output matrix
    vector <pair<NT, IT>> p_vec(nz);
    for (IT j = 0; j < ht_size; ++j) { // accumulate non-zero entry from hash table
      if (ht_check[j] != -1) {
        p_vec[index++] = make_pair(ht_value[j], ht_check[j]);
      }
    }
//      assert(index <= nz && "Index goes beyond p_vector limit");
//      assert(index < offset && "Index goes beyond output limit");
    sort(p_vec.begin(), p_vec.end(), sort_large<IT, NT>); // sort only non-zero elements
//  index = min(index, topK);
    for (IT j = 0; j < index; ++j) { // store the results
      colids[j] = p_vec[j].second;
      values[j] = p_vec[j].first;
    }
  }
  else {
    for (IT j = 0; j < ht_size; ++j) {
      if (ht_check[j] != -1) {
        colids[index] = ht_check[j];
        values[index] = ht_value[j];
        index++;
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
inline void
sort_and_store_table2mat_topK_jaccard(IT *ht_check, NT *ht_value, IT *colids, NT *values, IT nz, IT ht_size, IT topK,
                                      const IT *arpt, IT a_row_id) {
//    assert(nz > 0 && "NZ can not be negative!");
  IT index = 0;
  if (sortOutput) {
    // Sort elements in ascending order if necessary, and store them as output matrix
    vector <pair<NT, IT>> p_vec(nz);
    for (IT j = 0; j < ht_size; ++j) { // accumulate non-zero entry from hash table
      if (ht_check[j] != -1) {
        IT u = (arpt[a_row_id + 1] - arpt[a_row_id]) + (arpt[ht_check[j] + 1] - arpt[ht_check[j]]) - ht_value[j];
        NT jacc = (u == 0) ? 0.0 : 1.0 * ht_value[j] / u;
        p_vec[index++] = make_pair(jacc, ht_check[j]);
//      p_vec[index++] = make_pair(ht_value[j], ht_check[j]);
      }
    }
//      assert(index <= nz && "Index goes beyond p_vector limit");
//      assert(index < offset && "Index goes beyond output limit");
    sort(p_vec.begin(), p_vec.end(), sort_large<IT, NT>); // sort only non-zero elements
//    index = min(index, topK);
    for (IT j = 0; j < index; ++j) { // store the results
      colids[j] = p_vec[j].second;
      values[j] = p_vec[j].first;
    }
  }
  else {
    IT u;
    NT jacc;
    for (IT j = 0; j < ht_size; ++j) {
      if (ht_check[j] != -1) {
        u = (arpt[a_row_id + 1] - arpt[a_row_id]) + (arpt[ht_check[j] + 1] - arpt[ht_check[j]]) - ht_value[j];
        jacc = (u == 0) ? 0.0 : 1.0 * ht_value[j] / u;

        colids[index] = ht_check[j];
        values[index] = jacc;
        index++;
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
inline void
sort_and_store_table2mat_topK_cf(IT *ht_check, NT *ht_value, IT *colids, NT *values, IT nz, IT ht_size, IT topK,
                                      const IT *arpt, IT a_row_id) {
//    assert(nz > 0 && "NZ can not be negative!");
  IT index = 0;
  if (sortOutput) {
    // Sort elements in ascending order if necessary, and store them as output matrix
    vector <pair<NT, IT>> p_vec(nz);
    for (IT j = 0; j < ht_size; ++j) { // accumulate non-zero entry from hash table
      if (ht_check[j] != -1) {
        IT u = (arpt[a_row_id + 1] - arpt[a_row_id]) + (arpt[ht_check[j] + 1] - arpt[ht_check[j]]) - ht_value[j];
        NT jacc = (u == 0) ? 0.0 : 1.0 * ht_value[j] / u;
        NT annz = ((arpt[a_row_id + 1] - arpt[a_row_id]) + (arpt[ht_check[j] + 1] - arpt[ht_check[j]])) / 2.0;
        NT cf = u / annz / jacc;
        p_vec[index++] = make_pair(cf, ht_check[j]);
//      p_vec[index++] = make_pair(ht_value[j], ht_check[j]);
      }
    }
//      assert(index <= nz && "Index goes beyond p_vector limit");
//      assert(index < offset && "Index goes beyond output limit");
    sort(p_vec.begin(), p_vec.end(), sort_large<IT, NT>); // sort only non-zero elements
//  index = min(index, topK);
    for (IT j = 0; j < index; ++j) { // store the results
      colids[j] = p_vec[j].second;
      values[j] = p_vec[j].first;
    }
  }
  else {
    IT u;
    NT jacc, annz, cf;
    for (IT j = 0; j < ht_size; ++j) {
      if (ht_check[j] != -1) {
        u = (arpt[a_row_id + 1] - arpt[a_row_id]) + (arpt[ht_check[j] + 1] - arpt[ht_check[j]]) - ht_value[j];
        jacc = (u == 0) ? 0.0 : 1.0 * ht_value[j] / u;
        annz = ((arpt[a_row_id + 1] - arpt[a_row_id]) + (arpt[ht_check[j] + 1] - arpt[ht_check[j]])) / 2.0;
        cf = u / annz / jacc;

        colids[index] = ht_check[j];
        values[index] = cf;
        index++;
      }
    }
  }
}

/*
 * Numeric phase in Hash SpGEMM.
 */
template <bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
inline void hash_numeric(const IT *arpt, const IT *acol, const NT *aval, const IT *brpt, const IT *bcol, const NT *bval,
                         const IT *crpt, IT *ccol, NT *cval, const BIN<IT, NT> &bin,
                         const MultiplyOperation multop, const AddOperation addop, IT cnnz)
{
//  int numThreads;
//#pragma omp parallel
//  {
//    numThreads = omp_get_num_threads();
//  }
//  long long int* flops_per_thread = my_malloc<long long int>(numThreads);
#pragma omp parallel
    {
        IT tid = omp_get_thread_num();
        IT start_row = bin.rows_offset[tid];
        IT end_row = bin.rows_offset[tid + 1];

        IT *ht_check = bin.local_hash_table_id[tid];
        NT *ht_value = bin.local_hash_table_val[tid];
//        long long int flops = 0;

//        row-id: 1, 2
//        [1]: 1, 2, 3
//        [2]: 1, 2, 4
//        union-of-col-ids: 1, 2, 3, 4

//        cluster_size = 2
//        // prepare this for every clusters
//        vector < map< col-id, vector<NT>> > col_map
//        for(int r=0; r<num_rows; r+=cluster_sz) {
//          cluster_id = r / 2;
//          for(i=r; i< r+cluster_sz; i += 1) {
//            for(IT j = arpt[i]; j < arpt[i + 1]; ++j) {
//              IT t_acol = acol[j];
//              NT t_aval = aval[j];
//              if (col_map[cluster_id].find(t_acol) == col_map[cluster_id].end()) {
//                col_map[cluster_id][t_acol] = vector<NT>(cluster_size, 0.0);
//              }
//              col_map[cluster_id][t_acol][i-r] = t_aval;
//            }
//          }
//        }
//        col-map
//        col 1: <1, val1>, <2, val2>
//        col 2: <1, val1>, <2, val2>
//        col 3: <1, val1>
//        col 4: <2, val2>
//
//        col 1: [val1, val2]
//        col 2: [val1, val2]
//        col 3: [val1, 0]
//        col 4: [0, val2]

    // create a new BIN class that can process clusters instead of rows
    // for each cluster, we will need to calculate flops, nnz for the union of col-ids
    // allocate memory for values: nnz * cluster_size
//        hash-table:
//        col-ids: [1, 2, 3, 4]
//        values:  [[val1, val2], [val1, val2], [val1, 0], [0, val2]]
//
//        A_val: vector< pair<row-id, value> >

        // clusters
//      for (IT c = start_cluster; c < end_cluster; ++c) {
        for (IT i = start_row; i < end_row; ++i) {
            // todo: prepare a new BIN class based on the cluster
            IT bid = bin.bin_id[i];
            if (bid > 0) {
                IT offset = crpt[i];
                IT ht_size = MIN_HT_N << (bid - 1);
                for (IT j = 0; j < ht_size; ++j) {
                    ht_check[j] = -1;
                }
                // union of col-ids of cluster
                for (IT j = arpt[i]; j < arpt[i + 1]; ++j) { // A.cols
                    IT t_acol = acol[j];
                    NT t_aval = aval[j];
                    // B[i]
                    for (IT k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        NT t_val = multop(t_aval, bval[k]);
//                        for(pair<IT, NT> t: col_map[i])
//                        flops +=1;
                        IT key = bcol[k];
                        IT hash = (key * HASH_SCAL) & (ht_size - 1);
                        while (1) { // Loop for hash probing
                            if (ht_check[hash] == key) { // key is already inserted
                                ht_value[hash] = addop(t_val, ht_value[hash]);
//                                flops += 1;
                                break;
                            }
                            else if (ht_check[hash] == -1) { // insert new entry
                                ht_check[hash] = key;
                                ht_value[hash] = t_val;
                                break;
                            }
                            else {
                                hash = (hash + 1) & (ht_size - 1); // (hash + 1) % ht_size
                            }
                        }
                    }
                }
//                if(crpt[i + 1] - offset == 0) {
//                  cout << "i: " << i << ", bid: " << bid << endl;
//                }
//                if(i == 0) {
//                  cout << "i: " << i << ", bid: " << bid << ", offset: " << offset << ", next-offset: " << crpt[i + 1] << endl;
//                }
                sort_and_store_table2mat<sortOutput, IT, NT>(ht_check, ht_value,
                                                             ccol + offset, cval + offset,
                                                             crpt[i + 1] - offset, ht_size, cnnz - offset, i);
            }
        }
//        flops_per_thread[tid] = flops;
    }

//    sort(flops_per_thread, flops_per_thread+numThreads);
//    cout << "min-flops: " << flops_per_thread[0] << " max-flops: " << flops_per_thread[numThreads - 1] << endl;
}

/*
 * Numeric phase in Hash SpGEMM.
 */
template<bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
inline void
hash_numeric_topk(const IT *arpt, const IT *acol, const NT *aval, const IT *brpt, const IT *bcol, const NT *bval,
                  const IT *crpt, IT *ccol, NT *cval, const BIN<IT, NT> &bin,
                  const MultiplyOperation multop, const AddOperation addop, IT topK) {
#pragma omp parallel
  {
    IT tid = omp_get_thread_num();
    IT start_row = bin.rows_offset[tid];
    IT end_row = bin.rows_offset[tid + 1];

    IT *ht_check = bin.local_hash_table_id[tid];
    NT *ht_value = bin.local_hash_table_val[tid];

    for (IT i = start_row; i < end_row; ++i) {
      IT bid = bin.bin_id[i];
      if (bid > 0) {
        IT offset = crpt[i];
        IT ht_size = MIN_HT_N << (bid - 1);
        for (IT j = 0; j < ht_size; ++j) {
          ht_check[j] = -1;
        }
        // union of col-ids of cluster
        for (IT j = arpt[i]; j < arpt[i + 1]; ++j) { // A.cols
          IT t_acol = acol[j];
          NT t_aval = aval[j];
          // B[i]
          for (IT k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
            NT t_val = multop(t_aval, bval[k]);
            IT key = bcol[k];
            IT hash = (key * HASH_SCAL) & (ht_size - 1);
            while (1) { // Loop for hash probing
              if (ht_check[hash] == key) { // key is already inserted
                ht_value[hash] = addop(t_val, ht_value[hash]);
                break;
              } else if (ht_check[hash] == -1) { // insert new entry
                ht_check[hash] = key;
                ht_value[hash] = t_val;
                break;
              } else {
                hash = (hash + 1) & (ht_size - 1); // (hash + 1) % ht_size
              }
            }
          }
        }
//        sort_and_store_table2mat_topK<sortOutput, IT, NT>(ht_check, ht_value,
//                                              ccol + offset, cval + offset,
//                                              crpt[i + 1] - offset, ht_size, topK);
        sort_and_store_table2mat_topK_jaccard<sortOutput, IT, NT>(ht_check, ht_value,
                                              ccol + offset, cval + offset,
                                              crpt[i + 1] - offset, ht_size, topK,
                                              arpt, i);

//        assert(!sortOutput && "can't hande for sortOutput=True");
//        IT u, index = 0;
//        NT jacc;
//        for (IT j = 0; j < ht_size; ++j) {
//          if (ht_check[j] != -1) {
//            u = (arpt[i + 1] - arpt[i]) + (arpt[ht_check[j] + 1] - arpt[ht_check[j]]) - ht_value[j];
//            jacc = (u == 0) ? 0.0 : 1.0 * ht_value[j] / u;
//
//            ccol[offset+index] = ht_check[j];
//            cval[offset+index] = jacc;
//            index++;
//          }
//        }
//        sort_and_store_table2mat_topK_cf<sortOutput, IT, NT>(ht_check, ht_value,
//                                                      ccol + offset, cval + offset,
//                                                      crpt[i + 1] - offset, ht_size, topK,
//                                                      arpt, i);
      }
    }
  }

//    sort(flops_per_thread, flops_per_thread+numThreads);
//    cout << "min-flops: " << flops_per_thread[0] << " max-flops: " << flops_per_thread[numThreads - 1] << endl;
}

template <bool sortOutput, typename NT, typename MultiplyOperation, typename AddOperation>
inline void hash_numeric_vec(const int *arpt, const int *acol, const NT *aval, const int *brpt, const int *bcol, const NT *bval, const int *crpt, int *ccol, NT *cval, const BIN<int, NT> &bin, MultiplyOperation multop, AddOperation addop)
{
#ifdef VECTORIZE
    const __m256i init_m = _mm256_set1_epi32(-1);
    const __m256i true_m = _mm256_set1_epi32(0xffffffff);
#endif        

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int start_row = bin.rows_offset[tid];
        int end_row = bin.rows_offset[tid + 1];

        int *ht_check = bin.local_hash_table_id[tid];
        NT *ht_value = bin.local_hash_table_val[tid];

        for (int i = start_row; i < end_row; ++i) {
#ifdef VECTORIZE
            __m256i key_m, check_m, mask_m;
            int mask;
#endif            
            int bid = bin.bin_id[i];

            if (bid > 0) {
                int offset = crpt[i];
                int table_size = MIN_HT_N << (bid - 1);
                int ht_size = table_size >> VEC_LENGTH_BIT;

                for (int j = 0; j < table_size; ++j) {
                    ht_check[j] = -1;
                }
  
                for (int j = arpt[i]; j < arpt[i + 1]; ++j) {
                    int t_acol = acol[j];
                    NT t_aval = aval[j];
                    for (int k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        NT t_val = multop(t_aval, bval[k]);
	
                        int key = bcol[k];
                        int hash = (key * HASH_SCAL) & (ht_size - 1);
#ifdef VECTORIZE
                        key_m = _mm256_set1_epi32(key);
#endif
                        while (1) {
#ifdef VECTORIZE
                            check_m = _mm256_maskload_epi32(ht_check + (hash << VEC_LENGTH_BIT), true_m);
                            mask_m = _mm256_cmpeq_epi32(key_m, check_m);
                            mask = _mm256_movemask_epi8(mask_m);
                            if (mask != 0) {
                                int target = __builtin_ctz(mask) >> 2;
                                ht_value[(hash << VEC_LENGTH_BIT) + target] += t_val;
                                break;
                            }
#else
                            int flag = -1;
                            for (int l = 0; l < VEC_LENGTH; ++l) {
                                if (ht_check[(hash << VEC_LENGTH_BIT) + l] == key) {
                                    flag = l;
                                }
                            }
                            if (flag >= 0) {
                                ht_value[(hash << VEC_LENGTH_BIT) + flag] += t_val;
                                break;
                            }
#endif
                            else {
                                int cur_nz;
#ifdef VECTORIZE
                                mask_m = _mm256_cmpeq_epi32(check_m, init_m);
                                mask = _mm256_movemask_epi8(mask_m);
                                cur_nz = (32 - _popcnt32(mask)) >> 2;
#else
                                cur_nz = VEC_LENGTH;
                                for (int l = 0; l < VEC_LENGTH; ++l) {
                                    if (ht_check[(hash << VEC_LENGTH_BIT) + l] == -1) {
                                        cur_nz = l;
                                        break;
                                    }
                                }
#endif
                                if (cur_nz < VEC_LENGTH) {
                                    ht_check[(hash << VEC_LENGTH_BIT) + cur_nz] = key;
                                    ht_value[(hash << VEC_LENGTH_BIT) + cur_nz] = t_val;
                                    break;
                                }
                                else {
                                    hash = (hash + 1) & (ht_size - 1);
                                }
                            }
                        }
                    }
                }
                sort_and_store_table2mat<sortOutput, int, NT>(ht_check, ht_value,
                                                              ccol + offset, cval + offset,
                                                              crpt[i + 1] - offset, ht_size, crpt[i+1]);
            }
        }
    }
}

template <bool sortOutput, typename NT, typename MultiplyOperation, typename AddOperation>
inline void hash_numeric_vec(const long long int *arpt, const long long int *acol, const NT *aval, const long long int *brpt, const long long int *bcol, const NT *bval, const long long int *crpt, long long int *ccol, NT *cval, const BIN<long long int, NT> &bin, MultiplyOperation multop, AddOperation addop)
{
#ifdef VECTORIZE
    const __m256i init_m = _mm256_set1_epi64x(-1);
    const __m256i true_m = _mm256_set1_epi64x(0xffffffffffffffff);
#endif        

#pragma omp parallel
    {
        long long int tid = omp_get_thread_num();
        long long int start_row = bin.rows_offset[tid];
        long long int end_row = bin.rows_offset[tid + 1];

        long long int *ht_check = bin.local_hash_table_id[tid];
        NT *ht_value = bin.local_hash_table_val[tid];

        for (long long int i = start_row; i < end_row; ++i) {
#ifdef VECTORIZE
            __m256i key_m, check_m, mask_m;
            int mask;
#endif            
            long long int bid = bin.bin_id[i];

            if (bid > 0) {
                long long int offset = crpt[i];
                long long int table_size = MIN_HT_N << (bid - 1);
                long long int ht_size = table_size >> VEC_LENGTH_LONG_BIT;

                for (long long int j = 0; j < table_size; ++j) {
                    ht_check[j] = -1;
                }
  
                for (long long int j = arpt[i]; j < arpt[i + 1]; ++j) {
                    long long int t_acol = acol[j];
                    NT t_aval = aval[j];
                    for (long long int k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        NT t_val = multop(t_aval, bval[k]);
                        long long int key = bcol[k];
                        long long int hash = (key * HASH_SCAL) & (ht_size - 1);
#ifdef VECTORIZE
                        key_m = _mm256_set1_epi64x(key);
#endif
                        while (1) {
#ifdef VECTORIZE
                            check_m = _mm256_maskload_epi64(ht_check + (hash << VEC_LENGTH_LONG_BIT), true_m);
                            mask_m = _mm256_cmpeq_epi64(key_m, check_m);
                            mask = _mm256_movemask_epi8(mask_m);
                            if (mask != 0) {
                                int target = __builtin_ctz(mask) >> 3;
                                ht_value[(hash << VEC_LENGTH_LONG_BIT) + target] += t_val;
                                break;
                            }
#else
                            int flag = -1;
                            for (int l = 0; l < VEC_LENGTH_LONG; ++l) {
                                if (ht_check[(hash << VEC_LENGTH_LONG_BIT) + l] == key) {
                                    flag = l;
                                }
                            }
                            if (flag >= 0) {
                                ht_value[(hash << VEC_LENGTH_LONG_BIT) + flag] += t_val;
                                break;
                            }
#endif
                            else {
                                int cur_nz;
#ifdef VECTORIZE
                                mask_m = _mm256_cmpeq_epi64(check_m, init_m);
                                mask = _mm256_movemask_epi8(mask_m);
                                cur_nz = (32 - _popcnt32(mask)) >> 3;
#else
                                cur_nz = VEC_LENGTH_LONG;
                                for (int l = 0; l < VEC_LENGTH_LONG; ++l) {
                                    if (ht_check[(hash << VEC_LENGTH_LONG_BIT) + l] == -1) {
                                        cur_nz = l;
                                        break;
                                    }
                                }
#endif
                                if (cur_nz < VEC_LENGTH_LONG) {
                                    ht_check[(hash << VEC_LENGTH_LONG_BIT) + cur_nz] = key;
                                    ht_value[(hash << VEC_LENGTH_LONG_BIT) + cur_nz] = t_val;
                                    break;
                                }
                                else {
                                    hash = (hash + 1) & (ht_size - 1);
                                }
                            }
                        }
                    }
                }
                sort_and_store_table2mat<sortOutput, long long int, NT>(ht_check, ht_value,
                                                                        ccol + offset, cval + offset,
                                                                        crpt[i + 1] - offset, ht_size, crpt[i+1]);
            }
        }
    }
}

/*
 * Numeric phase (with inplace update for SMALL NNZs) in Hash SpGEMM.
 */
template <bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
inline void hash_numeric_with_inplace(const IT *arpt, const IT *acol, const NT *aval, const IT *brpt, const IT *bcol, const NT *bval, const IT *crpt, IT *ccol, NT *cval, const BIN<IT, NT> &bin, const MultiplyOperation multop, const AddOperation addop, IT cnnz, int inplace_cutoff)
{
#pragma omp parallel
  {
    IT tid = omp_get_thread_num();
    IT start_row = bin.rows_offset[tid];
    IT end_row = bin.rows_offset[tid + 1];

    IT row_write_ptr;

    IT *ht_check = bin.local_hash_table_id[tid];
    NT *ht_value = bin.local_hash_table_val[tid];
    for (IT i = start_row; i < end_row; ++i) {  //A.rows
      // in-place update
      if(bin.row_nz[i] > 0 && bin.row_nz[i] <= inplace_cutoff) {
        row_write_ptr = crpt[i];
        for (size_t j = arpt[i]; j < arpt[i + 1]; ++j) {
          size_t rowofb = acol[j];
          for (size_t k = brpt[rowofb]; k < brpt[rowofb + 1]; ++k) {
            if(sortOutput) {
              size_t l = crpt[i];
              long pos = (lower_bound(ccol+l, ccol + row_write_ptr, bcol[k])) - ccol;
              if(pos == row_write_ptr) {
                ccol[row_write_ptr] = bcol[k];
                cval[row_write_ptr] = multop(aval[j], bval[k]);
                row_write_ptr += 1;
              }
              else if(ccol[pos] == bcol[k]) {
                cval[pos] = addop(multop(aval[j], bval[k]), cval[pos]);
              }
              else {
                // do right shift
                l = row_write_ptr - 1;
                while (l >= pos) {
                  ccol[l + 1] = ccol[l];
                  cval[l + 1] = cval[l];
                  l -= 1;
                }
                ccol[pos] = bcol[k];
                cval[pos] = multop(aval[j], bval[k]);
                row_write_ptr += 1;
              }
            }
            else {
              size_t l = crpt[i];
              while (l < row_write_ptr) {
                if (ccol[l] == bcol[k]) {
                  cval[l] = addop(multop(aval[j], bval[k]), cval[l]);
                  break;
                }
                l += 1;
              }
              if (l == row_write_ptr) {
                ccol[row_write_ptr] = bcol[k];
                cval[row_write_ptr] = multop(aval[j], bval[k]);
                row_write_ptr += 1;
              }
            }
          }
        }
      }
        // hashtable update
      else {
        IT bid = bin.bin_id[i];
        if (bid > 0) {
          IT offset = crpt[i];
          IT ht_size = MIN_HT_N << (bid - 1);
          for (IT j = 0; j < ht_size; ++j) {
            ht_check[j] = -1;
          }
          for (IT j = arpt[i]; j < arpt[i + 1]; ++j) {
            IT t_acol = acol[j];
            NT t_aval = aval[j];
            for (IT k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
              NT t_val = multop(t_aval, bval[k]);
              IT key = bcol[k];
              IT hash = (key * HASH_SCAL) & (ht_size - 1);
              while (1) { // Loop for hash probing
                if (ht_check[hash] == key) { // key is already inserted
                  ht_value[hash] = addop(t_val, ht_value[hash]);
                  break;
                } else if (ht_check[hash] == -1) { // insert new entry
                  ht_check[hash] = key;
                  ht_value[hash] = t_val;
                  break;
                } else {
                  hash = (hash + 1) & (ht_size - 1); // (hash + 1) % ht_size
                }
              }
            }
          }
          sort_and_store_table2mat<sortOutput, IT, NT>(ht_check, ht_value,
                                                       ccol + offset, cval + offset,
                                                       crpt[i + 1] - offset, ht_size, cnnz - offset);
        }
      }
    }
  }
}

/*
 * Numeric phase (with inplace update for SMALL NNZs) in Hash SpGEMM.
 * The ccol (col-ids) will be prefilled in the symbolic phase (in sorted order).
 */
template <bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
inline void hash_numeric_with_inplace_V1(const IT *arpt, const IT *acol, const NT *aval,
                                         const IT *brpt, const IT *bcol, const NT *bval,
                                         const IT *crpt, IT *ccol, NT *cval, const BIN<IT, NT> &bin,
                                         const MultiplyOperation multop, const AddOperation addop, IT cnnz,
                                         int inplace_cutoff, vector<bool>& inplace_flags)
{
#pragma omp parallel
  {
    IT tid = omp_get_thread_num();
    IT start_row = bin.rows_offset[tid];
    IT end_row = bin.rows_offset[tid + 1];

    IT inplace_pos_st;
    IT *ht_check = bin.local_hash_table_id[tid];
    NT *ht_value = bin.local_hash_table_val[tid];
    for (IT i = start_row; i < end_row; ++i) {  //A.rows
      // in-place update
      if(bin.row_nz[i] > 0 && bin.row_nz[i] <= inplace_cutoff) {
        inplace_pos_st = i * inplace_cutoff;
        for (size_t j = arpt[i]; j < arpt[i + 1]; ++j) {
          size_t rowofb = acol[j];
          for (size_t k = brpt[rowofb]; k < brpt[rowofb + 1]; ++k) {
            size_t l = crpt[i];
            long pos = (lower_bound(ccol+l, ccol + crpt[i+1], bcol[k])) - ccol;
            assert(pos < crpt[i+1] && "bcol[k] should be found in the output row i");
            if(!inplace_flags[inplace_pos_st + (pos-l)]) {
              cval[pos] = multop(aval[j], bval[k]);
              inplace_flags[inplace_pos_st + (pos-l)] = 1;
            }
            else {
              cval[pos] = addop(multop(aval[j], bval[k]), cval[pos]);
            }
          }
        }
      }
      // hashtable update
      else {
        IT bid = bin.bin_id[i];
        if (bid > 0) {
          IT offset = crpt[i];
          IT ht_size = MIN_HT_N << (bid - 1);
          for (IT j = 0; j < ht_size; ++j) {
            ht_check[j] = -1;
          }
          for (IT j = arpt[i]; j < arpt[i + 1]; ++j) {
            IT t_acol = acol[j];
            NT t_aval = aval[j];
            for (IT k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
              NT t_val = multop(t_aval, bval[k]);
              IT key = bcol[k];
              IT hash = (key * HASH_SCAL) & (ht_size - 1);
              while (1) { // Loop for hash probing
                if (ht_check[hash] == key) { // key is already inserted
                  ht_value[hash] = addop(t_val, ht_value[hash]);
                  break;
                } else if (ht_check[hash] == -1) { // insert new entry
                  ht_check[hash] = key;
                  ht_value[hash] = t_val;
                  break;
                } else {
                  hash = (hash + 1) & (ht_size - 1); // (hash + 1) % ht_size
                }
              }
            }
          }
          sort_and_store_table2mat<sortOutput, IT, NT>(ht_check, ht_value,
                                                       ccol + offset, cval + offset,
                                                       crpt[i + 1] - offset, ht_size, cnnz - offset);
        }
      }
    }
  }
}

/*
 * Symbolic phase for Hash SpGEMM (with inplace update for SMALL NNZs).
 * The ccol (col-ids) will be prefilled in the symbolic phase (in sorted order).
 */
template <class IT, class NT>
inline void hash_symbolic_kernel_with_inplace(const IT *arpt, const IT *acol, const IT *brpt, const IT *bcol,
                                              BIN<IT, NT> &bin, IT *inplace_colids, int inplace_cutoff) {
#pragma omp parallel
  {
    IT tid = omp_get_thread_num();
    IT start_row = bin.rows_offset[tid];
    IT end_row = bin.rows_offset[tid + 1];

    IT inplace_pos_st;
    IT *check = bin.local_hash_table_id[tid];

    for (IT i = start_row; i < end_row; ++i) {
      IT nz = 0;
      IT bid = bin.bin_id[i];
      inplace_pos_st = i * inplace_cutoff;

      if (bid > 0) {
        IT ht_size = MIN_HT_S << (bid - 1); // determine hash table size for i-th row
        for (IT j = 0; j < ht_size; ++j) { // initialize hash table
          check[j] = -1;
        }

        for (IT j = arpt[i]; j < arpt[i + 1]; ++j) {
          IT t_acol = acol[j];
          for (IT k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
            IT key = bcol[k];
            IT hash = (key * HASH_SCAL) & (ht_size - 1);
            while (1) { // Loop for hash probing
              if (check[hash] == key) { // if the key is already inserted, it's ok
                break;
              }
              else if (check[hash] == -1) { // if the key has not been inserted yet, then it's added.
                check[hash] = key;
                if(nz < inplace_cutoff) inplace_colids[inplace_pos_st + nz] = key;
                nz++;
                break;
              }
              else { // linear probing: check next entry
                hash = (hash + 1) & (ht_size - 1); //hash = (hash + 1) % ht_size
              }
            }
          }
        }
      }
      bin.row_nz[i] = nz;
    }
  }
}

template <class IT, class NT>
inline void copy_symbolic_results(const IT *crpt, IT *ccol, BIN<IT, NT> &bin, IT *inplace_colids, int inplace_cutoff) {
#pragma omp parallel
  {
    IT tid = omp_get_thread_num();
    IT start_row = bin.rows_offset[tid];
    IT end_row = bin.rows_offset[tid + 1];

    IT inplace_pos_st;
//    IT *check = bin.local_hash_table_id[tid];

    for (IT i = start_row; i < end_row; ++i) {
      if(bin.row_nz[i] > 0 && bin.row_nz[i] <= inplace_cutoff) {
        sort(inplace_colids + (i * inplace_cutoff), inplace_colids + ((i * inplace_cutoff) + bin.row_nz[i]));
        memcpy(ccol + crpt[i], inplace_colids + (i * inplace_cutoff), bin.row_nz[i] * sizeof(IT));
      }
    }
  }
}

/*
 * Executing Hash SpGEMM with inplace update.
 * The function starts with initialization of hash table followed by symbolic phase and numeric phase with hash table.
 * The col-ids within the inplace cutoff nnz will be stored in the output matrix during symbolic phase (in sorted order),
 *  so that we can use binary search to update values during numeric phase.
 */
template <bool vectorProbing, bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void HashSpGEMMWithInplaceUpdateV1(const CSR<IT, NT> &a, const CSR<IT, NT> &b, CSR<IT, NT> &c,
                                   MultiplyOperation multop, AddOperation addop, int inplace_cutoff)
{
  BIN<IT, NT> bin(a.rows, MIN_HT_S);

  c.rows = a.rows;
  c.cols = b.cols;
  c.zerobased = true;

  /* Set max bin */
  bin.set_max_bin(a.rowptr, a.colids, b.rowptr, c.rows, c.cols);

  /* Create hash table (thread local) */
  bin.create_local_hash_table(c.cols);

  /* Symbolic Phase */
  c.rowptr = my_malloc<IT>(c.rows + 1);
  IT* inplace_colids = my_malloc<IT>(a.rows * inplace_cutoff);
  hash_symbolic_kernel_with_inplace(a.rowptr, a.colids, b.rowptr, b.colids, bin, inplace_colids, inplace_cutoff);
//  cout << "symbolic done" << endl;

  /* Set row pointer of matrix C */
  scan(bin.row_nz, c.rowptr, a.rows + 1);
  c.nnz = c.rowptr[c.rows];

  c.colids = my_malloc<IT>(c.nnz);
  c.values = my_malloc<NT>(c.nnz);

  copy_symbolic_results(c.rowptr, c.colids, bin, inplace_colids, inplace_cutoff);
  my_free<IT>(inplace_colids);
//  cout << "data copy done" << endl;

//  int tmp = 0;
//  for(IT i = 0; i < a.rows; i+=1) {
//    if(bin.row_nz[i] <= inplace_cutoff) tmp += 1;
//  }
//  cout << "total rows: " << a.rows << ", inplace: " << tmp << endl;

  /* Numeric Phase */
  vector<bool> inplace_flags(a.rows * inplace_cutoff, 0);
  hash_numeric_with_inplace_V1<sortOutput>(a.rowptr, a.colids, a.values, b.rowptr, b.colids, b.values,
                                           c.rowptr, c.colids, c.values, bin, multop, addop, c.nnz,
                                           inplace_cutoff, inplace_flags);
//  cout << "numeric done" << endl;
}

/*
 * Executing Hash SpGEMM with inplace update
 * The function starts with initialization of hash table followed by symbolic phase and numeric phase with hash table.
 */
template <bool vectorProbing, bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void HashSpGEMMWithInplaceUpdate(const CSR<IT, NT> &a, const CSR<IT, NT> &b, CSR<IT, NT> &c, MultiplyOperation multop, AddOperation addop, int inplace_cutoff, string out_nnz_freq_outfile, bool first_pass = false)
{
  BIN<IT, NT> bin(a.rows, MIN_HT_S);

  c.rows = a.rows;
  c.cols = b.cols;
  c.zerobased = true;

  /* Set max bin */
  bin.set_max_bin(a.rowptr, a.colids, b.rowptr, c.rows, c.cols);

  /* Create hash table (thread local) */
  bin.create_local_hash_table(c.cols);

  /* Symbolic Phase */
  c.rowptr = my_malloc<IT>(c.rows + 1);
  hash_symbolic<vectorProbing>(a.rowptr, a.colids, b.rowptr, b.colids, c.rowptr,
                               bin, c.rows, &(c.nnz), out_nnz_freq_outfile, first_pass);

  c.colids = my_malloc<IT>(c.nnz);
  c.values = my_malloc<NT>(c.nnz);

//  int tmp = 0;
//  for(IT i = 0; i < a.rows; i+=1) {
//    if(bin.row_nz[i] <= inplace_cutoff) tmp += 1;
//  }
//  cout << "total rows: " << a.rows << ", inplace: " << tmp << endl;

  /* Numeric Phase */
  hash_numeric_with_inplace<sortOutput>(a.rowptr, a.colids, a.values, b.rowptr, b.colids, b.values, c.rowptr, c.colids, c.values, bin, multop, addop, c.nnz, inplace_cutoff);
}

/*
 * Executing Hash SpGEMM
 * The function starts with initialization of hash table followed by symbolic phase and numeric phase with hash table.
 */
template <bool vectorProbing, bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void RowSpGEMM(const CSR<IT, NT> &a, const CSR<IT, NT> &b, CSR<IT, NT> &c, MultiplyOperation multop, AddOperation addop, string out_nnz_freq_outfile, bool first_pass = false)
{
//  double start, end, msec;
//  start = omp_get_wtime();
    BIN<IT, NT> bin(a.rows, MIN_HT_S);
  
    c.rows = a.rows;
    c.cols = b.cols;
    c.zerobased = true;

    /* Set max bin */
    bin.set_max_bin(a.rowptr, a.colids, b.rowptr, c.rows, c.cols);
    
    /* Create hash table (thread local) */
    bin.create_local_hash_table(c.cols);

    /* Symbolic Phase */
    c.rowptr = my_malloc<IT>(c.rows + 1);
//  end = omp_get_wtime();
//  msec = (end - start) * 1000;
//  cout << "BIN initialization time: " << msec << " milli seconds." << endl;

//  start = omp_get_wtime();
    hash_symbolic<vectorProbing>(a.rowptr, a.colids, b.rowptr, b.colids, c.rowptr,
                                 bin, c.rows, &(c.nnz), out_nnz_freq_outfile, first_pass);

    // todo: adding this to reduce hashtable size
    bin.set_bin_id(c.rows, c.cols, bin.min_ht_size);

    c.colids = nullptr;
    c.values = nullptr;
    c.colids = my_malloc<IT>(c.nnz);
    c.values = my_malloc<NT>(c.nnz);

//  cout << "[DONE] hash_symbolic " << c.nnz << endl;

    assert(c.colids != nullptr && c.values != nullptr && "C.colids & C.values are not initialized poperly!");
//  end = omp_get_wtime();
//  msec = (end - start) * 1000;
//  cout << "Symbolic time: " << msec << " milli seconds." << endl;

//    sanity_check(c.rowptr, c.nnz, c.rows);

    /* Numeric Phase */
//    if (vectorProbing) {
//        hash_numeric_vec<sortOutput>(a.rowptr, a.colids, a.values, b.rowptr, b.colids, b.values, c.rowptr, c.colids, c.values, bin, multop, addop);
//    }
//    else {
//  start = omp_get_wtime();
        hash_numeric<sortOutput>(a.rowptr, a.colids, a.values, b.rowptr, b.colids, b.values, c.rowptr, c.colids, c.values, bin, multop, addop, c.nnz);
//  end = omp_get_wtime();
//  msec = (end - start) * 1000;
//  cout << "Numeric time: " << msec << " milli seconds." << endl;
//    }
}

/*
 * Executing Hash SpGEMM
 * The function starts with initialization of hash table followed by symbolic phase and numeric phase with hash table.
 */
template <bool vectorProbing, bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void HashSpGEMMtoCheckMemoryRequirement(const CSR<IT, NT> &a, const CSR<IT, NT> &b, CSR<IT, NT> &c, MultiplyOperation multop, AddOperation addop, string out_nnz_freq_outfile, bool first_pass = false)
{
  BIN<IT, NT> bin(a.rows, MIN_HT_S);

  c.rows = a.rows;
  c.cols = b.cols;
  c.zerobased = true;

  /* Set max bin */
  bin.set_max_bin(a.rowptr, a.colids, b.rowptr, c.rows, c.cols);

  /* Create hash table (thread local) */
  bin.create_local_hash_table(c.cols);

  /* Symbolic Phase */
  c.rowptr = my_malloc<IT>(c.rows + 1);

  hash_symbolic<vectorProbing>(a.rowptr, a.colids, b.rowptr, b.colids, c.rowptr,
                               bin, c.rows, &(c.nnz), out_nnz_freq_outfile, first_pass);

  bin.set_bin_id(c.rows, c.cols, bin.min_ht_size);

  c.colids = nullptr;
  c.values = nullptr;
  c.colids = my_malloc<IT>(c.nnz);
  c.values = my_malloc<NT>(c.nnz);

  assert(c.colids != nullptr && c.values != nullptr && "C.colids & C.values are not initialized poperly!");

  printf("BIN size: %.2f bytes\n", bin.calculate_size_in_gb());
  printf("C size: %.2f bytes\n", c.calculate_size());
}

/*
 * Executing Hash SpGEMM
 * The function starts with initialization of hash table followed by symbolic phase and numeric phase with hash table.
 */
template <bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void HashSpGEMMTopK(const CSR<IT, NT> &a, const CSR<IT, NT> &b, CSR<IT, NT> &c, MultiplyOperation multop, AddOperation addop, IT topK, string out_nnz_freq_outfile, bool first_pass = false)
{
  BIN<IT, NT> bin(a.rows, MIN_HT_S);

  c.rows = a.rows;
  c.cols = b.cols;
  c.zerobased = true;

  /* Set max bin */
  bin.set_max_bin(a.rowptr, a.colids, b.rowptr, c.rows, c.cols);

  /* Create hash table (thread local) */
  bin.create_local_hash_table(c.cols);

  /* Symbolic Phase */
  c.rowptr = my_malloc<IT>(c.rows + 1);
  hash_symbolic_topK(a.rowptr, a.colids, b.rowptr, b.colids, c.rowptr,
                               bin, c.rows, &(c.nnz), topK, out_nnz_freq_outfile, first_pass);
//  hash_symbolic<false>(a.rowptr, a.colids, b.rowptr, b.colids, c.rowptr,
//                     bin, c.rows, &(c.nnz), out_nnz_freq_outfile, first_pass);

  cout << "[DONE] hash_symbolic_topK " << c.nnz << endl;

  // todo: adding this to reduce hashtable size
  bin.set_bin_id(c.rows, c.cols, bin.min_ht_size);

  c.colids = nullptr;
  c.values = nullptr;
  c.colids = my_malloc<IT>(c.nnz);
  c.values = my_malloc<NT>(c.nnz);

  c.calculate_size();

//  c.colids = my_malloc<IT>(a.rows * topK);
//  c.values = my_malloc<NT>(a.rows * topK);
//  memset(c.colids, -1, sizeof(c.colids) * (a.rows * topK));
//  memset(c.values, -1, sizeof(c.values));

  bin.calculate_size_in_gb();
  cout << "[DONE] memory allocation" << endl;

//  for(int i=0; i<c.rows; i+=1) cout << c.rowptr[i] << " ";
//  cout << endl;

  assert(c.colids != nullptr && c.values != nullptr && "C.colids & C.values are not initialized poperly!");

//  hash_numeric<sortOutput>(a.rowptr, a.colids, a.values, b.rowptr, b.colids, b.values, c.rowptr, c.colids, c.values, bin, multop, addop, c.nnz);
  hash_numeric_topk<sortOutput>(a.rowptr, a.colids, a.values, b.rowptr, b.colids, b.values, c.rowptr, c.colids, c.values, bin, multop, addop, topK);
//  bool sortedOutput = true;
//  hash_numeric<true>(a.rowptr, a.colids, a.values, b.rowptr, b.colids, b.values, c.rowptr, c.colids, c.values, bin, multop, addop, c.nnz);

  cout << "[DONE] hash_numeric_topk" << endl;
}

/*
 * Hash SpGEMM functions called without full template values
 */
template <bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void RowSpGEMM(const CSR<IT, NT> &a, const CSR<IT, NT> &b, CSR<IT, NT> &c, MultiplyOperation multop, AddOperation addop)
{
    RowSpGEMM<false, sortOutput, IT, NT, MultiplyOperation, AddOperation>(a, b, c, multop, addop);
}

template <typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void RowSpGEMM(const CSR<IT, NT> &a, const CSR<IT, NT> &b, CSR<IT, NT> &c, MultiplyOperation multop, AddOperation addop)
{
    RowSpGEMM<false, true, IT, NT, MultiplyOperation, AddOperation>(a, b, c, multop, addop);
}
