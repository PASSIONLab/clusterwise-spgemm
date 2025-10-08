#ifndef _CSR_VLENGTH_CLUSTER_H_
#define _CSR_VLENGTH_CLUSTER_H_

#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <cstdlib>
#include <algorithm>
#include <cassert>
#include "CSR.h"

#include <tbb/scalable_allocator.h>

#include <random>
#include "utility.h"

using namespace std;

/*
Class:  CSR_VlengthCluster
Author: Raqib

Simple container for sparse matrix in variable-length clustered CSR format
 - Construct from CSR
*/

template<class IT, class NT>
class CSR_VlengthCluster {
public:
  CSR_VlengthCluster() : nnzc(0), rows(0), cols(0) {}

  CSR_VlengthCluster(const CSR<IT, NT> &csr, const map<IT, vector<IT>> &reordered_dict);
  CSR_VlengthCluster(const CSR<IT, NT> &csr, const vector<IT> &offsets);

  NT calculate_average_fullness(const NT eps = 0.000001f);                    // calculate how many colids.vectors<> are fully occupied by value
  NT calculate_average_fill_factor(const NT eps = 0.000001f);                 // calculate ratio of count(not_empty(this.values)) / length(this.values)
  NT calculate_fill_factor(const IT cluster_id, const NT eps = 0.000001f);    // calculate ratio of count(not_empty(this.values[cluster_id])) / length(this.values[cluster_id])
  NT calculate_size();    // calculate size in Bytes
  NT calculate_size_in_gb();    // calculate size in GB

  void print_rows(IT start, IT end);  // Print CSR_VlengthCluster rows

  void make_empty() {
    if (nnzc > 0) {
      my_free<IT>(colids);
      my_free<NT>(values);
      nnzc = 0;
    }
    if (rows > 0) {
      my_free<IT>(rowptr);
      my_free<IT>(rowptr_val);
      my_free<IT>(cluster_sz);
      rows = 0;
    }
    cols = 0;
  }

  ~CSR_VlengthCluster() {
    make_empty();
  }

  bool isEmpty() {
    return (nnzc == 0);
  }

  void Sorted();

  IT csr_rows;          // number of rows in the original CSR format
  IT rows;              // number of rows in the clustered CSR format (i.e., number of clusters)
  IT cols;              // number of columns (max) in the original CSR format
  IT nnzc;              // sum of unique columns in each cluster (similar to nnz in CSR format, but should be larger or equal compared to CSR.nnz)
  IT nnzv;              // size of the values array; sum of {(rowptr[i+1] - rowptr[i]) * cluster_sz[i]} for all cluster[i]
  IT max_cluster_sz;    // max(cluster_sz); useful in SPA

  IT *cluster_sz;       // number of rows folded into cluster[i]
  IT *rowptr;           // row pointer for colids array
  IT *colids;           // unique column-ids in cluster[i]
  IT *rowptr_val;       // row pointer for values array
  NT *values;           // store values
                        //  - for column-id at i-th cluster, values will be stored
                        //  - from rowptr_val[i] to rowptr_val[i+1]
};

//! Construct a CSR_VlengthCluster object from a CSR
//! @offsets are the offset of the clusters to CSR rows
template<class IT, class NT>
CSR_VlengthCluster<IT, NT>::CSR_VlengthCluster(const CSR<IT, NT> &csr, const vector<IT> &offsets) : csr_rows(csr.rows), cols(csr.cols), nnzc(0), nnzv(0), max_cluster_sz(256) {
  rows = offsets[offsets.size() - 1];

  rowptr = my_malloc<IT>(rows + 1);
  rowptr_val = my_malloc<IT>(rows + 1);
  cluster_sz = my_malloc<IT>(rows);

  IT *work = my_malloc<IT>(rows);                 // store number of columns per-cluster
  IT *work_val = my_malloc<IT>(rows);             // store the number of values per-cluster (e.g., work[i] * cluster_sz[i])
  std::fill(work, work + rows, (IT) 0);                    // initilized to zero
  vector < map < IT, vector < NT >, less < IT>> > col_map(rows);    // for each cluster map the column-ids to values

  IT cluster_id = 0;

  for (size_t r=1; r<offsets.size(); r+=1) {                                        // loop over the clusters
    cluster_sz[cluster_id] = offsets[r] - offsets[r - 1];
    IT ii = 0;
    for (IT i=offsets[r - 1]; i<offsets[r]; i+=1) {                                                             // loop over the row-ids of cluster r
      for (IT j = csr.rowptr[i]; j < csr.rowptr[i + 1]; j += 1) {                       // loop over the columns of CSR.rows[i]
        IT t_acol = csr.colids[j];
        NT t_aval = csr.values[j];
        if (col_map[cluster_id].find(t_acol) == col_map[cluster_id].end()) {
          col_map[cluster_id][t_acol] = vector < NT > (cluster_sz[cluster_id], 0.0);
        }
        col_map[cluster_id][t_acol][ii] = t_aval;
      }
      ii += 1;
    }
    nnzc += col_map[cluster_id].size();
    work[cluster_id] = col_map[cluster_id].size();

    nnzv += (work[cluster_id] * cluster_sz[cluster_id]);
    work_val[cluster_id] = work[cluster_id] * cluster_sz[cluster_id];

    cluster_id += 1;
  }

  colids = my_malloc<IT>(nnzc);
  values = my_malloc<NT>(nnzv);

  rowptr[rows] = CumulativeSum(work, rows);    // cumulative sum of work
  copy(work, work + rows, rowptr);

  rowptr_val[rows] = CumulativeSum(work_val, rows);    // cumulative sum of work_val
  copy(work_val, work_val + rows, rowptr_val);

  for (IT i = 0; i < rows; i += 1) {                  // loop over cluster-ids
    for (auto it: col_map[i]) {                       // loop over columns of cluster[i]
      colids[work[i]++] = it.first;
      for (IT j = 0; j < cluster_sz[i]; j += 1) {            // loop over values of col-id last
        if (work_val[i] >= nnzv) {
          cout << "rows: " << rows << ", i: " << i;
          cout << ", rowptr_val[i]: " << rowptr_val[i] << ", nnzv: " << nnzv << endl;
        }
        assert(work_val[i] < nnzv && "trying to write beyond the value boundary");
        values[work_val[i]++] = it.second[j];
      }
    }
  }

  my_free<IT>(work);        // free memory
  my_free<IT>(work_val);    // free memory
}

//! Construct a CSR_VlengthCluster object from a CSR
//! @reordered_dict maps cluster-id to CSR rows
template<class IT, class NT>
CSR_VlengthCluster<IT, NT>::CSR_VlengthCluster(const CSR<IT, NT> &csr, const map<IT, vector<IT>> &reordered_dict): csr_rows(
    csr.rows), cols(csr.cols), nnzc(0), nnzv(0), max_cluster_sz(256) {
  rows = reordered_dict.size();

  rowptr = my_malloc<IT>(rows + 1);
  rowptr_val = my_malloc<IT>(rows + 1);
  cluster_sz = my_malloc<IT>(rows);

  IT *work = my_malloc<IT>(rows);                 // store number of columns per-cluster
  IT *work_val = my_malloc<IT>(rows);             // store the number of values per-cluster (e.g., work[i] * cluster_sz[i])
  std::fill(work, work + rows, (IT) 0);                    // initilized to zero
  vector < map < IT, vector < NT >, less < IT>> > col_map(rows);    // for each cluster map the column-ids to values

  IT cluster_id = 0;
  for (auto &r: reordered_dict) {                                        // loop over the clusters
    cluster_sz[cluster_id] = r.second.size();
    IT ii = 0;
    for (IT i: r.second) {                                                             // loop over the row-ids of cluster r
      for (IT j = csr.rowptr[i]; j < csr.rowptr[i + 1]; j += 1) {                       // loop over the columns of CSR.rows[i]
        IT t_acol = csr.colids[j];
        NT t_aval = csr.values[j];
        if (col_map[cluster_id].find(t_acol) == col_map[cluster_id].end()) {
          col_map[cluster_id][t_acol] = vector < NT > (cluster_sz[cluster_id], 0.0);
        }
        col_map[cluster_id][t_acol][ii] = t_aval;
      }
      ii += 1;
    }
    nnzc += col_map[cluster_id].size();
    work[cluster_id] = col_map[cluster_id].size();

    nnzv += (work[cluster_id] * cluster_sz[cluster_id]);
    work_val[cluster_id] = work[cluster_id] * cluster_sz[cluster_id];

    cluster_id += 1;
  }

  colids = my_malloc<IT>(nnzc);
  values = my_malloc<NT>(nnzv);

  rowptr[rows] = CumulativeSum(work, rows);    // cumulative sum of work
  copy(work, work + rows, rowptr);

  rowptr_val[rows] = CumulativeSum(work_val, rows);    // cumulative sum of work_val
  copy(work_val, work_val + rows, rowptr_val);

  for (IT i = 0; i < rows; i += 1) {                  // loop over cluster-ids
    for (auto it: col_map[i]) {                       // loop over columns of cluster[i]
      colids[work[i]++] = it.first;
      for (IT j = 0; j < cluster_sz[i]; j += 1) {            // loop over values of col-id last
        if (work_val[i] >= nnzv) {
          cout << "rows: " << rows << ", i: " << i;
          cout << ", rowptr_val[i]: " << rowptr_val[i] << ", nnzv: " << nnzv << endl;
        }
        assert(work_val[i] < nnzv && "trying to write beyond the value boundary");
        values[work_val[i]++] = it.second[j];
      }
    }
  }

  my_free<IT>(work);        // free memory
  my_free<IT>(work_val);    // free memory
}

//! calculate how many colids.vectors<> are fully occupied by value
template <class IT, class NT>
NT CSR_VlengthCluster<IT,NT>::calculate_average_fullness(const NT eps)
{
  NT up = 0.0;
  IT down = 0;
//#pragma omp parallel for
  for (IT i = 0; i < rows; ++i)
  {
    for (IT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
      IT k = 0;
      while(k < cluster_sz[i]) {
        if(fabs(values[rowptr_val[i] + ((j - rowptr[i]) * cluster_sz[i]) + k] - 0.0f) < eps) break;
        k += 1;
      }
      if (k == cluster_sz[i]) up += 1;
      down += 1;
    }
  }
  if(down > 0) up /= down;
  return up;
}

//! calculate ratio of count(not_empty(this.values)) / length(this.values)
template <class IT, class NT>
NT CSR_VlengthCluster<IT,NT>::calculate_average_fill_factor(const NT eps)
{
  NT non_empty = 0.0;
//#pragma omp parallel for
  for (IT i = 0; i < rows; ++i)
  {
    for (IT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
      for (IT k = 0; k < cluster_sz[i]; ++k) {
        if(fabs(values[rowptr_val[i] + ((j - rowptr[i]) * cluster_sz[i]) + k] - 0.0f) >= eps) non_empty += 1;
      }
    }
  }
  if(nnzv > 0) non_empty /= nnzv;
  return non_empty;
}

//! calculate ratio of count(not_empty(this.values[cluster_id])) / length(this.values[cluster_id])
template <class IT, class NT>
NT CSR_VlengthCluster<IT,NT>::calculate_fill_factor(const IT cluster_id, const NT eps)
{
  NT non_empty = 0.0;
//#pragma omp parallel for
  for (IT j = rowptr[cluster_id]; j < rowptr[cluster_id + 1]; ++j) {
    for (IT k = 0; k < cluster_sz[cluster_id]; ++k) {
      if(fabs(values[rowptr_val[cluster_id] + ((j - rowptr[cluster_id]) * cluster_sz[cluster_id]) + k] - 0.0f) >= eps) non_empty += 1;
    }
  }
  IT num_cols = rowptr[cluster_id + 1] - rowptr[cluster_id];
  if(num_cols > 0) non_empty /= (num_cols * cluster_sz[cluster_id]);
  return non_empty;
}

//! calculate size in Bytes
template <class IT, class NT>
NT CSR_VlengthCluster<IT,NT>::calculate_size() {
  NT total = 0;
  total += ((rows + 1) * sizeof(IT));             // rowptr
  total += ((rows + 1) * sizeof(IT));             // rowptr_val
  total += (rows * sizeof(IT));                   // cluster_sz
  total += (nnzc * sizeof(IT));                   // colids
  total += (nnzv * sizeof(NT));                   // values
  return total;
}

template<class IT, class NT>
NT CSR_VlengthCluster<IT, NT>::calculate_size_in_gb() {
  NT size_bytes = 0;

  if (cluster_sz)      size_bytes += rows * sizeof(IT);
  if (rowptr)          size_bytes += (rows + 1) * sizeof(IT);
  if (colids)          size_bytes += nnzc * sizeof(IT);
  if (rowptr_val)      size_bytes += (rows + 1) * sizeof(IT);
  if (values)          size_bytes += nnzv * sizeof(NT);

  // Return size in GB
//  return static_cast<NT>(size_bytes) / static_cast<NT>(1L << 30);
  return static_cast<NT>(size_bytes);
}

//! Print rows of CSR_VlengthCluster
template<class IT, class NT>
void CSR_VlengthCluster<IT, NT>::print_rows(IT start, IT end) {
  for (IT i = start; i < end; ++i) {
    cout << i << ":";
    for (IT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
      cout << " " << colids[j] << "(";
      for (IT k = 0; k < cluster_sz[i]; k += 1) {
        cout << values[rowptr_val[i] + ((j - rowptr[i]) * cluster_sz[i]) + k] << " ";
      }
      cout << ")";
    }
    cout << endl;
  }
}

#endif  //_CSR_VLENGTH_CLUSTER_H_
