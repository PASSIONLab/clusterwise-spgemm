#ifndef _CSR_FLENGTH_CLUSTER_H_
#define _CSR_FLENGTH_CLUSTER_H_

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
Class:  CSR_FlengthCluster
Author: Raqib

Simple container for sparse matrix in fixed-length clustered CSR format
 - Construct from CSR
*/

template<class IT, class NT>
class CSR_FlengthCluster {
public:
  CSR_FlengthCluster() : nnzc(0), rows(0), cols(0) {}

  CSR_FlengthCluster(const CSR<IT, NT> &csr, IT cluster_size);

  NT calculate_average_fullness(const NT eps = 0.000001f);      // calculate how many colids.vectors<> are fully occupied by value
  NT calculate_average_fill_factor(const NT eps = 0.000001f);   // calculate ratio of count(not_empty(this.values)) / length(this.values)
  NT calculate_fill_factor(const IT cluster_id, const NT eps = 0.000001f);  // calculate ratio of count(not_empty(this.values[cluster_id])) / length(this.values[cluster_id])
  double calculate_size();    // calculate size in Bytes

  void print_rows(IT start, IT end);  // Print CSR_FlengthCluster rows

  void make_empty() {
    if (nnzc > 0) {
      my_free<IT>(colids);
      my_free<NT>(values);
      nnzc = 0;
    }
    if (rows > 0) {
      my_free<IT>(rowptr);
      rows = 0;
    }
    cols = 0;
  }

  ~CSR_FlengthCluster() {
    make_empty();
  }

  bool isEmpty() {
    return (nnzc == 0);
  }

  void Sorted();

  IT csr_rows;    // number of rows in the original CSR format
  IT rows;        // number of rows in the clustered CSR format (i.e., number of clusters)
  IT cols;        // number of columns (max) in the original CSR format
  IT cluster_sz;  // number of rows folded into a cluster
  IT nnzc;        // sum of unique columns in each cluster (similar to nnz in CSR format, but should be larger or equal compared to CSR.nnz)

  IT *rowptr;     // row pointer
  IT *colids;     // unique column-ids in each cluster
  NT *values;     // store values
                  //  - for column-id at i-th cluster, values will be stored
                  //  - from (i * cluster_sz) to {(i * cluster_sz) + cluster_sz} in this array.
};

//! Construct a CSR_FlengthCluster object from a CSR
template<class IT, class NT>
CSR_FlengthCluster<IT, NT>::CSR_FlengthCluster(const CSR<IT, NT> &csr, IT cluster_size): cols(csr.cols), cluster_sz(cluster_size) {
  csr_rows = csr.rows;
  rows = csr.rows / cluster_size;
  if (csr.rows % cluster_size) rows += 1;

  rowptr = my_malloc<IT>(rows + 1);

  nnzc = 0;
  IT *work = my_malloc<IT>(rows);     // store number of columns per-cluster
  std::fill(work, work + rows, (IT) 0);        // initilized to zero
  vector < map < IT, vector < NT >, less < IT>> > col_map(rows);    // for each cluster map column-ids to values

  for (IT r = 0; r < csr.rows; r += cluster_sz) {                           // loop over clusters
    IT cluster_id = r / cluster_sz;
    for (IT i = r; i < r + cluster_sz && i < csr.rows; i += 1) {            // loop over CSR.rows
      for (IT j = csr.rowptr[i]; j < csr.rowptr[i + 1]; j += 1) {           // loop over columns of CSR.rows[i]
        IT t_acol = csr.colids[j];    // col-id
        NT t_aval = csr.values[j];    // value
        if (col_map[cluster_id].find(t_acol) == col_map[cluster_id].end()) {
          col_map[cluster_id][t_acol] = vector < NT > (cluster_size, 0.0);
        }
        col_map[cluster_id][t_acol][i - r] = t_aval;
      }
    }
    nnzc += col_map[cluster_id].size();
    work[cluster_id] = col_map[cluster_id].size();
  }

  colids = my_malloc<IT>(nnzc);
  values = my_malloc<NT>(nnzc * cluster_sz);

  rowptr[rows] = CumulativeSum(work, rows);    // cumulative sum of work
  copy(work, work + rows, rowptr);

  IT last;
  for (IT i = 0; i < rows; i += 1) {                        // loop over cluster-ids
    for (auto it: col_map[i]) {                             // loop over columns of cluster[i]
      colids[last = work[i]++] = it.first;
      for (IT j = 0; j < cluster_sz; j += 1) {                     // loop over values of col-id last
        values[(last * cluster_sz) + j] = it.second[j];
      }
    }
  }

  my_free<IT>(work);      // free memory
}

//! calculate how many colids.vectors<> are fully occupied by value
template <class IT, class NT>
NT CSR_FlengthCluster<IT,NT>::calculate_average_fullness(const NT eps)
{
  NT up = 0.0;
//  IT down = nnzc;
//#pragma omp parallel for
  for (IT i = 0; i < rows; ++i)
  {
    for (IT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
      IT k = 0;
      while(k < cluster_sz) {
        if(fabs(values[(j * cluster_sz) + k] - 0.0f) < eps) break;
        k += 1;
      }
      if (k == cluster_sz) up += 1;
//      down += 1;
    }
  }
  if(nnzc > 0) up /= nnzc;
  return up;
}

//! calculate ratio of count(not_empty(this.values)) / length(this.values)
template <class IT, class NT>
NT CSR_FlengthCluster<IT,NT>::calculate_average_fill_factor(const NT eps)
{
  NT non_empty = 0.0;
//#pragma omp parallel for
  for (IT i = 0; i < rows; ++i)
  {
    for (IT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
      for (IT k = 0; k < cluster_sz; ++k) {
        if(fabs(values[(j * cluster_sz) + k] - 0.0f) >= eps) non_empty += 1;
      }
    }
  }
  if(nnzc > 0) non_empty /= (nnzc * cluster_sz);
  return non_empty;
}

//! calculate ratio of count(not_empty(this.values[cluster_id])) / length(this.values[cluster_id])
template <class IT, class NT>
NT CSR_FlengthCluster<IT,NT>::calculate_fill_factor(const IT cluster_id, const NT eps)
{
  NT non_empty = 0.0;
//#pragma omp parallel for
  for (IT j = rowptr[cluster_id]; j < rowptr[cluster_id + 1]; ++j) {
    for (IT k = 0; k < cluster_sz; ++k) {
      if(fabs(values[(j * cluster_sz) + k] - 0.0f) >= eps) non_empty += 1;
    }
  }
  IT num_cols = rowptr[cluster_id + 1] - rowptr[cluster_id];
  if(num_cols > 0) non_empty /= (num_cols * cluster_sz);
  return non_empty;
}

//! calculate size in Bytes
template <class IT, class NT>
double CSR_FlengthCluster<IT,NT>::calculate_size() {
  size_t total = 0;
  total += ((rows + 1) * sizeof(IT));             // rowptr
  total += (nnzc * sizeof(IT));                   // colids
  total += ((nnzc * cluster_sz) * sizeof(NT));    // values
  return static_cast<double>(total);
}

//! Print rows of CSR_FlengthCluster
template<class IT, class NT>
void CSR_FlengthCluster<IT, NT>::print_rows(IT start, IT end) {
  for (IT i = start; i < end; ++i) {
    cout << i << ":";
    for (IT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
      cout << " " << colids[j] << "(";
      for (IT k = 0; k < cluster_sz; k += 1) {
        cout << values[(j * cluster_sz) + k] << " ";
      }
      cout << ")";
    }
    cout << endl;
  }
}

#endif  //_CSR_FLENGTH_CLUSTER_H_
