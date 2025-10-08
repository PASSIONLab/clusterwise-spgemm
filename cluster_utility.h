#ifndef _CLUSTER_UTILITY_H
#define _CLUSTER_UTILITY_H

#include <stdlib.h>
#include <stdint.h>
#include <climits>
#include <iostream>
#include <cmath>
#include <vector>
#include <map>
#include <fstream>

#include "CSR.h"
#include "CSR_FlengthCluster.h"
#include "CSR_VlengthCluster.h"

#include <omp.h>

using namespace std;

//const double eps = 0.0000001f;
const double eps = 0.000001f;

//! Construct a CSR object from a CSR_VlengthCluster
//! Convert @csr_flength_cluster to @csr
template<class IT, class NT>
void csr_vlength_cluster2csr(CSR<IT, NT> &csr, const CSR_VlengthCluster<IT, NT> &csr_flength_cluster) {
  csr.nnz = 0;
  csr.rows = csr_flength_cluster.csr_rows;
  csr.cols = csr_flength_cluster.cols;
  csr.zerobased = true;

  IT *work = my_malloc<IT>(csr.rows);     // store number of columns per-row
  std::fill(work, work + csr.rows, (IT) 0);        // initialized to zero

  IT row_id = 0, val_idx;
  NT val;
  // count the nnz per CSR.row and use this count to allocate memory for CSR (similar to symbolic phase of SpGEMM)
  for (IT i = 0; i < csr_flength_cluster.rows; i++) {                                     // loop over clusters
    for (IT j = csr_flength_cluster.rowptr[i]; j < csr_flength_cluster.rowptr[i + 1]; j++) {      // loop over columns of cluster-i
      // values of cluster[i] starts at rowptr_val[i]
      // for every column j (0-based index), values will be stored on {(j - rowptr[i]) * cluster_sz[i]}
      val_idx = csr_flength_cluster.rowptr_val[i] + ((j - csr_flength_cluster.rowptr[i]) * csr_flength_cluster.cluster_sz[i]);
      for (IT l = 0; l < csr_flength_cluster.cluster_sz[i]; l += 1) {                     // loop over the values (e.g., CSR.rows) of cluster-i
        val = csr_flength_cluster.values[val_idx + l];                                    // value of CSR.row[row_id]
        if (fabs(val - 0.0f) >= eps) {
          work[row_id + l]++;                                                     // row counts (i.e, work holds the "row difference array")
          csr.nnz += 1;
        }
      }
    }
    row_id += csr_flength_cluster.cluster_sz[i];
  }

  csr.rowptr = my_malloc<IT>(csr.rows + 1);
  csr.colids = my_malloc<IT>(csr.nnz);
  csr.values = my_malloc<NT>(csr.nnz);

  if (csr.nnz > 0) {
    csr.rowptr[csr.rows] = CumulativeSum(work, csr.rows);    // cumulative sum of work
    copy(work, work + csr.rows, csr.rowptr);

    IT col_id, last;
    row_id = 0;
    // actually saving data in CSR
    for (IT i = 0; i < csr_flength_cluster.rows; i++) {                                           // loop over clusters
      for (IT j = csr_flength_cluster.rowptr[i]; j < csr_flength_cluster.rowptr[i + 1]; j++) {            // loop over columns of cluster-i
        col_id = csr_flength_cluster.colids[j];
        // values of cluster[i] starts at rowptr_val[i]
        // for every column j (0-based index), values will be stored on {(j - rowptr[i]) * cluster_sz[i]}
        val_idx = csr_flength_cluster.rowptr_val[i] + ((j - csr_flength_cluster.rowptr[i]) * csr_flength_cluster.cluster_sz[i]);
        for (IT l = 0; l < csr_flength_cluster.cluster_sz[i]; l += 1) {                           // loop over the values of col_id
          val = csr_flength_cluster.values[val_idx + l];
          if (fabs(val - 0.0f) >= eps) {
            csr.colids[last = work[row_id + l]++] = col_id;
            csr.values[last] = val;
          }
        }
      }
      row_id += csr_flength_cluster.cluster_sz[i];
    }
  } else {
    cout << "nnz should not be 0" << endl;
    exit(-1);
  }
  my_free<IT>(work);
}

//! Construct a CSR object from a CSR_FlengthCluster
//! Convert @csr_flength_cluster to @csr
template<class IT, class NT>
void csr_flength_cluster2csr(CSR<IT, NT> &csr, const CSR_FlengthCluster<IT, NT> &csr_flength_cluster) {
  csr.nnz = 0;
  csr.rows = csr_flength_cluster.csr_rows;
  csr.cols = csr_flength_cluster.cols;
  csr.zerobased = true;

  IT *work = my_malloc<IT>(csr.rows);     // store number of columns per-row
  std::fill(work, work + csr.rows, (IT) 0);        // initialized to zero

  IT row_id, col_id, last;
  NT val;
  // count the nnz per CSR.row and use this count to allocate memory for CSR (similar to symbolic phase of SpGEMM)
  for (IT i = 0; i < csr_flength_cluster.rows; i++) {                                     // loop over clusters
    for (IT j = csr_flength_cluster.rowptr[i]; j < csr_flength_cluster.rowptr[i + 1]; j++) {      // loop over columns of cluster-i
      for (IT l = 0; l < csr_flength_cluster.cluster_sz; l += 1) {                        // loop over the values (e.g., CSR.rows) of cluster-i
        row_id = (i * csr_flength_cluster.cluster_sz) + l;
        val = csr_flength_cluster.values[(j * csr_flength_cluster.cluster_sz) + l];               // value of CSR.row[row_id]
        if (fabs(val - 0.0f) >= eps) {
          work[row_id]++;                                                         // row counts (i.e, work holds the "row difference array")
          csr.nnz += 1;
        }
      }
    }
  }

  csr.rowptr = my_malloc<IT>(csr.rows + 1);
  csr.colids = my_malloc<IT>(csr.nnz);
  csr.values = my_malloc<NT>(csr.nnz);

  if (csr.nnz > 0) {
    csr.rowptr[csr.rows] = CumulativeSum(work, csr.rows);    // cumulative sum of work
    copy(work, work + csr.rows, csr.rowptr);

    IT last;
    // actually saving data in CSR
    for (IT i = 0; i < csr_flength_cluster.rows; i++) {                                     // loop over clusters
      for (IT j = csr_flength_cluster.rowptr[i]; j < csr_flength_cluster.rowptr[i + 1]; j++) {      // loop over columns of cluster-i
        col_id = csr_flength_cluster.colids[j];
        for (IT l = 0; l < csr_flength_cluster.cluster_sz; l += 1) {                        // loop over the values of col_id
          row_id = (i * csr_flength_cluster.cluster_sz) + l;
          val = csr_flength_cluster.values[(j * csr_flength_cluster.cluster_sz) + l];
          if (fabs(val - 0.0f) >= eps) {
            csr.colids[last = work[row_id]++] = col_id;
            csr.values[last] = val;
          }
        }
      }
    }
  } else {
    cout << "nnz should not be 0" << endl;
    exit(-1);
  }

  my_free<IT>(work);
}

#endif    //_CLUSTER_UTILITY_H

