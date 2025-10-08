# Cluster-wise SpGEMM
## Introduction
Sparse matrix-sparse matrix multiplication (SpGEMM) is a key kernel in many scientific applications and graph workloads. Unfortunately, SpGEMM is bottlenecked by data movement due to its irregular memory access patterns. We address these issues with `hierarchical clustering` for SpGEMM that leverages both row reordering and cluster-wise computation to improve reuse in the second input (B) matrix with a novel row-clustered matrix format and access pattern in the first input (A) matrix.

This library provides a shared memory parallel implementation of different Cluster-wise SpGEMM kernels. For more information, please read our preprint paper ["Improving SpGEMM Performance Through Matrix Reordering and Cluster-wise Computation"](https://arxiv.org/abs/2507.21253) on arXiv.

## Requirements
### Hardware
The experiments were executed using one CPU node on the Perlmutter supercomputer at NERSC, utilizing 64 threads. Each Perlmutter CPU node consists of two AMD EPYC 7763 (Milan) processors, with a total of 512 GB DDR4 memory. Each processor features 64 cores, 204.8 GB/s memory bandwidth, a 64 MiB L2 cache, and access to the full memory pool. Similar results are expected on other hardware platforms.

### Software
The following software is required to compile and run the code (mentioned versions are those used in our experiments):
* Intel C++ Compiler (`icpc`) (version 2024.1.0)
* OpenMP
* Intel Threading Building Blocks (TBB) (version 2019.9)
* GNU Make (version 4.2.1)

## Using Cluster-wise SpGEMM
All the SpGEMM implementations use `hashtable` as the sparse accumulator.

### SpGEMM Kernels
- *hash_mult.h*: Row-wise SpGEMM implementation.
- *hash_mult_flengthcluster.h*: Fixed-length cluster-wise SpGEMM implementation.
- *hash_mult_vlengthcluster.h*: Variable-length cluster-wise SpGEMM implementation.

### Data structure
- *CSC.h*: Compressed Sparse Column format implementation
- *CSR.h*: Compressed Sparse Row format implementation
- *CSR_FlengthCluster.h*: `CSR_Cluster` format implementation for fixed-length clusters
- *CSR_VlengthCluster.h*: `CSR_Cluster` format implementation for variable-length clusters
- *BIN.h*: the data structure for managing load balance in row-wise SpGEMM
- *BIN_FlengthCluster.h*: the data structure for managing load balance in fixed-length cluster-wise SpGEMM
- *BIN_VlengthCluster.h*: the data structure for managing load balance in variable-length cluster-wise SpGEMM

### Sample codes
*Files under `sample` directory*: Sample codes with main function reading matrix data files, and then compute SpGEMM.

| **Implementation**            | **Description**                                                     | **Input Parameter**                                                                          |
|-------------------------------|---------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| RowSpGEMM                     | Row-wise SpGEMM                                                     | `text <matrix-A> <matrix-B>`                                                                   |
| ReorderedRowSpGEMM            | Row-wise SpGEMM (with reordering)                                   | `text <matrix-A> <row-ordering of matrix-A>`                                                   |
| FlengthClusterSpGEMM          | Fixed-length cluster-wise SpGEMM                                    | `text <matrix-A> <matrix-B>`                                                                   |
| ReorderedFlengthClusterSpGEMM | Fixed-length cluster-wise SpGEMM (with reordering)                  | `text <matrix-A> <row-ordering of matrix-A>`                                                   |
| VlengthClusterSpGEMM          | Variable-length cluster-wise SpGEMM                                 | `text <matrix-A> <matrix-B>`                                                                   |
| ReorderedVlengthClusterSpGEMM | Variable-length cluster-wise SpGEMM (with reordering)               | `text <matrix-A> <row-ordering of matrix-A>`                                                   |
| HierarchicalClusterSpGEMM     | Hierarchical cluster-wise SpGEMM                                    | `text <matrix-A> <matrix-B> <candidate close-pairs in matrix-A>`                               |
| tsRowSpGEMM                   | Row-wise SpGEMM for tall-skinny matrix                              | `text <matrix-A> <directory path of tall-skinny matrix-B>`                                     |
| tsReorderedRowSpGEMM          | Row-wise SpGEMM (with reordering) for tall-skinny matrix            | `text <matrix-A> <row-ordering of matrix-A> <directory path of tall-skinny matrix-B>`          |
| tsHierarchicalClusterSpGEMM   | Hierarchical cluster-wise SpGEMM for tall-skinny matrix             | `text <matrix-A> <directory path of tall-skinny matrix-B> <candidate close-pairs in matrix-A>` |

Two utility codes are also placed in the `sample` directory. These code is used to generate candidate close-pairs and randomly shuffle the row-ids of an input matrix.

| **Implementation**            | **Description**                                                     | **Input Parameter**                                                                          |
|-------------------------------|---------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| GenerateCandidatePairs        | Candidate close-pair generator using AxA<sup>T</sup> (keep top-k per-row) | `text <matrix-A> <matrix-A> <top-k> -s`                                                        |
| ShuffleRowIDs                 | Randomly shuffling the row-ids of a input matrix                          | `text <matrix-A> -s`                                                                           |

### Performance Tip
We recommend configuring your environment for OpenMP based on your hardware by setting the environment variables, including but not limited to `OMP_NUM_THREADS`, `OMP_PROC_BIND`, and `OMP_PLACES`.

### How to use in your program
Copy all the header files (*.h) in this directory to your program. Below is an example of how to use the hierarchical cluster-wise SpGEMM in your program:

```cpp
#include "utility.h"
#include "cluster_utility.h"
#include "multiply.h"
#include "CSR.h"
#include "CSR_VlengthCluster.h"
#include "hash_mult_vlengthcluster.h"
```

## Replicating Our Benchmarks
###Get the reordering code:
```
> git clone https://github.com/PASSIONLab/clusterwise-spgemm.git
> export CLUSTERWISE_ROOT=$(pwd)/clusterwise-spgemm

> cd $CLUSTERWISE_ROOT
```
### Build the code:
Before compiling the code, first modify the `makefile` with correct path to Intel Compiler.
```
> make clean && make sample_hw
```
### Get the datasets:
If you haven’t downloaded the datasets yet, visit [reordering-spgemm](https://github.com/PASSIONLab/reordering-spgemm) for instructions on downloading them and generating different matrix reorderings.

```
> export DATA_PATH=/pscratch/sd/r/raqib/dataset-spgemm
```

Get random reordering by shuffling the order of rows:
```
> export SHUFFLED_DATA_PATH=$DATA_PATH/reordering/shuffle_order
> mkdir -p $SHUFFLED_DATA_PATH
> sh scripts/reordering/reorder_random.sh > scripts/reordering/reorder_random.out
```

Validate if all the reordering paths are correctly set in the environment.
```
> export RABBIT_DATA_PATH=$DATA_PATH/reordering/rabbit_order
> export AMD_DATA_PATH=$DATA_PATH/reordering/amd_order
> export RCM_DATA_PATH=$DATA_PATH/reordering/rcm_order
> export ND_DATA_PATH=$DATA_PATH/reordering/nd_order
> export GP_DATA_PATH=$DATA_PATH/reordering/gp_order
> export HP_DATA_PATH=$DATA_PATH/reordering/hp_order
> export GRAY_DATA_PATH=$DATA_PATH/reordering/gray_order
> export DEGREE_DATA_PATH=$DATA_PATH/reordering/degree_order
> export SLASHBURN_DATA_PATH=$DATA_PATH/reordering/slashburn_order
```

### Run benchmark on row-wise SpGEMM:
```
> sh scripts/rowwise/1_rowwise_original.sh > scripts/rowwise/1_rowwise_original.out
> sh scripts/rowwise/2_rowwise_random.sh > scripts/rowwise/2_rowwise_random.out
> sh scripts/rowwise/3_rowwise_rabbit.sh > scripts/rowwise/3_rowwise_rabbit.out
> sh scripts/rowwise/4_rowwise_amd.sh > scripts/rowwise/4_rowwise_amd.out
> sh scripts/rowwise/5_rowwise_rcm.sh > scripts/rowwise/5_rowwise_rcm.out
> sh scripts/rowwise/6_rowwise_nd.sh > scripts/rowwise/6_rowwise_nd.out
> sh scripts/rowwise/7_rowwise_gp.sh > scripts/rowwise/7_rowwise_gp.out
> sh scripts/rowwise/8_rowwise_hp.sh > scripts/rowwise/8_rowwise_hp.out
> sh scripts/rowwise/9_rowwise_gray.sh > scripts/rowwise/9_rowwise_gray.out
> sh scripts/rowwise/10_rowwise_degree.sh > scripts/rowwise/10_rowwise_degree.out
> sh scripts/rowwise/11_rowwise_slashburn.sh > scripts/rowwise/11_rowwise_slashburn.out
```

### Run benchmark on fixed-length cluster-wise SpGEMM:
```
> sh scripts/flength_clusterwise/1_flength_original.sh > scripts/flength_clusterwise/1_flength_original.out
> sh scripts/flength_clusterwise/2_flength_random.sh > scripts/flength_clusterwise/2_flength_random.out
> sh scripts/flength_clusterwise/3_flength_rabbit.sh > scripts/flength_clusterwise/3_flength_rabbit.out
> sh scripts/flength_clusterwise/4_flength_amd.sh > scripts/flength_clusterwise/4_flength_amd.out
> sh scripts/flength_clusterwise/5_flength_rcm.sh > scripts/flength_clusterwise/5_flength_rcm.out
> sh scripts/flength_clusterwise/6_flength_nd.sh > scripts/flength_clusterwise/6_flength_nd.out
> sh scripts/flength_clusterwise/7_flength_gp.sh > scripts/flength_clusterwise/7_flength_gp.out
> sh scripts/flength_clusterwise/8_flength_hp.sh > scripts/flength_clusterwise/8_flength_hp.out
> sh scripts/flength_clusterwise/9_flength_gray.sh > scripts/flength_clusterwise/9_flength_gray.out
> sh scripts/flength_clusterwise/10_flength_degree.sh > scripts/flength_clusterwise/10_flength_degree.out
> sh scripts/flength_clusterwise/11_flength_slashburn.sh > scripts/flength_clusterwise/11_flength_slashburn.out
```

### Run benchmark on variable-length cluster-wise SpGEMM:
```
> sh scripts/vlength_clusterwise/1_vlength_original.sh > scripts/vlength_clusterwise/1_vlength_original.out
> sh scripts/vlength_clusterwise/2_vlength_random.sh > scripts/vlength_clusterwise/2_vlength_random.out
> sh scripts/vlength_clusterwise/3_vlength_rabbit.sh > scripts/vlength_clusterwise/3_vlength_rabbit.out
> sh scripts/vlength_clusterwise/4_vlength_amd.sh > scripts/vlength_clusterwise/4_vlength_amd.out
> sh scripts/vlength_clusterwise/5_vlength_rcm.sh > scripts/vlength_clusterwise/5_vlength_rcm.out
> sh scripts/vlength_clusterwise/6_vlength_nd.sh > scripts/vlength_clusterwise/6_vlength_nd.out
> sh scripts/vlength_clusterwise/7_vlength_gp.sh > scripts/vlength_clusterwise/7_vlength_gp.out
> sh scripts/vlength_clusterwise/8_vlength_hp.sh > scripts/vlength_clusterwise/8_vlength_hp.out
> sh scripts/vlength_clusterwise/9_vlength_gray.sh > scripts/vlength_clusterwise/9_vlength_gray.out
> sh scripts/vlength_clusterwise/10_vlength_degree.sh > scripts/vlength_clusterwise/10_vlength_degree.out
> sh scripts/vlength_clusterwise/11_vlength_slashburn.sh > scripts/vlength_clusterwise/11_vlength_slashburn.out
```

### Generate candidate close-pairs for hierarchical cluster-wise SpGEMM:
```
> export CLOSE_PAIR_DATA_PATH=$DATA_PATH/reordering/close_pairs
> mkdir -p $CLOSE_PAIR_DATA_PATH
> sh scripts/h_clusterwise/1_generate_close_pairs.sh > scripts/h_clusterwise/1_generate_close_pairs.out
```
### Run benchmark on hierarchical cluster-wise SpGEMM:
```
> sh scripts/h_clusterwise/2_hierarchical_spgemm.sh > scripts/h_clusterwise/2_hierarchical_spgemm.out
```

### Run benchmark on Tall-skinny matrices:
If the tall-skinny matrices haven’t been generated yet, follow the steps below to produce them from the BC frontiers output by CombBLAS.

#### Tall-skinny Row-wise SpGEMM:
```
> sh scripts/tall_skinny/rowwise/1_ts_original.sh > scripts/tall_skinny/rowwise/1_ts_original.out
> sh scripts/tall_skinny/rowwise/2_ts_random.sh > scripts/tall_skinny/rowwise/2_ts_random.out
> sh scripts/tall_skinny/rowwise/3_ts_rabbit.sh > scripts/tall_skinny/rowwise/3_ts_rabbit.out
> sh scripts/tall_skinny/rowwise/4_ts_amd.sh > scripts/tall_skinny/rowwise/4_ts_amd.out
> sh scripts/tall_skinny/rowwise/5_ts_rcm.sh > scripts/tall_skinny/rowwise/5_ts_rcm.out
> sh scripts/tall_skinny/rowwise/6_ts_nd.sh > scripts/tall_skinny/rowwise/6_ts_nd.out
> sh scripts/tall_skinny/rowwise/7_ts_gp.sh > scripts/tall_skinny/rowwise/7_ts_gp.out
> sh scripts/tall_skinny/rowwise/8_ts_hp.sh > scripts/tall_skinny/rowwise/8_ts_hp.out
> sh scripts/tall_skinny/rowwise/9_ts_gray.sh > scripts/tall_skinny/rowwise/9_ts_gray.out
> sh scripts/tall_skinny/rowwise/10_ts_degree.sh > scripts/tall_skinny/rowwise/10_ts_degree.out
> sh scripts/tall_skinny/rowwise/11_ts_slashburn.sh > scripts/tall_skinny/rowwise/11_ts_slashburn.out
```

#### Tall-skinny Cluster-wise SpGEMM:
```
> sh scripts/tall_skinny/h_clusterwise/1_ts_hierarchical_spgemm.sh > sh scripts/tall_skinny/h_clusterwise/1_ts_hierarchical_spgemm.out
```

## Generating Tall-skinny Matrix
In our experiments, we generate the tall-skinny matrices from BFS frontiers produced by [CombBLAS](https://github.com/PASSIONLab/CombBLAS) during Betweenness Centrality (BC) computations. As the number of frontiers varies among datasets, we only take the first 10 forward frontier matrices.

Here is the example code to add in the [CombBLAS/Applications/BetwCent.cpp](https://github.com/PASSIONLab/CombBLAS/blob/master/Applications/BetwCent.cpp) to save the first 10 forward frontiers:

```cpp
SpParHelper::Print("Exploring via BFS...\n");
int iter = 0;
while( fringe.getnnz() > 0 )
{
    nsp += fringe;
    Dist<bool>::MPI_DCCols * level = new Dist<bool>::MPI_DCCols( fringe ); 
    bfs.push_back(level);
    
    if(iter < 10) {
      string frontier_filename = 
          frontier_dir + "/" + string("batch_") + std::to_string(i) + 
          string("_forwarditer_") + std::to_string(iter) + string(".mtx");
      fringe.ParallelWriteMM(frontier_filename, true);
      iter++;
    }

    fringe = PSpGEMM<PTBOOLINT>(AT, fringe);
    fringe = EWiseMult(fringe, nsp, true);
}
```

The full implementation for saving the first ten frontier matrices (forward and backward) is available in the following commits: [forward](https://github.com/biqar/CombBLAS/commit/639ff674233a8b0e890d8379c5522ac8ccb6eeb8) and [backward](https://github.com/biqar/CombBLAS/commit/1c816ecc317442c45a3d51f82e9ef57038de9a4f).

To generate all the tall-skinny matrices used in our experiments, follow these steps:
```
# Step-1: Download the CombBLAS library (with the saving forward frontier matrices code)
> git clone https://github.com/biqar/CombBLAS.git
> export COMBBLAS_ROOT=$(pwd)/CombBLAS
> cd $COMBBLAS_ROOT

# Step-2: Checkout to the commit that contains the saving frontier matrices code
> git checkout save_forward_frontier_mtx

# Step-3: Compile and Install CombBLAS
> mkdir _build
> mkdir _install
> cd _build
> cmake .. -DCMAKE_INSTALL_PREFIX=../_install
> make
> make install

# Step-4: Generate the tall-skinny matrix for the input graphs used in our experiments
> export TS_DATA_PATH=$DATA_PATH/tall_skinny
> mkdir -p $TS_DATA_PATH
> sh scripts/tall_skinny/generator/1_save_bc_frontier_matrices.sh
```

## Citation and Acknowledgments
If you find this code useful, please cite our paper:
```
@article{cluster-wise-spgemm,
  title={Improving SpGEMM Performance Through Matrix Reordering and Cluster-wise Computation},
  author={Islam, Abdullah Al Raqibul and Xu, Helen and Dai, Dong and Bulu{\c{c}}, Ayd{\i}n},
  journal={arXiv preprint arXiv:2507.21253},
  year={2025}
}
```
