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

//#include <unistd.h>
//#include <sys/ioctl.h>
//#include <linux/perf_event.h>
//#include <sys/syscall.h>

#include <filesystem>
namespace fs = std::filesystem;

#include "../utility.h"
#include "../CSR.h"
#include "../multiply.h"

#include "../hash_mult.h"
#include "sample_common.hpp"

using namespace std;

#define VALUETYPE double
//#define INDEXTYPE int
#define INDEXTYPE int64_t

int main(int argc, char* argv[])
{
    const bool sortOutput = false;
    vector<int> tnums;
	  CSR<INDEXTYPE,VALUETYPE> A_csr, B_csr;

	if (argc < 4) {
        cout << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|matrix1.txt} {scale|matrix2.txt} {edgefactor|product.txt} <numthreads>" << endl;
        return -1;
    }
    else if (argc < 6) {
        cout << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|matrix1.txt} {scale|matrix2.txt} {edgefactor|product.txt} <numthreads>" << endl;

#ifdef KNL_EXE
        cout << "Running on 68, 136, 204, 272 threads" << endl << endl;
        tnums = {68, 136, 204, 272};
        // tnums = {1, 2, 4, 8, 16, 32, 64, 68, 128, 136, 192, 204, 256, 272}; // for scalability test
#else
        cout << "Running on 32, 64 threads" << endl; 
        tnums = {64}; // for perlmutter
#endif
    }
	else {
        cout << "Running on " << argv[6] << " processors" << endl << endl;
        tnums = {atoi(argv[5])};
    }

    /* Generating input matrices based on argument */
    SetInputMatricesAsCSR(A_csr, B_csr, argv);

  	A_csr.Sorted();
  	B_csr.Sorted();

    INDEXTYPE cluster_size = 8;
    
    /* Count total number of floating-point operations */
    long long int nfop = get_flop(A_csr, B_csr);
    cout << "Total number of floating-point operations including addition and multiplication in SpGEMM (A * B): " << nfop << endl << endl;

    double start, end, msec, ave_msec, mflops;

    /* Execute Hash-SpGEMM */
    cout << "Evaluation of RowSpGEMM" << endl;
    for (int tnum : tnums) {
        omp_set_num_threads(tnum);
        
        CSR<INDEXTYPE,VALUETYPE> C_csr;
    
        /* First execution is excluded from evaluation */
        RowSpGEMM<false, sortOutput>(A_csr, B_csr, C_csr, multiplies<VALUETYPE>(), plus<VALUETYPE>(), "");
        C_csr.make_empty();

        ave_msec = 0;
        for (int i = 0; i < ITERS; ++i) {
            start = omp_get_wtime();
            RowSpGEMM<false, sortOutput>(A_csr, B_csr, C_csr, multiplies<VALUETYPE>(), plus<VALUETYPE>(), "");
            end = omp_get_wtime();
            msec = (end - start) * 1000;
            ave_msec += msec;
            if (i < ITERS - 1) {
                C_csr.make_empty();
            }
        }

        ave_msec /= ITERS;
        mflops = (double)nfop / ave_msec / 1000;

//        printf("RowSpGEMM returned with %d nonzeros. Compression ratio is %f\n", C_csr.nnz, (double)(nfop / 2) / (double)(C_csr.nnz));
        cout << "RowSpGEMM returned with " << C_csr.nnz << " nonzeros. Compression ratio is " << ((double)(nfop / 2) / (double)(C_csr.nnz)) << endl;
        printf("RowSpGEMM with %3d threads computes C = A * B in %f [milli seconds] (%f [MFLOPS])\n\n", tnum, ave_msec, mflops);

        C_csr.make_empty();
    }

    A_csr.make_empty();
    B_csr.make_empty();

    return 0;
}
