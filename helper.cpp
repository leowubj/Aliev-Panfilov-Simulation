/* 
 * Utilities for the Aliev-Panfilov code
 * Scott B. Baden, UCSD
 * Nov 2, 2015
 */

#include <iostream>
#include <assert.h>
// Needed for memalign
#include <malloc.h>

#include "cblock.h" // added for control_block
// #ifdef _MPI_
#include <mpi.h>
// #endif

using namespace std;

extern control_block cb; // use cb's info (Piazza post @239)

void printMat(const char mesg[], double *E, int m, int n);

void printSubMat(const char mesg[], double *E, int m, int n, int ny, int nx, int startIdx);


//
// Initialization
//
// We set the right half-plane of E_prev to 1.0, the left half plane to 0
// We set the botthom half-plane of R to 1.0, the top half plane to 0
// These coordinates are in world (global) coordinate and must
// be mapped to appropriate local indices when parallelizing the code
//
#define _MPI_ 1 // mpi will be defined in runtime regardless of this
#ifdef _MPI_
void init (double *E,double *E_prev,double *R,int m,int n){
    
    // printf("Initializing E and R\n");
    // if (cb.noComm) {
    //     printf("cb.noComm is true, no communication\n");
    // } else {
    //     printf("cb.noComm is false, use communication\n");
    // } 

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nx = cb.n / cb.px; 
    int ny = cb.m / cb.py; 

    int x_extra = cb.n % cb.px;
    int y_extra = cb.m % cb.py;
  
    
    if (rank == 0) { // only rank 0 know the full initial condition
        // int i, j; // declaring these index variable globally won't work, not sure why.
        // instead of allocating more memory locally , we specifically allocate more memory for rank 0 in main()
        // printf("rank 0, nx: %d, ny: %d, dataSize: %d, x_extra:%d, y_extra:%d \n", nx, ny, dataSize, x_extra, y_extra);

        double* initE_prev = (double*) memalign(16, sizeof(double)*(m+2)*(n+2));
        double* initR = (double*) memalign(16, sizeof(double)*(m+2)*(n+2));

        for (int i=0; i < (m+2)*(n+2); i++)
            initE_prev[i] = initR[i] = 0;

        // printf("i = %d\n", i);

        for (int i = (n+2); i < (m+1)*(n+2); i++) {
            int colIndex = i % (n+2);		// gives the base index (first row's) of the current index
            // Need to compute (n+1)/2 rather than n/2 to work with odd numbers
            if(colIndex == 0 || colIndex == (n+1) || colIndex < ((n+1)/2+1))
                continue;
            initE_prev[i] = 1.0;
        }

        // printf("i = %d\n", i);

        for (int i = 0; i < (m+2)*(n+2); i++) {
            int rowIndex = i / (n+2);		// gives the current row number in 2D array representation
            int colIndex = i % (n+2);		// gives the base index (first row's) of the current index
            // Need to compute (m+1)/2 rather than m/2 to work with odd numbers
            if(colIndex == 0 || colIndex == (n+1) || rowIndex < ((m+1)/2+1))
                continue;
            initR[i] = 1.0;
        }
        // printMat("Rank 0 initE_prev: \n", initE_prev, m, n);
        // printMat("Rank 0 initR: \n", initR, m, n);

        // initialize data in rank 0       
        for (int j = 0; j < ny+2; j++) {
            for (int i = 0; i < nx+2; i++) {
                E_prev[j*(nx+2) + i] = initE_prev[j*(cb.n+2) + i];
                R[j*(nx+2) + i] = initR[j*(cb.n+2) + i];
            }
        }
        // printSubMat("Rank 0 E_prev:", E_prev, m, n, ny, nx, 0);
        // printSubMat("Rank 0 R: \n", R, m, n, ny, nx, 0);

        // send initial data to other ranks 1, 2, ...
        MPI_Request requests[2 * (size-1)];
        for (int i = 1; i < size; i++) {
        
            // rank indices in 2D
            int curRankRowIdx = i / cb.px;
            int curRankColIdx = i % cb.px;

            int cur_nx = cb.n / cb.px, cur_ny = cb.m / cb.py;
            if (curRankColIdx < x_extra){
                cur_nx ++;
            }
            if (curRankRowIdx < y_extra){
                cur_ny ++;
            }     

            int I = curRankColIdx*nx + min(curRankColIdx, x_extra);
            int J = curRankRowIdx*ny + min(curRankRowIdx, y_extra);
            // printf("cb.py: %d, cb.px: %d, rank: %d, curRankRowIdx: %d, curRankColIdx: %d, I: %d, J: %d\n", cb.py, cb.px, i, curRankRowIdx, curRankColIdx, I, J);

            int curDataSize = (cur_ny+2) * (cur_nx+2);
            
            if (!cb.noComm) {
                // create a strided multi-block data type, then no need to pack
                // printf("cur_ny: %d, cur_nx: %d, cb.n: %d\n", cur_ny, cur_nx, cb.n);
                MPI_Datatype sendBuf;
                MPI_Type_vector(cur_ny+2 /* no. block */, cur_nx+2 /* block_size */, cb.n+2 /* stride */, MPI_DOUBLE, &sendBuf);
                MPI_Type_commit(&sendBuf);

                // printf("Sending initial data to rank %d\n", i);
                MPI_Isend(&initE_prev[J*(n+2)+I], 1, sendBuf, i, 1, MPI_COMM_WORLD, &requests[(i-1) * 2]);
                MPI_Isend(&initR[J*(n+2)+I], 1, sendBuf, i, 2, MPI_COMM_WORLD, &requests[(i-1)*2 + 1]);
                MPI_Type_free(&sendBuf);
                // printf("Sent initial data to rank %d\n", i);
            }
        }
        // may or may not need to call wait 
        if (!cb.noComm) {
            MPI_Waitall(2 * (size-1), requests, MPI_STATUS_IGNORE);
        }
    } else {
        int rankRowIdx = rank / cb.px;
        int rankColIdx = rank % cb.px;
        
        if (rankColIdx < x_extra){
            nx ++; // this rank has a 1 greater size in this dimension
        }
        if (rankRowIdx < y_extra){
            ny ++;
        } 
        int dataSize = (nx+2) * (ny+2);
        // receive initial data from other ranks 1, 2, ...
        if (!cb.noComm) {
            MPI_Request requests[2];
            MPI_Irecv(E_prev, dataSize, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &requests[0]);
            // received data guaranted to be in order by MPI
            MPI_Irecv(R, dataSize, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &requests[1]);
            
            MPI_Waitall(2, requests, MPI_STATUS_IGNORE);

            // printf("Rank = %d\n", rank);
            // printMat("E_prev: \n", E_prev, ny, nx);
            // printMat("R: \n", R, ny, nx);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    
    // We only print the meshes if they are small enough
#if 1
    // printMat("E_prev",E_prev,m,n);
    // printMat("R",R,m,n);
#endif
}
#else
// original init() method
void init (double *E,double *E_prev,double *R,int m,int n){
    int i;
    // printf("Initializing E and R\n");
    for (i=0; i < (m+2)*(n+2); i++)
        E_prev[i] = R[i] = 0;

    for (i = (n+2); i < (m+1)*(n+2); i++) {
	    int colIndex = i % (n+2);		// gives the base index (first row's) of the current index
        // Need to compute (n+1)/2 rather than n/2 to work with odd numbers
	    if(colIndex == 0 || colIndex == (n+1) || colIndex < ((n+1)/2+1))
	        continue;
        E_prev[i] = 1.0;
    }

    for (i = 0; i < (m+2)*(n+2); i++) {
        int rowIndex = i / (n+2);		// gives the current row number in 2D array representation
        int colIndex = i % (n+2);		// gives the base index (first row's) of the current index
        // Need to compute (m+1)/2 rather than m/2 to work with odd numbers
	    if(colIndex == 0 || colIndex == (n+1) || rowIndex < ((m+1)/2+1))
	        continue;
        R[i] = 1.0;
    }
    // We only print the meshes if they are small enough
#if 1
    printMat("E_prev",E_prev,m,n);
    printMat("R",R,m,n);
#endif
}
#endif

#ifdef _MPI_
double *alloc1D(int m,int n) {

    // n = cb.n + 2, m = cb.m + 2
    // int nx = (cb.n+cb.px-1) / cb.px, ny = (cb.m+cb.py-1) / cb.py; 
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double *E;

    if (rank == 0) {
        // allocate the size of a whole matrix for E_prev, E, R
        // Ensures that allocatdd memory is aligned on a 16 byte boundary
        assert(E= (double*) memalign(16, sizeof(double)*m*n) );
        return(E);
    }
    int nx = cb.n / cb.px; 
    int ny = cb.m / cb.py; 

    int x_extra = cb.n % cb.px;
    int y_extra = cb.m % cb.py;

    int rankRowIdx = rank / cb.px;
    int rankColIdx = rank % cb.px;

    if (rankColIdx < x_extra){
        nx ++; // this rank has a 1 greater size in this dimension
    }
    if (rankRowIdx < y_extra){
        ny ++;
    }   

    assert(E= (double*) memalign(16, sizeof(double)*(nx+2)*(ny+2)) );
    return(E);
}
#else
// original alloc1D() method
double *alloc1D(int m,int n) {
    int nx=n, ny=m;
    double *E;
    // Ensures that allocatdd memory is aligned on a 16 byte boundary
    assert(E= (double*) memalign(16, sizeof(double)*nx*ny) );
    return(E);
}
#endif

void printMat(const char mesg[], double *E, int m, int n){
    if (m > 34 || n > 34) return;
    int i;
    printf("%s\n",mesg);
    for (i=0; i < (m+2)*(n+2); i++){
        int rowIndex = i / (n+2);
        int colIndex = i % (n+2);
        // if ((colIndex>0) && (colIndex<n+1))
        //     if ((rowIndex > 0) && (rowIndex < m+1))
                printf("%6.3f ", E[i]);
        if (colIndex == n+1)
	        printf("\n");
    }
}

void printSubMat(const char mesg[], double *E, int m, int n, int ny, int nx, int startIdx) {
    if (m > 34 || n > 34) return;
    int i;
    printf("%s\n",mesg);
    for (i=0; i < (ny+2)*(nx+2); i++){
        int rowIndex = i / (nx+2);
        int colIndex = i % (nx+2);
            printf("%6.3f ", E[startIdx+colIndex+rowIndex*(n+2)]);
        if (colIndex == nx+1)
	        printf("\n");
    }
}
