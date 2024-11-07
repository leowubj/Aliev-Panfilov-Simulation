/* 
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 * 
 * Modified and  restructured by Scott B. Baden, UCSD
 * 
 */

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "time.h"
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"
#include <emmintrin.h> // SSE2
#include <mpi.h> //
using namespace std;

void repNorms(double l2norm, double mx, double dt, int m,int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void printMat2(const char mesg[], double *E, int m, int n);
void printSubMat2(const char mesg[], double *E, int m, int n, int ny, int nx, int startIdx, int rank);

extern control_block cb;

#define TOP 1
#define BOT 2
#define LEFT 3
#define RIGHT 4

// #ifdef SSE_VEC
// If you intend to vectorize using SSE instructions, you must
// disable the compiler's auto-vectorizer
// __attribute__((optimize("no-tree-vectorize")))
// #endif 

// The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//
double L2Norm(double sumSq){
    double l2norm = sumSq /  (double) ((cb.m)*(cb.n));
    l2norm = sqrt(l2norm);
    return l2norm;
}

#define _MPI_ 1 // redundant since already set in makefile and compiled in runtime
#ifdef _MPI_ // method using MPI
void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf){
    // Simulated time is different from the integer timestep number
    double t = 0.0;

    double *E = *_E, *E_prev = *_E_prev;
    double *R_tmp = R;
    double *E_tmp = *_E;
    double *E_prev_tmp = *_E_prev;
    double mx, sumSq;
    int niter;
    // int m = cb.m, n=cb.n;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int n = cb.n / cb.px; 
    int m = cb.m / cb.py; 

    int x_extra = cb.n % cb.px;
    int y_extra = cb.m % cb.py;

    int rankRowIdx = rank / cb.px;
    int rankColIdx = rank % cb.px;

    if (rankColIdx < x_extra){
        n ++; // this rank has a 1 greater size in this dimension
    }
    if (rankRowIdx < y_extra){
        m ++;
    }    
    
    // int innerBlockRowStartIndex, innerBlockEndLeftIndex;
    int innerBlockRowStartLeftIndex, innerBlockRowEndLeftIndex;
    int innerBlockRowStartRightIndex, innerBlockRowEndRightIndex;
    int outerBlockRowEndLeftIndex, outerBlockRowEndRightIndex;
    int rowLen;
    innerBlockRowStartLeftIndex = (n+2)+1;
    innerBlockRowEndLeftIndex = (((m+2)*(n+2) - 1) - (n)) - (n+2);
    innerBlockRowStartRightIndex = n+2+n;
    innerBlockRowEndRightIndex = (((m+2)*(n+2) - 1) - 1) - (n+2);
    outerBlockRowEndLeftIndex = innerBlockRowEndLeftIndex + (n+2); // one row below
    outerBlockRowEndRightIndex = innerBlockRowEndRightIndex + (n+2);
    rowLen = n+2;
    
    // printf("rank %d, innerBlockRowStartLeftIndex: %d, innerBlockRowEndLeftIndex: %d, innerBlockRowStartRightIndex: %d, outerBlockRowStartLeftIndex: %d, outerBlockRowEndLeftIndex: %d\n", rank, innerBlockRowStartLeftIndex, innerBlockRowEndLeftIndex, innerBlockRowStartRightIndex, outerBlockRowStartLeftIndex, outerBlockRowEndLeftIndex);

    int topNeighborRank = rank >= cb.px ? rank - cb.px : -1;
    int bottomNeighborRank = rank + cb.px < size ? rank + cb.px : -1;
    int leftNeighborRank = rank % cb.px != 0 ? rank - 1 : -1;
    int rightNeighborRank = (rank+1) % cb.px != 0 ? rank + 1 : -1;
    // printf("rank: %d, topNeighborRank: %d, bottomNeighborRank: %d, leftNeighborRank: %d, rightNeighborRank: %d\n", 
    //     rank, topNeighborRank, bottomNeighborRank, leftNeighborRank , rightNeighborRank);

    MPI_Datatype sendBuf; // custom data type to send strided blocks
    MPI_Type_vector(m, 1, n+2, MPI_DOUBLE, &sendBuf);
    MPI_Type_commit(&sendBuf);

    // if (rank == 0) {
    //     printf("rank is %d\n", rank);
    //     printSubMat2("Printing E_prev before all iterations", E_prev, cb.m, cb.n, m, n, outerBlockRowStartLeftIndex-1, rank);
    //     printSubMat2("Printing R before all iterations", R, cb.m, cb.n, m, n, outerBlockRowStartLeftIndex-1, rank);
    // }

    // if (rank == 0) {
    //     // clear out cells that are not under the computation of rank 0
    //     for (int j = 0; j < cb.m+2; j++) {
    //         for (int i = 0; i < cb.n+2; i++) {
    //             if (i != 0 && j != 0 && i != n+1 && j != m+1 && j < m+2 && i < n+2) continue; // exclude boundary and extra cells
    //             E_prev[j*(cb.n+2)+i] = 0;
    //         }
    //     }
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
    // if (rank == 0) printMat2("Printing E_prev for rank 0 before all iterations", E_prev, cb.m, cb.n);
    // if (rank == 0) printMat2("Printing R for rank 0 before all iterations", R, cb.m, cb.n);

    // We continue to sweep over the mesh until the simulation has reached the desired number of iterations
    for (niter = 0; niter < cb.niters; niter++){
        if (cb.debug && (niter==0)){
            stats(E_prev,m,n,&mx,&sumSq);
            double l2norm = L2Norm(sumSq);
            repNorms(l2norm,mx,dt,m,n,-1, cb.stats_freq);
            if (cb.plot_freq)
                plotter->updatePlot(E,  -1, m+1, n+1);
        }

        /* 
        * Copy data from boundary of the computational box to the padding region, set up for differencing computational box's boundary.
        * These are physical boundary conditions, and are not to be confused with ghost cells that we would use in an MPI implementation.
        * The reason why we copy boundary conditions is to avoid computing single sided differences at the boundaries which increase the running time of solve().
        */
        // 4 FOR LOOPS set up the padding needed for the boundary conditions
        // int i,j;
        // int ct1 = 0, ct2 = 0, ct3 = 0, ct4 = 0;
        MPI_Request requests[8] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};

        if (!cb.noComm) {

            if (topNeighborRank != -1) { // Not a cell at the top boundary

                MPI_Isend(&E_prev[innerBlockRowStartLeftIndex], n, MPI_DOUBLE, topNeighborRank, TOP, MPI_COMM_WORLD, &requests[0]);
                MPI_Irecv(&E_prev[innerBlockRowStartLeftIndex-rowLen], n, MPI_DOUBLE, topNeighborRank, BOT, MPI_COMM_WORLD, &requests[1]);

            } else {
                // Fills in the TOP Ghost Cells
                for (int i = 1; i < n+1; i++) {
                    E_prev[i] = E_prev[i + rowLen*2];
                    // ct1 += 1;
                }
            }

            if (bottomNeighborRank != -1) { // Not a cell at the bottom boundary
               
                MPI_Isend(&E_prev[innerBlockRowEndLeftIndex], n, MPI_DOUBLE, bottomNeighborRank, BOT, MPI_COMM_WORLD, &requests[2]);
                MPI_Irecv(&E_prev[outerBlockRowEndLeftIndex], n, MPI_DOUBLE, bottomNeighborRank, TOP, MPI_COMM_WORLD, &requests[3]);

            } else { // Fills in the BOTTOM Ghost Cells
                for (int i = outerBlockRowEndLeftIndex; i < outerBlockRowEndLeftIndex+n; i++) {
                    E_prev[i] = E_prev[i - rowLen*2];
                    // ct2 += 1;
                    // if (niter == 0) printf("%d, n: %d\n", ct2, n);
                }
            }

            if (leftNeighborRank != -1) { // Not a cell at the left boundary

                MPI_Isend(&E_prev[innerBlockRowStartLeftIndex], 1, sendBuf, leftNeighborRank, LEFT, MPI_COMM_WORLD, &requests[4]);
                MPI_Irecv(&E_prev[innerBlockRowStartLeftIndex-1], 1, sendBuf, leftNeighborRank, RIGHT, MPI_COMM_WORLD, &requests[5]);

            } else { // Fills in the LEFT Ghost Cells
                for (int i = innerBlockRowStartLeftIndex-1; i < outerBlockRowEndLeftIndex-1; i+=rowLen) {
                    E_prev[i] = E_prev[i+2];
                    // ct3 += 1;
                }	
            }

            if (rightNeighborRank != -1) { // Not a cell at the right boundary
                MPI_Isend(&E_prev[innerBlockRowStartRightIndex], 1, sendBuf, rightNeighborRank, RIGHT, MPI_COMM_WORLD, &requests[6]);
                MPI_Irecv(&E_prev[innerBlockRowStartRightIndex+1], 1, sendBuf, rightNeighborRank, LEFT, MPI_COMM_WORLD, &requests[7]);
            } else { // Fills in the RIGHT Ghost Cells
                for (int i = innerBlockRowStartRightIndex+1; i < outerBlockRowEndRightIndex; i += rowLen) {
                    E_prev[i] = E_prev[i-2];
                    // ct4 += 1;
                }
            }
        }
#define FUSED 1
#define SEP 1
#define SSE2_VEC 1

#ifdef FUSED
#ifdef SEP
        for(int j = 2*rowLen+2; j <= (m-1)*rowLen+2; j+=rowLen) {
            E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            R_tmp = R + j;
#ifndef SSE2_VEC
            for(int i = 0; i < n-2; i++) {
                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+rowLen]+E_prev_tmp[i-rowLen]);
                E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
                R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
            }
#else
            for (int i=0; i < n-2; i+=2) {
                // load 2 cells each time into a 128 bit SSE2 register
                __m128d E_prev_center = _mm_loadu_pd(&E_prev_tmp[i]); // use _mm_loadu_pd() instead of _mm_load_pd() in case memory alignment is not guaranteed
                __m128d E_prev_right = _mm_loadu_pd(&E_prev_tmp[i+1]);
                __m128d E_prev_left = _mm_loadu_pd(&E_prev_tmp[i-1]);
                __m128d E_prev_bottom = _mm_loadu_pd(&E_prev_tmp[i+rowLen]);
                __m128d E_prev_top = _mm_loadu_pd(&E_prev_tmp[i-rowLen]);
                __m128d R_new = _mm_loadu_pd(&R_tmp[i]);

                // compute E_tmp[i]
                __m128d E_sum1 = _mm_add_pd(E_prev_right, E_prev_left);
                __m128d E_sum2 = _mm_add_pd(E_prev_bottom, E_prev_top);
                __m128d E_sum3 = _mm_add_pd(E_sum1, E_sum2);
                __m128d E_mid = _mm_mul_pd(_mm_set1_pd(4.0), E_prev_center);
                __m128d E_dif = _mm_sub_pd(E_sum3, E_mid);
                __m128d E_off = _mm_mul_pd(_mm_set1_pd(alpha), E_dif);
                __m128d E_new = _mm_add_pd(E_prev_center, E_off);

                // E_tmp[i], second line
                __m128d E_pro1 = _mm_mul_pd(_mm_set1_pd(kk), E_prev_center);
                __m128d E_sub1 = _mm_sub_pd(E_prev_center, _mm_set1_pd(a));
                __m128d E_sub2 = _mm_sub_pd(E_prev_center, _mm_set1_pd(1.0));
                __m128d E_pro2 = _mm_mul_pd(E_sub1, E_sub2);
                __m128d E_pro3 = _mm_mul_pd(E_pro1, E_pro2);
                __m128d E_pro4 = _mm_mul_pd(E_prev_center, R_new);
                __m128d E_sum4 = _mm_add_pd(E_pro3, E_pro4);
                __m128d E_off2 = _mm_mul_pd(_mm_set1_pd(dt), E_sum4);
                E_new = _mm_sub_pd(E_new, E_off2);

                // compute R_tmp[i]
                __m128d E_pro5 = _mm_mul_pd(_mm_set1_pd(M1), R_new);
                __m128d E_sum5 = _mm_add_pd(E_prev_center, _mm_set1_pd(M2));
                __m128d E_div1 = _mm_div_pd(E_pro5, E_sum5);
                __m128d E_sum6 = _mm_add_pd(_mm_set1_pd(epsilon), E_div1);
                __m128d E_pro6 = _mm_mul_pd(_mm_set1_pd(dt), E_sum6);
                __m128d E_sub3 = _mm_sub_pd(E_prev_center, _mm_set1_pd(b));
                __m128d E_sub4 = _mm_sub_pd(E_sub3, _mm_set1_pd(1.0));
                __m128d E_pro7 = _mm_mul_pd(E_pro1, E_sub4);
                __m128d E_sum7 = _mm_add_pd(R_new, E_pro7);
                __m128d E_pro8 = _mm_mul_pd(E_pro6, E_sum7);
                R_new = _mm_sub_pd(R_new, E_pro8);

                // store E_tmp[i] and R_tmp[i]
                _mm_storeu_pd(&E_tmp[i], E_new);
                _mm_storeu_pd(&R_tmp[i], R_new);
            }
#endif
        }      
        if (!cb.noComm){
            MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);
            // MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes
        }

        int js = innerBlockRowStartLeftIndex;
        E_tmp = E + js;
        E_prev_tmp = E_prev + js;
        R_tmp = R + js;
#ifndef SSE2_VEC
        for(int i = 0; i < n; i++) {
            E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+rowLen]+E_prev_tmp[i-rowLen]);
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
#else 
        for (int i=0; i<n; i+=2) {
            // load 2 cells each time into a 128 bit SSE2 register
            __m128d E_prev_center = _mm_loadu_pd(&E_prev_tmp[i]); // use _mm_loadu_pd() instead of _mm_load_pd() in case memory alignment is not guaranteed
            __m128d E_prev_right = _mm_loadu_pd(&E_prev_tmp[i+1]);
            __m128d E_prev_left = _mm_loadu_pd(&E_prev_tmp[i-1]);
            __m128d E_prev_bottom = _mm_loadu_pd(&E_prev_tmp[i+rowLen]);
            __m128d E_prev_top = _mm_loadu_pd(&E_prev_tmp[i-rowLen]);
            __m128d R_new = _mm_loadu_pd(&R_tmp[i]);

            // compute E_tmp[i]
            __m128d E_sum1 = _mm_add_pd(E_prev_right, E_prev_left);
            __m128d E_sum2 = _mm_add_pd(E_prev_bottom, E_prev_top);
            __m128d E_sum3 = _mm_add_pd(E_sum1, E_sum2);
            __m128d E_mid = _mm_mul_pd(_mm_set1_pd(4.0), E_prev_center);
            __m128d E_dif = _mm_sub_pd(E_sum3, E_mid);
            __m128d E_off = _mm_mul_pd(_mm_set1_pd(alpha), E_dif);
            __m128d E_new = _mm_add_pd(E_prev_center, E_off);

            // E_tmp[i], second line
            __m128d E_pro1 = _mm_mul_pd(_mm_set1_pd(kk), E_prev_center);
            __m128d E_sub1 = _mm_sub_pd(E_prev_center, _mm_set1_pd(a));
            __m128d E_sub2 = _mm_sub_pd(E_prev_center, _mm_set1_pd(1.0));
            __m128d E_pro2 = _mm_mul_pd(E_sub1, E_sub2);
            __m128d E_pro3 = _mm_mul_pd(E_pro1, E_pro2);
            __m128d E_pro4 = _mm_mul_pd(E_prev_center, R_new);
            __m128d E_sum4 = _mm_add_pd(E_pro3, E_pro4);
            __m128d E_off2 = _mm_mul_pd(_mm_set1_pd(dt), E_sum4);
            E_new = _mm_sub_pd(E_new, E_off2);

            // compute R_tmp[i]
            __m128d E_pro5 = _mm_mul_pd(_mm_set1_pd(M1), R_new);
            __m128d E_sum5 = _mm_add_pd(E_prev_center, _mm_set1_pd(M2));
            __m128d E_div1 = _mm_div_pd(E_pro5, E_sum5);
            __m128d E_sum6 = _mm_add_pd(_mm_set1_pd(epsilon), E_div1);
            __m128d E_pro6 = _mm_mul_pd(_mm_set1_pd(dt), E_sum6);
            __m128d E_sub3 = _mm_sub_pd(E_prev_center, _mm_set1_pd(b));
            __m128d E_sub4 = _mm_sub_pd(E_sub3, _mm_set1_pd(1.0));
            __m128d E_pro7 = _mm_mul_pd(E_pro1, E_sub4);
            __m128d E_sum7 = _mm_add_pd(R_new, E_pro7);
            __m128d E_pro8 = _mm_mul_pd(E_pro6, E_sum7);
            R_new = _mm_sub_pd(R_new, E_pro8);

            // store E_tmp[i] and R_tmp[i]
            _mm_storeu_pd(&E_tmp[i], E_new);
            _mm_storeu_pd(&R_tmp[i], R_new);
        }
#endif

        int je = innerBlockRowEndLeftIndex;
        E_tmp = E + je;
        E_prev_tmp = E_prev + je;
        R_tmp = R + je;
#ifndef SSE2_VEC
        for(int i = 0; i < n; i++) {
            E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+rowLen]+E_prev_tmp[i-rowLen]);
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
#else
        for (int i=0; i<n; i+=2) {
            // load 2 cells each time into a 128 bit SSE2 register
            __m128d E_prev_center = _mm_loadu_pd(&E_prev_tmp[i]); // use _mm_loadu_pd() instead of _mm_load_pd() in case memory alignment is not guaranteed
            __m128d E_prev_right = _mm_loadu_pd(&E_prev_tmp[i+1]);
            __m128d E_prev_left = _mm_loadu_pd(&E_prev_tmp[i-1]);
            __m128d E_prev_bottom = _mm_loadu_pd(&E_prev_tmp[i+rowLen]);
            __m128d E_prev_top = _mm_loadu_pd(&E_prev_tmp[i-rowLen]);
            __m128d R_new = _mm_loadu_pd(&R_tmp[i]);

            // compute E_tmp[i]
            __m128d E_sum1 = _mm_add_pd(E_prev_right, E_prev_left);
            __m128d E_sum2 = _mm_add_pd(E_prev_bottom, E_prev_top);
            __m128d E_sum3 = _mm_add_pd(E_sum1, E_sum2);
            __m128d E_mid = _mm_mul_pd(_mm_set1_pd(4.0), E_prev_center);
            __m128d E_dif = _mm_sub_pd(E_sum3, E_mid);
            __m128d E_off = _mm_mul_pd(_mm_set1_pd(alpha), E_dif);
            __m128d E_new = _mm_add_pd(E_prev_center, E_off);

            // E_tmp[i], second line
            __m128d E_pro1 = _mm_mul_pd(_mm_set1_pd(kk), E_prev_center);
            __m128d E_sub1 = _mm_sub_pd(E_prev_center, _mm_set1_pd(a));
            __m128d E_sub2 = _mm_sub_pd(E_prev_center, _mm_set1_pd(1.0));
            __m128d E_pro2 = _mm_mul_pd(E_sub1, E_sub2);
            __m128d E_pro3 = _mm_mul_pd(E_pro1, E_pro2);
            __m128d E_pro4 = _mm_mul_pd(E_prev_center, R_new);
            __m128d E_sum4 = _mm_add_pd(E_pro3, E_pro4);
            __m128d E_off2 = _mm_mul_pd(_mm_set1_pd(dt), E_sum4);
            E_new = _mm_sub_pd(E_new, E_off2);

            // compute R_tmp[i]
            __m128d E_pro5 = _mm_mul_pd(_mm_set1_pd(M1), R_new);
            __m128d E_sum5 = _mm_add_pd(E_prev_center, _mm_set1_pd(M2));
            __m128d E_div1 = _mm_div_pd(E_pro5, E_sum5);
            __m128d E_sum6 = _mm_add_pd(_mm_set1_pd(epsilon), E_div1);
            __m128d E_pro6 = _mm_mul_pd(_mm_set1_pd(dt), E_sum6);
            __m128d E_sub3 = _mm_sub_pd(E_prev_center, _mm_set1_pd(b));
            __m128d E_sub4 = _mm_sub_pd(E_sub3, _mm_set1_pd(1.0));
            __m128d E_pro7 = _mm_mul_pd(E_pro1, E_sub4);
            __m128d E_sum7 = _mm_add_pd(R_new, E_pro7);
            __m128d E_pro8 = _mm_mul_pd(E_pro6, E_sum7);
            R_new = _mm_sub_pd(R_new, E_pro8);

            // store E_tmp[i] and R_tmp[i]
            _mm_storeu_pd(&E_tmp[i], E_new);
            _mm_storeu_pd(&R_tmp[i], R_new);
        }
#endif

#ifndef SSE2_VEC   
        for(int j = innerBlockRowStartLeftIndex+rowLen; j <= innerBlockRowEndLeftIndex-rowLen; j+=rowLen) {
            E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            R_tmp = R + j;
        
            int i1 = 0;
            int i2 = n-1;
            E_tmp[i1] = E_prev_tmp[i1]+alpha*(E_prev_tmp[i1+1]+E_prev_tmp[i1-1]-4*E_prev_tmp[i1]+E_prev_tmp[i1+rowLen]+E_prev_tmp[i1-rowLen]);
            E_tmp[i1] += -dt*(kk*E_prev_tmp[i1]*(E_prev_tmp[i1]-a)*(E_prev_tmp[i1]-1)+E_prev_tmp[i1]*R_tmp[i1]);
            R_tmp[i1] += dt*(epsilon+M1* R_tmp[i1]/( E_prev_tmp[i1]+M2))*(-R_tmp[i1]-kk*E_prev_tmp[i1]*(E_prev_tmp[i1]-b-1));
            
            E_tmp[i2] = E_prev_tmp[i2]+alpha*(E_prev_tmp[i2+1]+E_prev_tmp[i2-1]-4*E_prev_tmp[i2]+E_prev_tmp[i2+rowLen]+E_prev_tmp[i2-rowLen]);
            E_tmp[i2] += -dt*(kk*E_prev_tmp[i2]*(E_prev_tmp[i2]-a)*(E_prev_tmp[i2]-1)+E_prev_tmp[i2]*R_tmp[i2]);
            R_tmp[i2] += dt*(epsilon+M1* R_tmp[i2]/( E_prev_tmp[i2]+M2))*(-R_tmp[i2]-kk*E_prev_tmp[i2]*(E_prev_tmp[i2]-b-1));
        }
#else
        for (int j = 0; j < m-2; j+=2) {
            E_tmp = E + innerBlockRowStartLeftIndex+rowLen + j*rowLen;
            E_prev_tmp = E_prev + innerBlockRowStartLeftIndex+rowLen + j*rowLen;
            R_tmp = R + innerBlockRowStartLeftIndex+rowLen + j*rowLen;

            int i = 0;
            // load 2 cells each time into a 128 bit SSE2 register
            __m128d E_prev_center = _mm_loadl_pd(_mm_setzero_pd(), &E_prev_tmp[i]); // load first element
            E_prev_center = _mm_loadh_pd(E_prev_center, &E_prev_tmp[i+rowLen]); // load second element
            __m128d E_prev_right = _mm_loadl_pd(_mm_setzero_pd(), &E_prev_tmp[i+1]);
            E_prev_right = _mm_loadh_pd(E_prev_right, &E_prev_tmp[i+1+rowLen]);
            __m128d E_prev_left = _mm_loadl_pd(_mm_setzero_pd(), &E_prev_tmp[i-1]);
            E_prev_left = _mm_loadh_pd(E_prev_left, &E_prev_tmp[i-1+rowLen]);
            __m128d E_prev_bottom = _mm_loadl_pd(_mm_setzero_pd(), &E_prev_tmp[i+rowLen]);
            E_prev_bottom = _mm_loadh_pd(E_prev_bottom, &E_prev_tmp[i+rowLen+rowLen]);
            __m128d E_prev_top = _mm_loadl_pd(_mm_setzero_pd(), &E_prev_tmp[i-rowLen]);
            E_prev_top = _mm_loadh_pd(E_prev_top, &E_prev_tmp[i]);
            __m128d R_new = _mm_loadl_pd(_mm_setzero_pd(), &R_tmp[i]);
            R_new = _mm_loadh_pd(R_new, &R_tmp[i+rowLen]);

            // __m128d E_prev_right = _mm_loadu_pd(&E_prev_tmp[i+1]);
            // __m128d E_prev_left = _mm_loadu_pd(&E_prev_tmp[i-1]);
            // __m128d E_prev_bottom = _mm_loadu_pd(&E_prev_tmp[i+rowLen]);
            // __m128d E_prev_top = _mm_loadu_pd(&E_prev_tmp[i-rowLen]);
            // __m128d R_new = _mm_loadu_pd(&R_tmp[i]);

            // compute E_tmp[i]
            __m128d E_sum1 = _mm_add_pd(E_prev_right, E_prev_left);
            __m128d E_sum2 = _mm_add_pd(E_prev_bottom, E_prev_top);
            __m128d E_sum3 = _mm_add_pd(E_sum1, E_sum2);
            __m128d E_mid = _mm_mul_pd(_mm_set1_pd(4.0), E_prev_center);
            __m128d E_dif = _mm_sub_pd(E_sum3, E_mid);
            __m128d E_off = _mm_mul_pd(_mm_set1_pd(alpha), E_dif);
            __m128d E_new = _mm_add_pd(E_prev_center, E_off);

            // E_tmp[i], second line
            __m128d E_pro1 = _mm_mul_pd(_mm_set1_pd(kk), E_prev_center);
            __m128d E_sub1 = _mm_sub_pd(E_prev_center, _mm_set1_pd(a));
            __m128d E_sub2 = _mm_sub_pd(E_prev_center, _mm_set1_pd(1.0));
            __m128d E_pro2 = _mm_mul_pd(E_sub1, E_sub2);
            __m128d E_pro3 = _mm_mul_pd(E_pro1, E_pro2);
            __m128d E_pro4 = _mm_mul_pd(E_prev_center, R_new);
            __m128d E_sum4 = _mm_add_pd(E_pro3, E_pro4);
            __m128d E_off2 = _mm_mul_pd(_mm_set1_pd(dt), E_sum4);
            E_new = _mm_sub_pd(E_new, E_off2);

            // compute R_tmp[i]
            __m128d E_pro5 = _mm_mul_pd(_mm_set1_pd(M1), R_new);
            __m128d E_sum5 = _mm_add_pd(E_prev_center, _mm_set1_pd(M2));
            __m128d E_div1 = _mm_div_pd(E_pro5, E_sum5);
            __m128d E_sum6 = _mm_add_pd(_mm_set1_pd(epsilon), E_div1);
            __m128d E_pro6 = _mm_mul_pd(_mm_set1_pd(dt), E_sum6);
            __m128d E_sub3 = _mm_sub_pd(E_prev_center, _mm_set1_pd(b));
            __m128d E_sub4 = _mm_sub_pd(E_sub3, _mm_set1_pd(1.0));
            __m128d E_pro7 = _mm_mul_pd(E_pro1, E_sub4);
            __m128d E_sum7 = _mm_add_pd(R_new, E_pro7);
            __m128d E_pro8 = _mm_mul_pd(E_pro6, E_sum7);
            R_new = _mm_sub_pd(R_new, E_pro8);

            // store E_tmp[i] and R_tmp[i]
            // _mm_storeu_pd(&E_tmp[i], E_new);
            // _mm_storeu_pd(&R_tmp[i], R_new);
            // _mm_store_sd(&E_tmp[i], E_new); // or _mm_storeu_pd()
            // _mm_store_sd(&R_tmp[i], R_new);
            _mm_storel_pd(&E_tmp[i], E_new);
            _mm_storel_pd(&R_tmp[i], R_new);
            _mm_storeh_pd(&E_tmp[i+rowLen], E_new);
            _mm_storeh_pd(&R_tmp[i+rowLen], R_new);

            i = n - 1;
            E_prev_center = _mm_loadl_pd(_mm_setzero_pd(), &E_prev_tmp[i]); // load first element
            E_prev_center = _mm_loadh_pd(E_prev_center, &E_prev_tmp[i+rowLen]); // load second element
            E_prev_right = _mm_loadl_pd(_mm_setzero_pd(), &E_prev_tmp[i+1]);
            E_prev_right = _mm_loadh_pd(E_prev_right, &E_prev_tmp[i+1+rowLen]);
            E_prev_left = _mm_loadl_pd(_mm_setzero_pd(), &E_prev_tmp[i-1]);
            E_prev_left = _mm_loadh_pd(E_prev_left, &E_prev_tmp[i-1+rowLen]);
            E_prev_bottom = _mm_loadl_pd(_mm_setzero_pd(), &E_prev_tmp[i+rowLen]);
            E_prev_bottom = _mm_loadh_pd(E_prev_bottom, &E_prev_tmp[i+rowLen+rowLen]);
            E_prev_top = _mm_loadl_pd(_mm_setzero_pd(), &E_prev_tmp[i-rowLen]);
            E_prev_top = _mm_loadh_pd(E_prev_top, &E_prev_tmp[i]);
            R_new = _mm_loadl_pd(_mm_setzero_pd(), &R_tmp[i]);
            R_new = _mm_loadh_pd(R_new, &R_tmp[i+rowLen]);

            // compute E_tmp[i]
            E_sum1 = _mm_add_pd(E_prev_right, E_prev_left);
            E_sum2 = _mm_add_pd(E_prev_bottom, E_prev_top);
            E_sum3 = _mm_add_pd(E_sum1, E_sum2);
            E_mid = _mm_mul_pd(_mm_set1_pd(4.0), E_prev_center);
            E_dif = _mm_sub_pd(E_sum3, E_mid);
            E_off = _mm_mul_pd(_mm_set1_pd(alpha), E_dif);
            E_new = _mm_add_pd(E_prev_center, E_off);

            // E_tmp[i], second line
            E_pro1 = _mm_mul_pd(_mm_set1_pd(kk), E_prev_center);
            E_sub1 = _mm_sub_pd(E_prev_center, _mm_set1_pd(a));
            E_sub2 = _mm_sub_pd(E_prev_center, _mm_set1_pd(1.0));
            E_pro2 = _mm_mul_pd(E_sub1, E_sub2);
            E_pro3 = _mm_mul_pd(E_pro1, E_pro2);
            E_pro4 = _mm_mul_pd(E_prev_center, R_new);
            E_sum4 = _mm_add_pd(E_pro3, E_pro4);
            E_off2 = _mm_mul_pd(_mm_set1_pd(dt), E_sum4);
            E_new = _mm_sub_pd(E_new, E_off2);

            // compute R_tmp[i]
            E_pro5 = _mm_mul_pd(_mm_set1_pd(M1), R_new);
            E_sum5 = _mm_add_pd(E_prev_center, _mm_set1_pd(M2));
            E_div1 = _mm_div_pd(E_pro5, E_sum5);
            E_sum6 = _mm_add_pd(_mm_set1_pd(epsilon), E_div1);
            E_pro6 = _mm_mul_pd(_mm_set1_pd(dt), E_sum6);
            E_sub3 = _mm_sub_pd(E_prev_center, _mm_set1_pd(b));
            E_sub4 = _mm_sub_pd(E_sub3, _mm_set1_pd(1.0));
            E_pro7 = _mm_mul_pd(E_pro1, E_sub4);
            E_sum7 = _mm_add_pd(R_new, E_pro7);
            E_pro8 = _mm_mul_pd(E_pro6, E_sum7);
            R_new = _mm_sub_pd(R_new, E_pro8);

            // store E_tmp[i] and R_tmp[i]
            // _mm_storeu_pd(&E_tmp[i], E_new);
            // _mm_storeu_pd(&R_tmp[i], R_new);
            _mm_storel_pd(&E_tmp[i], E_new);
            _mm_storel_pd(&R_tmp[i], R_new);
            _mm_storeh_pd(&E_tmp[i+rowLen], E_new);
            _mm_storeh_pd(&R_tmp[i+rowLen], R_new);
        }
#endif
#else
        if (!cb.noComm) {
            MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);
            MPI_Barrier(MPI_COMM_WORLD); 
        }
        for(int j = innerBlockRowStartLeftIndex; j <= innerBlockRowEndLeftIndex; j+=rowLen) {
            E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            R_tmp = R + j;

#ifndef SSE2_VEC
            for(int i = 0; i < n; i++) {
                // // declare these as registers since we are not changing their value
                // register double E_prev_center = E_prev_tmp[i];
                // register double E_prev_right = E_prev_tmp[i+1];
                // register double E_prev_left = E_prev_tmp[i-1];
                // register double E_prev_bottom = E_prev_tmp[i+rowLen];
                // register double E_prev_top = E_prev_tmp[i-rowLen];
                // E_tmp[i] = E_prev_center+alpha*(E_prev_right+E_prev_left-4*E_prev_tmp[i]+E_prev_bottom+E_prev_top);
                // E_tmp[i] += -dt*(kk*E_prev_center*(E_prev_center-a)*(E_prev_center-1)+E_prev_center*R_tmp[i]);
                // R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));

                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+rowLen]+E_prev_tmp[i-rowLen]);
                E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
                R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
            }
#else
            for (int i = 0; i < n; i += 2) {
                // load 2 cells each time into a 128 bit SSE2 register
                __m128d E_prev_center = _mm_loadu_pd(&E_prev_tmp[i]); // use _mm_loadu_pd() instead of _mm_load_pd() in case memory alignment is not guaranteed
                __m128d E_prev_right = _mm_loadu_pd(&E_prev_tmp[i+1]);
                __m128d E_prev_left = _mm_loadu_pd(&E_prev_tmp[i-1]);
                __m128d E_prev_bottom = _mm_loadu_pd(&E_prev_tmp[i+rowLen]);
                __m128d E_prev_top = _mm_loadu_pd(&E_prev_tmp[i-rowLen]);
                __m128d R_new = _mm_loadu_pd(&R_tmp[i]);

                // compute E_tmp[i]
                __m128d E_sum1 = _mm_add_pd(E_prev_right, E_prev_left);
                __m128d E_sum2 = _mm_add_pd(E_prev_bottom, E_prev_top);
                __m128d E_sum3 = _mm_add_pd(E_sum1, E_sum2);
                __m128d E_mid = _mm_mul_pd(_mm_set1_pd(4.0), E_prev_center);
                __m128d E_dif = _mm_sub_pd(E_sum3, E_mid);
                __m128d E_off = _mm_mul_pd(_mm_set1_pd(alpha), E_dif);
                __m128d E_new = _mm_add_pd(E_prev_center, E_off);

                // E_tmp[i], second line
                __m128d E_pro1 = _mm_mul_pd(_mm_set1_pd(kk), E_prev_center);
                __m128d E_sub1 = _mm_sub_pd(E_prev_center, _mm_set1_pd(a));
                __m128d E_sub2 = _mm_sub_pd(E_prev_center, _mm_set1_pd(1.0));
                __m128d E_pro2 = _mm_mul_pd(E_sub1, E_sub2);
                __m128d E_pro3 = _mm_mul_pd(E_pro1, E_pro2);
                __m128d E_pro4 = _mm_mul_pd(E_prev_center, R_new);
                __m128d E_sum4 = _mm_add_pd(E_pro3, E_pro4);
                __m128d E_off2 = _mm_mul_pd(_mm_set1_pd(dt), E_sum4);
                E_new = _mm_sub_pd(E_new, E_off2);

                // compute R_tmp[i]
                __m128d E_pro5 = _mm_mul_pd(_mm_set1_pd(M1), R_new);
                __m128d E_sum5 = _mm_add_pd(E_prev_center, _mm_set1_pd(M2));
                __m128d E_div1 = _mm_div_pd(E_pro5, E_sum5);
                __m128d E_sum6 = _mm_add_pd(_mm_set1_pd(epsilon), E_div1);
                __m128d E_pro6 = _mm_mul_pd(_mm_set1_pd(dt), E_sum6);
                __m128d E_sub3 = _mm_sub_pd(E_prev_center, _mm_set1_pd(b));
                __m128d E_sub4 = _mm_sub_pd(E_sub3, _mm_set1_pd(1.0));
                __m128d E_pro7 = _mm_mul_pd(E_pro1, E_sub4);
                __m128d E_sum7 = _mm_add_pd(R_new, E_pro7);
                __m128d E_pro8 = _mm_mul_pd(E_pro6, E_sum7);
                R_new = _mm_sub_pd(R_new, E_pro8);

                // store E_tmp[i] and R_tmp[i]
                _mm_storeu_pd(&E_tmp[i], E_new);
                _mm_storeu_pd(&R_tmp[i], R_new);
            }
#endif
        }
#endif

#else
        // need to change later
        // Solve for the excitation, a PDE
        for(int j = innerBlockRowStartLeftIndex; j <= innerBlockRowEndLeftIndex; j+=rowLen) {
            E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            for(int i = 0; i < n; i++) {
                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+rowLen]+E_prev_tmp[i-rowLen]);
            }
        }

        /* 
        * Solve the ODE, advancing excitation and recovery variables
        *     to the next timtestep
        */
        for(int j = innerBlockRowStartLeftIndex; j <= innerBlockRowEndLeftIndex; j+=rowLen) {
            E_tmp = E + j;
            R_tmp = R + j;
            E_prev_tmp = E_prev + j;
            for(int i = 0; i < n; i++) {
                E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
                R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
            }
        }
        
#endif
        if (!cb.noComm){
            // MPI_Barrier(MPI_COMM_WORLD);
        }
        /////////////////////////////////////////////////////////////////////////////////
        if (cb.stats_freq){
            if ( !(niter % cb.stats_freq)){
                stats(E,m,n,&mx,&sumSq);
                double l2norm = L2Norm(sumSq);
                repNorms(l2norm,mx,dt,m,n,niter, cb.stats_freq);
            }
        }

        if (cb.plot_freq){
            if (!(niter % cb.plot_freq)){
                plotter->updatePlot(E,  niter, m, n);
            }
        }
        // Swap current and previous meshes
        double *tmp = E; E = E_prev; E_prev = tmp;

    } //end of 'niter' loop at the beginning

    //  printMat2("Rank 0 Matrix E_prev", E_prev, m,n);  // return the L2 and infinity norms via in-out parameters

    // printf("rank: %d, cb.m: %d, cb.n: %d, m %d, n: %d, Linf: %.5f, sumSq: %.5f\n", rank, cb.m, cb.n, m, n, Linf, sumSq);
    // if (rank == 0) {
    //     // clear out cells that are not under the computation of rank 0
    //     for (int j = 0; j < cb.m+2; j++) {
    //         for (int i = 0; i < cb.n+2; i++) {
    //             if (i != 0 && j != 0 && i != n+1 && j != m+1 && j < m+2 && i < n+2) continue; // exclude boundary and extra cells
    //             E_prev[j*(cb.n+2)+i] = 0;
    //         }
    //     }
    // }
    stats(E_prev,m,n,&Linf,&sumSq);
    // MPI_Barrier(MPI_COMM_WORLD);
    // if (rank == 0) printMat2("Printing E_prev for rank 0 after all iterations", E_prev, cb.m, cb.n);
    // if (rank == 0) printSubMat2("Printing E_prev for rank 0 after all iterations", E_prev, cb.m, cb.n, m, n, innerBlockRowStartLeftIndex, rank);
    
    // printf("rank: %d, Linf: %.5f, sumSq: %.5f\n", rank, Linf, sumSq);
    // printf("rank: %d, cb.m: %d, cb.n: %d, m %d, n: %d, Linf: %.5f, sumSq: %.5f\n", rank, cb.m, cb.n, m, n, Linf, sumSq);
    

    if (!cb.noComm) {
        MPI_Barrier(MPI_COMM_WORLD);
        double Linf_global = 0, sumSq_global = 0;
        // all processors need to call MPI_reduce()
        // MPI_Reduce(&Linf, &Linf_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        // MPI_Reduce(&sumSq, &sumSq_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        // since MPI_Reduce() is blocking, we don't need this call MPI_Barrier(MPI_COMM_WORLD)
        // using MPI_Allreduce() may lead to more communication overhead
        MPI_Allreduce(&Linf, &Linf_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&sumSq, &sumSq_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        L2 = L2Norm(sumSq_global);
        Linf = Linf_global;
    } else {
        L2 = L2Norm(sumSq);
    }

    // Swap pointers so we can re-use the arrays
    *_E = E;
    *_E_prev = E_prev;
    MPI_Type_free(&sendBuf);
}
#else
// original solve() method
void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf){
    // Simulated time is different from the integer timestep number
    double t = 0.0;

    double *E = *_E, *E_prev = *_E_prev;
    double *R_tmp = R;
    double *E_tmp = *_E;
    double *E_prev_tmp = *_E_prev;
    double mx, sumSq;
    int niter;
    int m = cb.m, n=cb.n;
    int innerBlockRowStartIndex = (n+2)+1;
    int innerBlockRowEndIndex = (((m+2)*(n+2) - 1) - (n)) - (n+2);

    // We continue to sweep over the mesh until the simulation has reached
    // the desired number of iterations
    for (niter = 0; niter < cb.niters; niter++){
        if (cb.debug && (niter==0)){
            stats(E_prev,m,n,&mx,&sumSq);
            double l2norm = L2Norm(sumSq);
            repNorms(l2norm,mx,dt,m,n,-1, cb.stats_freq);

            if (cb.plot_freq)
                plotter->updatePlot(E,  -1, m+1, n+1);
        }

        /* 
        * Copy data from boundary of the computational box to the
        * padding region, set up for differencing computational box's boundary
        *
        * These are physical boundary conditions, and are not to be confused
        * with ghost cells that we would use in an MPI implementation
        *
        * The reason why we copy boundary conditions is to avoid
        * computing single sided differences at the boundaries
        * which increase the running time of solve()
        *
        */
    
        // 4 FOR LOOPS set up the padding needed for the boundary conditions
        int i,j;

        // Fills in the TOP Ghost Cells
        for (i = 0; i < (n+2); i++) {
            E_prev[i] = E_prev[i + (n+2)*2];
        }

        // Fills in the RIGHT Ghost Cells
        for (i = (n+1); i < (m+2)*(n+2); i+=(n+2)) {
            E_prev[i] = E_prev[i-2];
        }

        // Fills in the LEFT Ghost Cells
        for (i = 0; i < (m+2)*(n+2); i+=(n+2)) {
            E_prev[i] = E_prev[i+2];
        }	

        // Fills in the BOTTOM Ghost Cells
        for (i = ((m+2)*(n+2)-(n+2)); i < (m+2)*(n+2); i++) {
            E_prev[i] = E_prev[i - (n+2)*2];
        }
        //////////////////////////////////////////////////////////////////////////////
#define FUSED 1
#ifdef FUSED
        // Solve for the excitation, a PDE
        for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
            E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            R_tmp = R + j;
            for(i = 0; i < n; i++) {
                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
                E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
                R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
            }
        }
#else
        // Solve for the excitation, a PDE
        for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
            E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            for(i = 0; i < n; i++) {
                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            }
        }

        /* 
        * Solve the ODE, advancing excitation and recovery variables
        *     to the next timtestep
        */

        for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
            E_tmp = E + j;
            R_tmp = R + j;
            E_prev_tmp = E_prev + j;
            for(i = 0; i < n; i++) {
                E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
                R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
            }
        }
#endif
        /////////////////////////////////////////////////////////////////////////////////
        if (cb.stats_freq){
            if ( !(niter % cb.stats_freq)){
                stats(E,m,n,&mx,&sumSq);
                double l2norm = L2Norm(sumSq);
                repNorms(l2norm,mx,dt,m,n,niter, cb.stats_freq);
            }
        }

        if (cb.plot_freq){
            if (!(niter % cb.plot_freq)){
                plotter->updatePlot(E,  niter, m, n);
            }
        }
        // Swap current and previous meshes
        double *tmp = E; E = E_prev; E_prev = tmp;

    } //end of 'niter' loop at the beginning

    //  printMat2("Rank 0 Matrix E_prev", E_prev, m,n);  // return the L2 and infinity norms via in-out parameters
    stats(E_prev,m,n,&Linf,&sumSq);
    L2 = L2Norm(sumSq);

    // Swap pointers so we can re-use the arrays
    *_E = E;
    *_E_prev = E_prev;
}
#endif

void printMat2(const char mesg[], double *E, int m, int n){
    int i;
#if 0
    if (m>8)
        return;
#else
    if (m>34)
        return;
#endif
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

void printSubMat2(const char mesg[], double *E, int m, int n, int ny, int nx, int startIdx, int rank) {
    if (m > 34 || n > 34) return;
    int i;
    printf("%s\n",mesg);
    if (rank == 0) {
        for (i=0; i < (ny+2)*(nx+2); i++){
            int rowIndex = i / (nx+2);
            int colIndex = i % (nx+2);
                printf("%6.3f ", E[startIdx+colIndex+rowIndex*(n+2)]);
            if (colIndex == nx+1)
                printf("\n");
        }
    } else {
        for (i=0; i < (ny+2)*(nx+2); i++){
        int rowIndex = i / (nx+2);
        int colIndex = i % (nx+2);
                printf("%6.3f ", E[i]);
        if (colIndex == nx+1)
	        printf("\n");
        }
    }
}