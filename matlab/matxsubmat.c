/*=========================================================
 * matxsubmat.c - Matrix product of a matrix A and  
 * a submatrix of B without creating a memory copy.  
 * Calls the BLAS function dgemm.
 *
 * C = matrixMultiply(A,B,ind1,ind2) computes the product 
 *                    C = A*B(:,ind1:ind2)
 * where A, B, and C are matrices containing real values.
 *
 * This is a MEX-file for MATLAB.
 * Adopted from MATLAB's matrixMultiply.c demo MEX file.
 * Stefan Guettel, 10/2023.
 *=======================================================*/

#include "mex.h"
#include "blas.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
#if MX_HAS_INTERLEAVED_COMPLEX
    mxDouble *A, *B, *C; /* pointers to input & output matrices*/
#else
    double *A, *B, *C;   /* pointers to input & output matrices*/
#endif
    ptrdiff_t m,n,p;     /* matrix dimensions */
    /* form of op(A) & op(B) to use in matrix multiplication */
    char *chn = "N";
    /* scalar values to use in dgemm */
    double one = 1.0, zero = 0.0; 
    int col_s, col_e;    /* start / end indices of B's columns to use */

#if MX_HAS_INTERLEAVED_COMPLEX
    A = mxGetDoubles(prhs[0]); /* first input matrix */
    B = mxGetDoubles(prhs[1]); /* second input matrix */
#else
    A = mxGetPr(prhs[0]);      /* first input matrix */
    B = mxGetPr(prhs[1]);      /* second input matrix */
#endif

    /* dimensions of input matrices */
    m = mxGetM(prhs[0]);  
    p = mxGetN(prhs[0]);
    //n = mxGetN(prhs[1]);
    
    // TURNING OFF ALL INPUT CHECKS FOR PERFORMANCE.
    /*
    // check to make sure the first input argument is a real matrix 
    if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
      mexErrMsgIdAndTxt( "MATLAB:matrixMultiply:fieldNotRealMatrix",
              "First input argument must be a real matrix.");
    }

    // check to make sure the second input argument is a real matrix 
    if( !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1])) {
      mexErrMsgIdAndTxt( "MATLAB:matrixMultiply:fieldNotRealMatrix",
              "Second input argument must be a real matrix.");
    }

    if (p != mxGetM(prhs[1])) {
        mexErrMsgIdAndTxt("MATLAB:matrixMultiply:matchdims",
                "Inner dimensions of matrix multiply do not match.");
    }
    */

    /* modify column pointer and output dimension */
    col_s = (int) mxGetScalar(prhs[2]);
    col_e = (int) mxGetScalar(prhs[3]);
    /* B = B + p; this would be pointer to second column of B */
    B = B + p*(col_s-1); /* pointer to starting column */
    n = (ptrdiff_t) (col_e - col_s + 1); /* number of columns to compute */

    /* create output matrix C */
    plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
#if MX_HAS_INTERLEAVED_COMPLEX
    C = mxGetDoubles(plhs[0]);
#else
    C = mxGetPr(plhs[0]);
#endif

    /* Pass arguments to Fortran by reference */
    dgemm(chn, chn, &m, &n, &p, &one, A, &m, B, &p, &zero, C, &m);
}
