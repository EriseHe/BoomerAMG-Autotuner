/* Standalone application adapter; no changes to the learner or active runner.
 * Settings match setup.space.DEFAULT_SETUP_PARAMS and the current
 * l1_jacobi_direct_coarse experiment profile. */
#include "native_default.h"
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_utilities.h"
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

int pilot_default(int n, const int *ia, const int *ja, const double *a,
                  const double *b, double *x, double tol, int maxit,
                  int cg, double *stats)
{
    static int initialized=0;
    if (!initialized) {
        int yes=0; MPI_Initialized(&yes);
        if (!yes) {int argc=0; char **argv=NULL; MPI_Init(&argc,&argv);}
        HYPRE_Init(); initialized=1;
    }
    int ranks=0; MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    if (ranks!=1 || n<1 || ia[0]!=0) return -1;
    HYPRE_ClearAllErrors();
    for(int i=0;i<6;i++) stats[i]=0;
    HYPRE_IJMatrix ijA=NULL;
    HYPRE_IJVector ijb=NULL,ijx=NULL;
    HYPRE_ParCSRMatrix A=NULL;
    HYPRE_ParVector B=NULL,X=NULL;
    HYPRE_Solver amg=NULL,pcg=NULL;
    HYPRE_BigInt *rows=malloc(n*sizeof(*rows));
    HYPRE_BigInt *cols=malloc(ia[n]*sizeof(*cols));
    HYPRE_Int *counts=malloc(n*sizeof(*counts));
    if(!rows || !cols || !counts) return -2;
    double t=MPI_Wtime();
    for(int i=0;i<n;i++){rows[i]=i; counts[i]=ia[i+1]-ia[i];}
    for(int i=0;i<ia[n];i++) cols[i]=ja[i];
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD,0,n-1,0,n-1,&ijA);
    HYPRE_IJMatrixSetObjectType(ijA,HYPRE_PARCSR);
    HYPRE_IJMatrixSetRowSizes(ijA,counts);
    HYPRE_IJMatrixInitialize(ijA);
    HYPRE_IJMatrixSetValues(ijA,n,counts,rows,cols,a);
    HYPRE_IJMatrixAssemble(ijA);
    HYPRE_IJMatrixGetObject(ijA,(void**)&A);
    HYPRE_IJVectorCreate(MPI_COMM_WORLD,0,n-1,&ijb);
    HYPRE_IJVectorSetObjectType(ijb,HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(ijb);
    HYPRE_IJVectorSetValues(ijb,n,rows,b);
    HYPRE_IJVectorAssemble(ijb);
    HYPRE_IJVectorGetObject(ijb,(void**)&B);
    HYPRE_IJVectorCreate(MPI_COMM_WORLD,0,n-1,&ijx);
    HYPRE_IJVectorSetObjectType(ijx,HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(ijx);
    HYPRE_IJVectorSetValues(ijx,n,rows,x);
    HYPRE_IJVectorAssemble(ijx);
    HYPRE_IJVectorGetObject(ijx,(void**)&X);
    stats[0]=MPI_Wtime()-t;
    HYPRE_BoomerAMGCreate(&amg);
    HYPRE_BoomerAMGSetPrintLevel(amg,0);
    HYPRE_BoomerAMGSetStrongThreshold(amg,0.25);
    HYPRE_BoomerAMGSetMaxRowSum(amg,0.90);
    HYPRE_BoomerAMGSetTruncFactor(amg,0.0);
    HYPRE_BoomerAMGSetCoarsenType(amg,10);
    HYPRE_BoomerAMGSetInterpType(amg,6);
    HYPRE_BoomerAMGSetPMaxElmts(amg,4);
    HYPRE_BoomerAMGSetAggNumLevels(amg,0);
    HYPRE_BoomerAMGSetAggInterpType(amg,4);
    HYPRE_BoomerAMGSetAggTruncFactor(amg,0.0);
    HYPRE_BoomerAMGSetAggPMaxElmts(amg,0);
    HYPRE_BoomerAMGSetRelaxType(amg,18);
    HYPRE_BoomerAMGSetCycleRelaxType(amg,18,1);
    HYPRE_BoomerAMGSetCycleRelaxType(amg,18,2);
    HYPRE_BoomerAMGSetCycleRelaxType(amg,9,3);
    HYPRE_BoomerAMGSetCycleType(amg,1);
    HYPRE_BoomerAMGSetCumNnzAP(amg,1.0);
    HYPRE_BoomerAMGSetTol(amg,cg?0.0:tol);
    HYPRE_BoomerAMGSetMaxIter(amg,cg?1:maxit);
    if(cg){
        HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD,&pcg);
        HYPRE_PCGSetTol(pcg,tol);
        HYPRE_PCGSetMaxIter(pcg,maxit);
        HYPRE_PCGSetTwoNorm(pcg,1);
        HYPRE_PCGSetPrintLevel(pcg,0);
        HYPRE_PCGSetLogging(pcg,1);
        HYPRE_ParCSRPCGSetPrecond(pcg,HYPRE_BoomerAMGSolve,HYPRE_BoomerAMGSetup,amg);
    }
    int err=HYPRE_GetError();
    if(!err){
        t=MPI_Wtime();
        err=cg?HYPRE_ParCSRPCGSetup(pcg,A,B,X):HYPRE_BoomerAMGSetup(amg,A,B,X);
        stats[1]=MPI_Wtime()-t;
        if(!err){
            t=MPI_Wtime();
            err=cg?HYPRE_ParCSRPCGSolve(pcg,A,B,X):HYPRE_BoomerAMGSolve(amg,A,B,X);
            stats[2]=MPI_Wtime()-t;
            stats[5]=err;
            HYPRE_ClearAllErrors();
            HYPRE_Int it=0; HYPRE_Real r=INFINITY;
            if(cg){HYPRE_PCGGetNumIterations(pcg,&it);HYPRE_PCGGetFinalRelativeResidualNorm(pcg,&r);}
            else {HYPRE_BoomerAMGGetNumIterations(amg,&it);HYPRE_BoomerAMGGetFinalRelativeResidualNorm(amg,&r);}
            stats[3]=r;stats[4]=it;
            HYPRE_IJVectorGetValues(ijx,n,rows,x);
        }
    }
    if(pcg)HYPRE_ParCSRPCGDestroy(pcg);
    if(amg)HYPRE_BoomerAMGDestroy(amg);
    if(ijx)HYPRE_IJVectorDestroy(ijx);
    if(ijb)HYPRE_IJVectorDestroy(ijb);
    if(ijA)HYPRE_IJMatrixDestroy(ijA);
    free(rows);free(cols);free(counts);
    HYPRE_ClearAllErrors();
    return err & ~HYPRE_ERROR_CONV;
}
