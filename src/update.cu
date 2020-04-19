#include "defs.h"
#include "cuda_defs.h"

__global__ void compute_dhalf(real *cons, real *dhalf, real *F_1, real *F_2,real *F_3,
        real *dx1, real *dx2, real *dx3, real dt, int nx1, int nx2, int nx3, int size_x1, int size_x12, int ntot, int offset, int nf) {
    int i,j,k;
    int indx;


    for(indx = blockIdx.x*blockDim.x + threadIdx.x; indx<ntot; indx+=blockDim.x*gridDim.x) {
    	unpack_indices(indx,&i,&j,&k,size_x1,size_x12);
        if ((i>=-2)&&(i<nx1+2)&&(j>=-2)&&(j<nx2+2)&&(k>=-2)&&(k<nx3+2)) {
            dhalf[indx] = cons[indx] +  .5*dt/dx1[i]*(F_1[indx - 1]        - F_1[indx]);
#ifdef DIMS2
            dhalf[indx] += .5*dt/dx2[j]*(F_2[indx - size_x1]  - F_2[indx]);
#endif
#ifdef DIMS3
            dhalf[indx] += .5*dt/dx3[k]*(F_3[indx - size_x12] - F_3[indx]);
#endif
        }
    }
    return;

}

__global__ void update_cons(real *cons, real *intenergy, real *F_1, real *F_2, real *F_3,
        real *dx1, real *dx2, real *dx3, real g1, real dt, int nx1, int nx2, int nx3, int size_x1, int size_x12, int ntot, int offset, int nf) {
    int i,j,k,n;
    int indx;
    int nan_check;
    real dtdx1;
#ifdef DIMS2
    real dtdx2;
#endif
#ifdef DIMS3
    real dtdx3;
#endif
#ifdef DUAL_ENERGY
    real Ein,Ek;
#endif

    for(indx = blockIdx.x*blockDim.x + threadIdx.x; indx<ntot; indx+=blockDim.x*gridDim.x) {
    	unpack_indices(indx,&i,&j,&k,size_x1,size_x12);
        if ((i>=0)&&(i<nx1)&&(j>=0)&&(j<nx2)&&(k>=0)&&(k<nx3)) {

            dtdx1 = dt/dx1[i];
#ifdef DIMS2
            dtdx2 = dt/dx2[j];
#endif
#ifdef DIMS3
            dtdx3 = dt/dx3[k];
#endif
            for(n=0;n<nf;n++) {
            	//printf("%d (%d,%d), %lg \n",n,i,j,F_1[indx + n*ntot]);
                cons[indx + n*ntot] += dtdx1*(F_1[indx - 1        + n*ntot]- F_1[indx + n*ntot]);
#ifdef DIMS2
                cons[indx + n*ntot] += dtdx2*(F_2[indx - size_x1  + n*ntot]- F_2[indx + n*ntot]);
#endif
#ifdef DIMS3
                cons[indx + n*ntot] += dtdx3*(F_3[indx - size_x12 + n*ntot]- F_3[indx + n*ntot]);
#endif

            }
#ifdef DUAL_ENERGY
            Ek = .5*(
                    cons[indx + 1*ntot]*cons[indx + 1*ntot] +
                    cons[indx + 2*ntot]*cons[indx + 2*ntot] +
                    cons[indx + 3*ntot]*cons[indx + 3*ntot])/cons[indx];
            Ein = cons[indx + 4*ntot] - Ek;
            if (Ein < DETOL*Ek) {
                /* Set Ein from S */
                Ein = E_from_S(cons[indx], cons[indx+5*ntot],g1);
                printf("d=%lg S=%lg Ein=%lg Ek=%lg \n", cons[indx], cons[indx + 5*ntot], Ein, Ek);
                cons[indx + 4*ntot] = Ein + Ek;
                intenergy[indx] = Ein;
            }

            else {
                /* Sync S with Ein */
                cons[indx + 5*ntot] = S_from_E(cons[indx], Ein, g1);
                intenergy[indx] = Ein;
            }

#else
            intenergy[indx] = cons[indx+4*ntot] - .5*(
                    cons[indx + 1*ntot]*cons[indx + 1*ntot] +
                    cons[indx + 2*ntot]*cons[indx + 2*ntot] +
                    cons[indx + 3*ntot]*cons[indx + 3*ntot])/cons[indx];
#endif

        }
    }
    return;

}
__global__ void transverse_update(real *UL_1, real *UL_2, real *UL_3,
        real *UR_1, real *UR_2, real *UR_3,
        real *F_1, real *F_2, real *F_3, real *dx1, real *dx2, real *dx3, real g1, real dt,
        int nx1, int nx2, int nx3, int size_x1, int size_x12, int ntot, int offset, int nf) {

	/*
	 * 			G(i,j)			G(i+1,j)
	 *		+	+	+	+	+	+	+	+	+
	 *		+				+				+
	 *		+		UL(i,j)	+ UR(i,j)		+
	 *		+	  			+	 			+
	 *		+	  (i,j)		+ 	  (i+1,j)	+
	 *		+				+ 				+
	 * 		+	+	+	+	+	+	+	+	+
	 *			G(i,j-1)		G(i+1,j-1)
	 *
	 *
	 */
    int i,j,k,n;
    int indx;
    real dtdx;
#ifdef DUAL_ENERGY
    real Ein,Ek;
#endif

    for(indx = blockIdx.x*blockDim.x + threadIdx.x; indx<ntot;indx+=blockDim.x*gridDim.x) {
    	unpack_indices(indx,&i,&j,&k,size_x1,size_x12);
        /* X1 - direction */
        if ((i>=-2)&&(i<nx1+2)&&(j>=-2)&&(j<nx2+3)&&(k>=-2)&&(k<nx3+3)) {
            dtdx = .5*dt/dx2[j];
            for(n=0;n<nf;n++) {
                UL_1[indx + n*ntot] += dtdx*(F_2[indx - size_x1     + n*ntot] -F_2[indx     + n*ntot]);
                UR_1[indx + n*ntot] += dtdx*(F_2[indx - size_x1 + 1 + n*ntot] -F_2[indx + 1 + n*ntot]);

            }
            /* Add X3 flux */
#ifdef DIMS3
			dtdx = .5*dt/dx3[k];
			for(n=0;n<nf;n++) {
				UL_1[indx + n*ntot] += dtdx*(F_3[indx - size_x12     + n*ntot] -F_3[indx     + n*ntot]);
				UR_1[indx + n*ntot] += dtdx*(F_3[indx - size_x12 + 1 + n*ntot] -F_3[indx + 1 + n*ntot]);

			}

#endif
#ifdef DUAL_ENERGY
            Ek = .5*(
                    UL_1[indx + 1*ntot]*UL_1[indx + 1*ntot] +
                    UL_1[indx + 2*ntot]*UL_1[indx + 2*ntot] +
                    UL_1[indx + 3*ntot]*UL_1[indx + 3*ntot])/UL_1[indx];
            Ein = UL_1[indx + 4*ntot] - Ek;
            if (Ein < DETOL*Ek) {
                /* Set Ein from S */
                Ein = E_from_S(UL_1[indx], UL_1[indx+5*ntot],g1);
                UL_1[indx + 4*ntot] = Ein + Ek;
            }
            else {
                /* Sync S with Ein */
                UL_1[indx + 5*ntot] = S_from_E(UL_1[indx], Ein, g1);
            }
            Ek = .5*(
                    UR_1[indx + 1*ntot]*UR_1[indx + 1*ntot] +
                    UR_1[indx + 2*ntot]*UR_1[indx + 2*ntot] +
                    UR_1[indx + 3*ntot]*UR_1[indx + 3*ntot])/UR_1[indx];
            Ein = UR_1[indx + 4*ntot] - Ek;
            if (Ein < DETOL*Ek) {
                /* Set Ein from S */
                Ein = E_from_S(UR_1[indx], UR_1[indx+5*ntot],g1);
                UR_1[indx + 4*ntot] = Ein + Ek;
            }
            else {
                /* Sync S with Ein */
                UR_1[indx + 5*ntot] = S_from_E(UR_1[indx], Ein, g1);
            }
#endif
        }
        /* X2 - direction */
        if ((i>=-2)&&(i<nx1+3)&&(j>=-2)&&(j<nx2+2)&&(k>=-2)&&(k<nx3+3)) {
            /* Add X1 flux */
            dtdx = .5*dt/dx1[i];
            for(n=0;n<nf;n++) {
                UL_2[indx + n*ntot] += dtdx*(F_1[indx - 1           + n*ntot] -F_1[indx           + n*ntot]);
                UR_2[indx + n*ntot] += dtdx*(F_1[indx - 1 + size_x1 + n*ntot] -F_1[indx + size_x1 + n*ntot]);

            }
            /* Add X3 flux */
#ifdef DIMS3
                /* Add X1 flux */
			dtdx = .5*dt/dx3[k];
			for(n=0;n<nf;n++) {
				UL_2[indx + n*ntot] += dtdx*(F_3[indx - size_x12           + n*ntot] -F_3[indx           + n*ntot]);
				UR_2[indx + n*ntot] += dtdx*(F_3[indx - size_x12 + size_x1 + n*ntot] -F_3[indx + size_x1 + n*ntot]);

			}

#endif
#ifdef DUAL_ENERGY
            Ek = .5*(
                    UL_2[indx + 1*ntot]*UL_2[indx + 1*ntot] +
                    UL_2[indx + 2*ntot]*UL_2[indx + 2*ntot] +
                    UL_2[indx + 3*ntot]*UL_2[indx + 3*ntot])/UL_2[indx];
            Ein = UL_2[indx + 4*ntot] - Ek;
            if (Ein < DETOL*Ek) {
                /* Set Ein from S */
                Ein = E_from_S(UL_2[indx], UL_2[indx+5*ntot],g1);
                UL_2[indx + 4*ntot] = Ein + Ek;
            }
            else {
                /* Sync S with Ein */
                UL_2[indx + 5*ntot] = S_from_E(UL_2[indx], Ein, g1);
            }
            Ek = .5*(
                    UR_2[indx + 1*ntot]*UR_2[indx + 1*ntot] +
                    UR_2[indx + 2*ntot]*UR_2[indx + 2*ntot] +
                    UR_2[indx + 3*ntot]*UR_2[indx + 3*ntot])/UR_2[indx];
            Ein = UR_2[indx + 4*ntot] - Ek;
            if (Ein < DETOL*Ek) {
                /* Set Ein from S */
                Ein = E_from_S(UR_2[indx], UR_2[indx+5*ntot],g1);
                UR_2[indx + 4*ntot] = Ein + Ek;
            }
            else {
                /* Sync S with Ein */
                UR_2[indx + 5*ntot] = S_from_E(UR_2[indx], Ein, g1);
            }
#endif
        }
        /* X3 - direction */
#ifdef DIMS3
        if ((i>=-2)&&(i<nx1+3)&&(j>=-2)&&(j<nx2+3)&&(k>=-2)&&(k<nx3+2)) {
            /* Add X1 flux */
            dtdx = .5*dt/dx1[i];
            for(n=0;n<nf;n++) {
                UL_3[indx + n*ntot] += dtdx*(F_1[indx - 1            + n*ntot] -F_1[indx            + n*ntot]);
                UR_3[indx + n*ntot] += dtdx*(F_1[indx - 1 + size_x12 + n*ntot] -F_1[indx + size_x12 + n*ntot]);

            }
            /* Add X2 flux */
			dtdx = .5*dt/dx2[j];
			for(n=0;n<nf;n++) {
				UL_3[indx + n*ntot] += dtdx*(F_2[indx - size_x1            + n*ntot] -F_2[indx            + n*ntot]);
				UR_3[indx + n*ntot] += dtdx*(F_2[indx - size_x1 + size_x12 + n*ntot] -F_2[indx + size_x12 + n*ntot]);

			}


        }
#ifdef DUAL_ENERGY
            Ek = .5*(
                    UL_3[indx + 1*ntot]*UL_3[indx + 1*ntot] +
                    UL_3[indx + 2*ntot]*UL_3[indx + 2*ntot] +
                    UL_3[indx + 3*ntot]*UL_3[indx + 3*ntot])/UL_3[indx];
            Ein = UL_3[indx + 4*ntot] - Ek;
            if (Ein < DETOL*Ek) {
                /* Set Ein from S */
                Ein = E_from_S(UL_3[indx], UL_3[indx+5*ntot],g1);
                UL_3[indx + 4*ntot] = Ein + Ek;
            }
            else {
                /* Sync S with Ein */
                UL_3[indx + 5*ntot] = S_from_E(UL_3[indx], Ein, g1);
            }
            Ek = .5*(
                    UR_3[indx + 1*ntot]*UR_3[indx + 1*ntot] +
                    UR_3[indx + 2*ntot]*UR_3[indx + 2*ntot] +
                    UR_3[indx + 3*ntot]*UR_3[indx + 3*ntot])/UR_3[indx];
            Ein = UR_3[indx + 4*ntot] - Ek;
            if (Ein < DETOL*Ek) {
                /* Set Ein from S */
                Ein = E_from_S(UR_3[indx], UR_3[indx+5*ntot],g1);
                UR_3[indx + 4*ntot] = Ein + Ek;
            }
            else {
                /* Sync S with Ein */
                UR_3[indx + 5*ntot] = S_from_E(UR_3[indx], Ein, g1);
            }
#endif
#endif
    }
    return;

}

__global__ void cons_to_prim(real *cons, real *intenergy, real *prim, real g1,
        int nx1, int nx2, int nx3, int size_x1, int size_x12, int ntot, int offset, int nf) {
    int i,j,k,n;
    int indx;

    for(indx = blockIdx.x*blockDim.x + threadIdx.x; indx<ntot; indx+=blockDim.x*gridDim.x) {
    	unpack_indices(indx,&i,&j,&k,size_x1,size_x12);
        if ((i>=-NGHX1)&&(i<nx1+NGHX1)&&(j>=-NGHX2)&&(j<nx2+NGHX2)&&(k>=-NGHX3)&&(k<nx3+NGHX3)) {

        	prim[indx + 0 *ntot] = cons[indx + 0*ntot];
        	prim[indx + 1 *ntot] = cons[indx + 1*ntot]/cons[indx];
        	prim[indx + 2 *ntot] = cons[indx + 2*ntot]/cons[indx];
        	prim[indx + 3 *ntot] = cons[indx + 3*ntot]/cons[indx];
        	for(n=5;n<nf;n++) prim[indx + n*ntot] = cons[indx + n*ntot]/cons[indx];
#ifdef DUAL_ENERGY
            prim[indx + 4*ntot] = E_from_S(cons[indx],cons[indx +5*ntot],g1)*g1;
#else
        	prim[indx + 4 *ntot] = intenergy[indx] * g1;
#endif



        }
    }
    return;

}
__global__ void prim_to_cons(real *cons, real *intenergy, real *prim, real g1,
        int nx1, int nx2, int nx3, int size_x1, int size_x12, int ntot, int offset, int nf) {
    int i,j,k,n;
    int indx;

    for(indx = blockIdx.x*blockDim.x + threadIdx.x; indx<ntot; indx+=blockDim.x*gridDim.x) {
    	unpack_indices(indx,&i,&j,&k,size_x1,size_x12);
        if ((i>=-NGHX1)&&(i<nx1+NGHX1)&&(j>=-NGHX2)&&(j<nx2+NGHX2)&&(k>=-NGHX3)&&(k<nx3+NGHX3)) {

        	cons[indx + 0 *ntot] = prim[indx + 0*ntot];
        	cons[indx + 1 *ntot] = prim[indx + 1*ntot]*prim[indx];
        	cons[indx + 2 *ntot] = prim[indx + 2*ntot]*prim[indx];
        	cons[indx + 3 *ntot] = prim[indx + 3*ntot]*prim[indx];
        	intenergy[indx     ]      = prim[indx + 4*ntot]/g1;
        	cons[indx + 4 *ntot] = intenergy[indx] + .5*prim[indx]*(prim[indx + 1*ntot]*prim[indx + 1*ntot]
        	                                                      + prim[indx + 2*ntot]*prim[indx + 2*ntot]
        	                                                      + prim[indx + 3*ntot]*prim[indx + 3*ntot]);
        	for(n=5;n<nf;n++) cons[indx + n*ntot] = prim[indx + n*ntot]*prim[indx];



        }
    }
    return;

}

