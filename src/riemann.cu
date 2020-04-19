#include "defs.h"
#include "cuda_defs.h"


__global__ void riemann_fluxes(real *UL, real *UR, real *F, 
        int dir1,int nx1, int nx2, int nx3, int size_x1, int size_x12,
        int nf,int ntot, int offset, real g) {
    int i,j,k,n,indx;
    real dL,uL,pL,eL,aL,uL2,uL3;
    real dR,uR,pR,eR,aR,uR2,uR3;
#ifdef EXACT
    int use_left;
    real ds,us,ps;
#endif

#ifdef HLLC
    real Ufac, SL, SR, Sstar;
#endif
    real g1 = g-1;
    real g2 = g1/(2*g);
    real g3 = g1/(g+1);
    real g4 = 2./(g+1);
    real g5 = g2/g3;

    int il,iu,jl,ju,kl,ku;

    int dir2, dir3; 

    /* 1->2->3 
     * 2->3->1
     * 3->1->2
     */
    dir2 = (dir1)%3 + 1;
    dir3 = (dir2)%3 + 1;

    if (dir1 == 1) {
        il = -1; iu = nx1+1;
        jl = -NGHX2; ju = nx2+NGHX2;
        kl = -NGHX3; ku = nx3 + NGHX3;
    }
    else if (dir1 == 2) {
        il = -NGHX1; iu = nx1+NGHX1;
        jl = -1; ju = nx2+1;
        kl = -NGHX3; ku = nx3 + NGHX3;
    }
    else {
    	il = -NGHX1; iu = nx1+NGHX1;
    	jl = -NGHX2; ju = nx2+NGHX2;
		kl = -1; ku = nx3 + 1;
    }
    for(indx = blockIdx.x*blockDim.x + threadIdx.x; indx<ntot; indx+=blockDim.x*gridDim.x) {
    	unpack_indices(indx,&i,&j,&k,size_x1,size_x12);
    	if ((i>=il)&&(i<iu)&&(j>=jl)&&(j<ju)&&(k>=kl)&&(k<ku))  {
            dL = UL[indx + 0*ntot];
            uL   = UL[indx + dir1*ntot]/dL;
            uL2  = UL[indx + dir2*ntot]/dL;
            uL3  = UL[indx + dir3*ntot]/dL;
            eL = UL[indx + 4*ntot] ;
#ifndef DUAL_ENERGY
            pL   = (eL- .5*(uL*uL + uL2*uL2 + uL3*uL3)*dL)*g1;
#else
            pL = E_from_S(UL[indx],UL[indx + 5*ntot],g1)*g1;
#endif
            if (pL < PRESSUREFLOOR) pL = PRESSUREFLOOR;

            dR = UR[indx + 0*ntot];
            uR   = UR[indx + dir1*ntot]/dR;
            uR2  = UR[indx + dir2*ntot]/dR;
            uR3  = UR[indx + dir3*ntot]/dR;
            eR = UR[indx + 4*ntot];
#ifndef DUAL_ENERGY
            pR   = (eR - .5*(uR*uR + uR2*uR2 + uR3*uR3)*dR)*g1;
#else
            pR = E_from_S(UR[indx],UR[indx + 5*ntot],g1)*g1;
#endif
            if (pR < PRESSUREFLOOR) pR = PRESSUREFLOOR;

            aL   = sqrt(g*pL/dL);
            aR   = sqrt(g*pR/dR);
#ifdef EXACT
            use_left = exact_sample(dL,uL,pL,aL,
                    dR,uR,pR,aR,
                    &ds,&us,&ps,
                    g,g1,g2,g3,g4,g5,0.,EXACT_TOL);

            F[indx + 0*ntot] = ds*us;
            F[indx + dir1*ntot] = ds*us*us + ps;
            if (use_left) {
                F[indx + dir2*ntot] = ds*us*uL2 ;
                F[indx + dir3*ntot] = ds*us*uL3 ;
                F[indx + 4*ntot] = us*( ps/g1 + .5*ds*(us*us + uL2*uL2 + uL3*uL3) + ps);
                for(n=5;n<nf;n++) {
                    F[indx + n*ntot] = ds*us*UL[indx + n*ntot]/dL;
                }
            }
            else {
                F[indx + dir2*ntot] = ds*us*uR2 ;
                F[indx + dir3*ntot] = ds*us*uR3 ;
                F[indx + 4*ntot] = us*( ps/g1 + .5*ds*(us*us + uR2*uR2 + uR3*uR3) + ps);
                for(n=5;n<nf;n++) {
                    F[indx + n*ntot] = ds*us*UR[indx + n*ntot]/dR;
                }
            }

#endif
#ifdef HLLC
            hllc(dL,uL,pL,aL,
                    dR,uR,pR,aR,
                    &SL,&SR,&Sstar,
                    g,g1,g2,g3,g4,g5);

            if (SL>=0) {
                F[indx + 0*ntot] = dL*uL;
                F[indx + dir1*ntot] = dL*uL*uL+pL;
                F[indx + dir2*ntot] = dL*uL*uL2;
                F[indx + dir3*ntot] = dL*uL*uL3;
                F[indx + 4*ntot] = (eL+pL)*uL;
                for(n=5;n<nf;n++) {
                    F[indx + n*ntot] =dL*uL*UL[indx+n*ntot]/dL;
                } 
            }
            else {
                if (SR <= 0) {
                    F[indx + 0*ntot] = dR*uR;
                    F[indx + dir1*ntot] = dR*uR*uR+pR;
                    F[indx + dir2*ntot] = dR*uR*uR2;
                    F[indx + dir3*ntot] = dR*uR*uR3;
                    F[indx + 4*ntot] = (eR+pR)*uR;
                    for(n=5;n<nf;n++) {
                        F[indx + n*ntot] =dR*uR*UR[indx+n*ntot]/dR;
                    } 
                }
                else {
                    if (Sstar >= 0) {
                        Ufac = dL*(SL-uL)/(SL-Sstar);
                        F[indx + 0*ntot] = dL*uL + 
                            SL*(Ufac - UL[indx+0*ntot]);
                        F[indx + dir1*ntot] = dL*uL*uL + pL +
                            SL*(Ufac * Sstar - UL[indx + dir1*ntot]);
                        F[indx + dir2*ntot] = dL*uL*uL2 + 
                            SL*(Ufac * uL2 - UL[indx + dir2*ntot]);
                        F[indx + dir3*ntot] = dL*uL*uL3 + 
                            SL*(Ufac * uL3 - UL[indx + dir3*ntot]);
                        F[indx + 4*ntot] = (eL+pL)*uL + 
                            SL*(Ufac *(eL/dL + (Sstar-uL)*(Sstar+pL/(dL*(SL-uL))))-UL[indx + 4*ntot]);
                        for(n=5;n<nf;n++) {
                            F[indx + n*ntot] = dL*uL*UL[indx + n*ntot]/dL + 
                                SL*(Ufac * UL[indx + n*ntot]/dL - UL[indx + n*ntot]);
                        }
                    }
                    else {
                        Ufac = dR*(SR-uR)/(SR-Sstar);
                        F[indx + 0*ntot] = dR*uR + 
                            SR*(Ufac - UR[indx+0*ntot]);
                        F[indx + dir1*ntot] = dR*uR*uR + pR +
                            SR*(Ufac * Sstar - UR[indx + dir1*ntot]);
                        F[indx + dir2*ntot] = dR*uR*uR2 + 
                            SR*(Ufac * uR2 - UR[indx + dir2*ntot]);
                        F[indx + dir3*ntot] = dR*uR*uR3 + 
                            SR*(Ufac * uR3 - UR[indx + dir3*ntot]);
                        F[indx + 4*ntot] = (eR+pR)*uR + 
                            SR*(Ufac *(eR/dR + (Sstar-uR)*(Sstar+pR/(dR*(SR-uR))))-UR[indx + 4*ntot]);
                        for(n=5;n<nf;n++) {
                            F[indx + n*ntot] = dR*uR*UR[indx + n*ntot]/dR + 
                                SR*(Ufac * UR[indx + n*ntot]/dR - UR[indx + n*ntot]);
                        }

                    }
                    
                }
            }
#endif


            
        }
    }
    

    return;

}
#ifdef RPROB
#ifdef EXACT


__global__ void sample_ic_kernel(real dL,real uL, real pL, real dR, real uR, real pR, real g, real x0,real t, real *x1, real *out,
        int size_x1) {
    int i,use_left;
    real S,aL,aR;
#ifdef EXACT
    real ds,us,ps;
#endif

    real g1 = g-1;
    real g2 = g1/(2*g);
    real g3 = g1/(g+1);
    real g4 = 2./(g+1);
    real g5 = g2/g3;

    int il,iu;


    il = 0; iu = size_x1;
    for(i = blockIdx.x*blockDim.x + threadIdx.x; i<size_x1; i+=blockDim.x*gridDim.x) {
    	if ((i>=il)&&(i<iu))  {
    		S = (x1[i]-x0)/t;

            aL   = sqrt(g*pL/dL);
            aR   = sqrt(g*pR/dR);

            use_left = exact_sample(dL,uL,pL,aL,
                    dR,uR,pR,aR,
                    &ds,&us,&ps,
                    g,g1,g2,g3,g4,g5,S,EXACT_TOL);
            out[i + 0*size_x1] = ds;
            out[i + 1*size_x1] = us;
            out[i + 2*size_x1] = ps;



        }
    }

    return;

}
void sample_ic(GridCons *grid, Parameters *params) {
	int size_x1 = grid->size_x1;
	int i;
	real *d_out,*d_x1;
	real *out = (real *)malloc(sizeof(real)*size_x1*3);
	for(i=0;i<3*size_x1;i++) out[i] =0.;


	cudaMalloc((void**)&d_x1,sizeof(real)*size_x1);
	cudaCheckError();
	cudaMemcpy(d_x1,&grid->xc1[-NGHX1],sizeof(real)*size_x1,cudaMemcpyHostToDevice);
	cudaCheckError();

	cudaMalloc((void**)&d_out,sizeof(real)*size_x1*3);
	cudaCheckError();
	cudaMemcpy(d_out,out,sizeof(real)*size_x1*3,cudaMemcpyHostToDevice);
	cudaCheckError();

	sample_ic_kernel<<<1, size_x1>>>(params->dl,params->ul,params->pl,
			params->dr,params->ur,params->pr,
			params->gamma,params->x0,params->tend,
			d_x1,d_out,size_x1);
	cudaCheckError();
    cudaMemcpy(out,d_out,sizeof(real)*3*size_x1,cudaMemcpyDeviceToHost);
    cudaCheckError();

    FILE *f;
    f = fopen("ic_sample.dat","w");
    fprintf(f,"#x\tdens\tvx\tpres\n");
    fprintf(f, "#dL=%lg,uL=%lg,pL=%lg,dR=%lg,uR=%lg,pR=%lg,g=%lg,t=%lg\n", params->dl,params->ul,params->pl,
    		params->dr,params->ur,params->pr,params->gamma,params->tend);
    for(i=0;i<size_x1;i++) {
    	fprintf(f, "%lg\t%lg\t%lg\t%lg\n", grid->xc1[i-NGHX1],out[i],out[i + size_x1], out[i+2*size_x1]);
    }
    fclose(f);

	cudaFree(d_x1); cudaCheckError();
	cudaFree(d_out); cudaCheckError();
	free(out);
}
#endif
#endif
