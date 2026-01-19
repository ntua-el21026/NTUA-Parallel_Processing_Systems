#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "mpi.h"
#include "utils.h"

int main(int argc, char ** argv) {
    int rank,size;
    int global[2],local[2]; //global matrix dimensions and local matrix dimensions
    int global_padded[2];   //padded global matrix dimensions
    int grid[2];            //processor grid dimensions
    int i,j,t;
    int global_converged=0,converged=0; //flags for convergence
    MPI_Datatype dummy;     //dummy datatype
    double omega;           //relaxation factor

    struct timeval tts,ttf,tcs,tcf;   //Timers
    double ttotal=0,tcomp=0,total_time,comp_time;
    
    double ** U, ** u_current, ** u_previous, ** swap; 

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    //----Read arguments----//
    if (argc!=5) {
        fprintf(stderr,"Usage: mpirun .... ./exec X Y Px Py");
        exit(-1);
    }
    else {
        global[0]=atoi(argv[1]);
        global[1]=atoi(argv[2]);
        grid[0]=atoi(argv[3]);
        grid[1]=atoi(argv[4]);
    }

    //----Create 2D-cartesian communicator----//
    MPI_Comm CART_COMM;         
    int periods[2]={0,0};       
    int rank_grid[2];           
        
    MPI_Cart_create(MPI_COMM_WORLD,2,grid,periods,0,&CART_COMM);    
    MPI_Cart_coords(CART_COMM,rank,2,rank_grid);                    

    //----Compute local dimensions & Padding----//
    for (i=0;i<2;i++) {
        if (global[i]%grid[i]==0) {
            local[i]=global[i]/grid[i];
            global_padded[i]=global[i];
        }
        else {
            local[i]=(global[i]/grid[i])+1;
            global_padded[i]=local[i]*grid[i];
        }
    }

    //Initialization of omega
    omega=2.0/(1+sin(3.14/global[0]));

    //----Allocate global 2D-domain----//
    if (rank==0) {
        U=allocate2d(global_padded[0],global_padded[1]);   
        init2d(U,global[0],global[1]);
    }

    //----Allocate local 2D-subdomains----//
    u_previous=allocate2d(local[0]+2,local[1]+2);
    u_current=allocate2d(local[0]+2,local[1]+2);   
       
    //----Datatypes Definition----//
    MPI_Datatype global_block;
    MPI_Type_vector(local[0],local[1],global_padded[1],MPI_DOUBLE,&dummy);
    MPI_Type_create_resized(dummy,0,sizeof(double),&global_block);
    MPI_Type_commit(&global_block);

    MPI_Datatype local_block;
    MPI_Type_vector(local[0],local[1],local[1]+2,MPI_DOUBLE,&dummy);
    MPI_Type_create_resized(dummy,0,sizeof(double),&local_block);
    MPI_Type_commit(&local_block);

    //----Scatter parameters----//
    int * scatteroffset, * scattercounts;
    if (rank==0) {
        scatteroffset=(int*)malloc(size*sizeof(int));
        scattercounts=(int*)malloc(size*sizeof(int));
        for (i=0;i<grid[0];i++)
            for (j=0;j<grid[1];j++) {
                scattercounts[i*grid[1]+j]=1;
                scatteroffset[i*grid[1]+j]=(local[0]*local[1]*grid[1]*i+local[1]*j);
            }
    }

    //----Scatter----//
    MPI_Scatterv(rank == 0 ? &U[0][0] : NULL, 
                 scattercounts, scatteroffset, global_block, 
                 &u_previous[1][1], 1, local_block, 
                 0, MPI_COMM_WORLD);

    // Init u_current
    for (i = 1; i <= local[0]; i++)
        for (j = 1; j <= local[1]; j++)
            u_current[i][j] = u_previous[i][j];

    if (rank==0)
        free2d(U);

    //----Communication Datatypes----//
    MPI_Datatype row_type, col_type;
    MPI_Type_contiguous(local[1], MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);
    MPI_Type_vector(local[0], 1, local[1] + 2, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    //----Find Neighbors----//
    int north, south, east, west;
    MPI_Cart_shift(CART_COMM, 0, 1, &north, &south);
    MPI_Cart_shift(CART_COMM, 1, 1, &west, &east);

    //---Define iteration ranges-----//
    int i_min,i_max,j_min,j_max;

    i_min = 1;
    i_max = local[0];
    j_min = 1;
    j_max = local[1];

    if (rank_grid[0] == 0) i_min = 2; 
    if (rank_grid[0] == grid[0] - 1) {
        i_max = local[0] - (global_padded[0] - global[0]) - 1;
        if (global_padded[0] == global[0]) i_max = local[0] - 1;
    }

    if (rank_grid[1] == 0) j_min = 2;
    if (rank_grid[1] == grid[1] - 1) {
         j_max = local[1] - (global_padded[1] - global[1]) - 1;
         if (global_padded[1] == global[1]) j_max = local[1] - 1;
    }

    // Calculate global offsets to determine Red/Black parity correctly across processes
    // Since padding ensures equal local sizes:
    int global_i_offset = rank_grid[0] * local[0]; 
    int global_j_offset = rank_grid[1] * local[1];

    //----Computational core----//   
    gettimeofday(&tts, NULL);

    #ifdef TEST_CONV
    for (t=0;t<T && !global_converged;t++) {
    #endif
    #ifndef TEST_CONV
    #undef T
    #define T 256
    for (t=0;t<T;t++) {
    #endif

        // 1. Swap
        swap = u_previous;
        u_previous = u_current;
        u_current = swap;

        // 2. Communication (Halo Exchange)
        MPI_Request reqs[8];
        MPI_Status stats[8];
        int req_cnt = 0;

        MPI_Isend(&u_previous[1][1], 1, row_type, north, 1, CART_COMM, &reqs[req_cnt++]);
        MPI_Irecv(&u_previous[0][1], 1, row_type, north, 2, CART_COMM, &reqs[req_cnt++]);

        MPI_Isend(&u_previous[local[0]][1], 1, row_type, south, 2, CART_COMM, &reqs[req_cnt++]);
        MPI_Irecv(&u_previous[local[0]+1][1], 1, row_type, south, 1, CART_COMM, &reqs[req_cnt++]);

        MPI_Isend(&u_previous[1][1], 1, col_type, west, 3, CART_COMM, &reqs[req_cnt++]);
        MPI_Irecv(&u_previous[1][0], 1, col_type, west, 4, CART_COMM, &reqs[req_cnt++]);

        MPI_Isend(&u_previous[1][local[1]], 1, col_type, east, 4, CART_COMM, &reqs[req_cnt++]);
        MPI_Irecv(&u_previous[1][local[1]+1], 1, col_type, east, 3, CART_COMM, &reqs[req_cnt++]);

        MPI_Waitall(req_cnt, reqs, stats);

        // 3. Computation (Red-Black SOR)
        gettimeofday(&tcs, NULL); 

        // --- RED PHASE ---
        // Calculate Red cells ((i+j) is even). 
        // Reads from u_previous (Black neighbors).
        for (i = i_min; i <= i_max; i++) {
            for (j = j_min; j <= j_max; j++) {
                // Check parity using global coordinates
                if ( ((global_i_offset + i) + (global_j_offset + j)) % 2 == 0 ) {
                    u_current[i][j] = u_previous[i][j] + 
                                      (omega / 4.0) * (u_previous[i-1][j] + u_previous[i][j-1] +
                                                       u_previous[i+1][j] + u_previous[i][j+1] -
                                                       4.0 * u_previous[i][j]);
                } else {
                    // Copy Black cells to current (needed so Black phase can read them if accessed)
                    u_current[i][j] = u_previous[i][j];
                }
            }
        }

        // --- BLACK PHASE ---
        // Calculate Black cells ((i+j) is odd). 
        // Reads from u_current (Red neighbors - NEW values) and u_previous (self).
        for (i = i_min; i <= i_max; i++) {
            for (j = j_min; j <= j_max; j++) {
                if ( ((global_i_offset + i) + (global_j_offset + j)) % 2 == 1 ) {
                    u_current[i][j] = u_previous[i][j] + 
                                      (omega / 4.0) * (u_current[i-1][j] + u_current[i][j-1] +
                                                       u_current[i+1][j] + u_current[i][j+1] -
                                                       4.0 * u_previous[i][j]);
                }
            }
        }

        gettimeofday(&tcf, NULL); 
        tcomp += (tcf.tv_sec - tcs.tv_sec) + (tcf.tv_usec - tcs.tv_usec) * 0.000001;

        // 4. Convergence Check
        #ifdef TEST_CONV
        if (t % C == 0) {
            converged = converge(u_previous, u_current, i_min, i_max, j_min, j_max);
            MPI_Allreduce(&converged, &global_converged, 1, MPI_INT, MPI_LAND, CART_COMM);
        }       
        #endif
        
    }
    gettimeofday(&ttf,NULL);

    ttotal=(ttf.tv_sec-tts.tv_sec)+(ttf.tv_usec-tts.tv_usec)*0.000001;

    MPI_Reduce(&ttotal,&total_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&tcomp,&comp_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    //----Gather results----//
    if (rank==0) {
            U=allocate2d(global_padded[0],global_padded[1]);
    }

    MPI_Gatherv(&u_current[1][1], 1, local_block, 
                rank == 0 ? &U[0][0] : NULL, 
                scattercounts, scatteroffset, global_block, 
                0, MPI_COMM_WORLD);

    //----Printing results----//
    if (rank==0) {
        printf("RedBlackSOR X %d Y %d Px %d Py %d Iter %d ComputationTime %lf TotalTime %lf midpoint %lf\n",
                global[0],global[1],grid[0],grid[1],t,comp_time,total_time,U[global[0]/2][global[1]/2]);
    
        #ifdef PRINT_RESULTS
        char * s=malloc(50*sizeof(char));
        sprintf(s,"resRedBlackMPI_%dx%d_%dx%d",global[0],global[1],grid[0],grid[1]);
        fprint2d(s,U,global[0],global[1]);
        free(s);
        #endif
    }

    // Free Datatypes before Finalize
    MPI_Type_free(&row_type);
    MPI_Type_free(&col_type);
    MPI_Type_free(&global_block);
    MPI_Type_free(&local_block);
    if(rank==0){
        free(scattercounts);
        free(scatteroffset);
    }

    MPI_Finalize();
    return 0;
}