#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>

extern "C" {
#ifdef MATLAB_MEX_FILE
    #include "mex.h"
#else
    #include "utils.h"
#endif
}

#include <cub/cub.cuh>

#define BUFSIZE 100
#define checkCuda(result) __checkCuda(result, __FILE__, __LINE__)
inline cudaError_t __checkCuda(cudaError_t result, const char *file, const int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%i: %s\n", file, line, cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

#define BLOCK_X   32
#define BLOCK_Y    8
#define BLOCK_1D 512

void send_help(char *program) {
    fprintf(stderr, "Usage: %s [options] <n> <m> <pop_file>\n", program);
    fprintf(stderr, "Where:\n");
    fprintf(stderr, "  n         Population size.\n");
    fprintf(stderr, "  m         Number of objectives.\n");
    fprintf(stderr, "  pop_file  Path to the file containing the population.\n\n");

    fprintf(stderr, "Optional parameters:\n");
    fprintf(stderr, "  -h  Show this help\n");
    fprintf(stderr, "  -v  Verbosity level: 0 (only nds runtime)\n");
    fprintf(stderr, "                       1 (default, only front counts)\n");
    fprintf(stderr, "                       2 (front counts and contents).\n");
}

double get_time() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
        return ts.tv_sec * 1000 + ts.tv_nsec / 1000000.0;
    else
        return 0;
}

__global__ void init_indexes(int n, int m, int *indexes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n*m) {
        indexes[i] = i % n;
    }
}

__global__ void transpose(int nx, int ny, float *tr_pop, const float *pop) {
    __shared__ float tile[BLOCK_X][BLOCK_X+1];

    int i = blockIdx.x * BLOCK_X + threadIdx.x;
    int j = blockIdx.y * BLOCK_X + threadIdx.y;

    if (i < nx) {
        for (int k = 0; k < BLOCK_X; k+= BLOCK_Y) {
            if ((j + k) >= ny) {
                break;
            }
            tile[threadIdx.y + k][threadIdx.x] = pop[(j + k)*nx + i];
        }
    }

    __syncthreads();

    i = blockIdx.y * BLOCK_X + threadIdx.x;
    j = blockIdx.x * BLOCK_X + threadIdx.y;

    if (i < ny) {
        for (int k = 0; k < BLOCK_X; k+= BLOCK_Y) {
            if ((j + k) >= nx) {
                break;
            }
            tr_pop[(j + k)*ny + i] = tile[threadIdx.x][threadIdx.y + k];
        }
    }

}

__global__ void init_int_array(int value, int n, int *array) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        array[i] = value;
    }
}

__global__ void init_int_array_limited(int value, int n, int *array) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        array[i] = value;
    }
}

__global__ void find_positions(int n, int m, int *sorted_idx, int *positions, float *tr_pop) {     
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < n & j < m) {
        int idx = sorted_idx[j*n + i];
        float my_val = tr_pop[j*n + i];
        int next = 1;
        while (i + next < n && my_val == tr_pop[j*n + i + next]) {
            next++;
        }
        positions[j*n + idx] = i + ((next > 1) * next);
    }
}

__global__ void best_objective(int n, int m, int *positions, int *best_m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int min_pos = positions[i];
        int obj = 0;
        for (int j = 1; j < m; j++) {
            int pos = positions[j*n + i];
            if (pos < min_pos) {
                min_pos = pos;
                obj = j;
            }
        }
        best_m[i] = min_pos;
        best_m[n + i] = obj;
    }
}

__device__ int is_dominated(int n, int m, int a, int b, const float *tr_pop) {
    bool equal = true;
    for (int j = 0; j < m; j++) {
        if (tr_pop[j*n + a] < tr_pop[j*n + b]) {
            return 0;
        }
        equal = equal & !(tr_pop[j*n + a] > tr_pop[j*n + b]);
    }
    return !equal;
}

__device__ void domination_check_inner(int n, int m, int idx, const float *tr_pop,
                                 const int *sorted_idx, const int *best_m,
                                 int *dominating_idx, int *last_batch,
                                 int *final_ranks, int curr_rank) {

    __shared__  bool blk_dominated;

    if (final_ranks[idx] >= 0) {
        return;
    }
    if (threadIdx.x == 0) {
        blk_dominated = false;
    }

    __syncthreads();
    
    const int i = idx*blockDim.x + threadIdx.x;
    int dom_idx = dominating_idx[i];
    if (dom_idx >= 0) {
        int dom_rank = final_ranks[dom_idx];
        if (dom_rank >= 0 && dom_rank < curr_rank) {
            dominating_idx[i] = -1;
        } else {
            blk_dominated = true;
        }
    }

    __syncthreads();

    if (blk_dominated) {
        return;
    }

    const int pos = best_m[idx];
    const int offset = best_m[n + idx]*n;
    for (int j = last_batch[idx]; j < pos; j += blockDim.x) {

        __syncthreads();

        if (j + threadIdx.x < pos) {
            int j_idx = sorted_idx[offset + j + threadIdx.x];

            bool is_candidate = true;
            if (curr_rank > 0) {
                int j_rank = final_ranks[j_idx];
                if (j_rank >= 0 && j_rank < curr_rank) {
                    is_candidate = false;
                }
            }
            if (is_candidate && is_dominated(n, m, idx, j_idx, tr_pop)) {
                dominating_idx[i] = j_idx;
                blk_dominated = true;
            }
        }

        __syncthreads();

        if (blk_dominated) {
            if (threadIdx.x == 0) {
                last_batch[idx] = j + blockDim.x;
            }
            return;
        }
    }

    if (threadIdx.x == 0) {
        final_ranks[idx] = curr_rank;
    }
}

__global__ void domination_check(int n, int m, const float *tr_pop,
                                 const int *sorted_idx, const int *best_m,
                                 int *dominating_idx, int *last_batch,
                                 int *final_ranks, int curr_rank) {

    for (int idx = blockIdx.x; idx < n; idx += gridDim.x) {
        domination_check_inner(n, m, idx, tr_pop, sorted_idx, best_m, dominating_idx, last_batch, final_ranks, curr_rank);
        __syncthreads();
    }
}

void nds(int n, int m, float *h_pop, int verbosity, int *h_ranks) {
    int count = n*m;

    // Transpose population matrix
    float *d_pop, *d_tr_pop;
    checkCuda(cudaMalloc(&d_pop, count*sizeof(float)));
    checkCuda(cudaMalloc(&d_tr_pop, count*sizeof(float)));
    checkCuda(cudaMemcpy(d_pop, h_pop, count*sizeof(float), cudaMemcpyHostToDevice));
    {
        dim3 block(BLOCK_X, BLOCK_Y);
        dim3 grid((m + block.x - 1)/block.x, 
                  (ceil(n*BLOCK_Y/(float)BLOCK_X) + block.y - 1)/block.y);
        transpose<<<grid, block>>>(m, n, d_tr_pop, d_pop);
        checkCuda(cudaGetLastError());
    }
    checkCuda(cudaFree(d_pop));

    // Initialize population indexes
    int *d_sorted_idx;
    checkCuda(cudaMalloc(&d_sorted_idx, count*sizeof(int)));
    {
        dim3 block(BLOCK_1D);
        dim3 grid((count + block.x - 1)/block.x);
        init_indexes<<<grid, block>>>(n, m, d_sorted_idx);
        checkCuda(cudaGetLastError());
    }

    float *d_tr_pop_out;
    checkCuda(cudaMalloc(&d_tr_pop_out, count*sizeof(float)));
    int *d_sorted_idx_out;
    checkCuda(cudaMalloc(&d_sorted_idx_out, count*sizeof(int)));
    {
        cudaStream_t *streams = (cudaStream_t *) malloc(m*sizeof(cudaStream_t));
        for (int j = 0; j < m; j++) {
            checkCuda(cudaStreamCreate(&streams[j]));
        }

        // Determine temporary device storage requirements
        char     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                     d_tr_pop, d_tr_pop_out, d_sorted_idx, d_sorted_idx_out, n);

        // Allocate temporary storage
        checkCuda(cudaMalloc(&d_temp_storage, m*temp_storage_bytes));

        for (int j = 0; j < m; j++) {
            // Run sorting operation
            cub::DeviceRadixSort::SortPairs(&d_temp_storage[j*temp_storage_bytes], temp_storage_bytes,
                        &d_tr_pop[j*n], &d_tr_pop_out[j*n], &d_sorted_idx[j*n], &d_sorted_idx_out[j*n], n, 0, sizeof(float)*8, streams[j]);
        }
        checkCuda(cudaFree(d_temp_storage));
        
        for (int j = 0; j < m; j++) {
            checkCuda(cudaStreamSynchronize(streams[j]));
            checkCuda(cudaStreamDestroy(streams[j]));
        }
        checkCuda(cudaFree(d_sorted_idx));
    }

    d_sorted_idx = d_sorted_idx_out;

    // Find out in which list each individual is higher.
    int *d_best_m;
    checkCuda(cudaMalloc(&d_best_m, 2*n*sizeof(int)));
    {
        dim3 block(BLOCK_1D);
        dim3 grid((count + block.x - 1)/block.x);
        init_int_array<<<grid, block>>>(n, 2*n, d_best_m);
        checkCuda(cudaGetLastError());
    }
    int *d_positions;
    checkCuda(cudaMalloc(&d_positions, count*sizeof(int)));
    {
        dim3 block(BLOCK_X, BLOCK_Y);
        dim3 grid((n + block.x - 1)/block.x, (m + block.y - 1)/block.y);
        find_positions<<<grid, block>>>(n, m, d_sorted_idx, d_positions, d_tr_pop_out);
        checkCuda(cudaGetLastError());
    }
    {
        dim3 block(BLOCK_1D);
        dim3 grid((n + block.x - 1)/block.x);
        best_objective<<<grid, block>>>(n, m, d_positions, d_best_m);
        checkCuda(cudaGetLastError());
    }
    checkCuda(cudaFree(d_positions));
    checkCuda(cudaFree(d_tr_pop_out));

    {
        int *d_dominating_idx, *d_last_batch, *d_final_ranks;
        checkCuda(cudaMalloc(&d_dominating_idx, n*BLOCK_1D*sizeof(int)));
        checkCuda(cudaMalloc(&d_last_batch,   n*sizeof(int)));
        checkCuda(cudaMalloc(&d_final_ranks,  n*sizeof(int)));
        {
            dim3 block(BLOCK_1D);
            dim3 grid((n + block.x - 1)/block.x);
            init_int_array<<<grid, block>>>( 0, n, d_last_batch);
            init_int_array<<<grid, block>>>(-1, n, d_final_ranks);
        }
        {
            dim3 block(BLOCK_1D);
            dim3 grid(min(n, 65535));
            init_int_array_limited<<<grid, block>>>(-1, n*BLOCK_1D, d_dominating_idx);

            int num_sorted = 0;
            int rank = 0;
            while (num_sorted < n) {
                domination_check<<<grid, block>>>(n, m, d_tr_pop, d_sorted_idx, 
                        d_best_m, d_dominating_idx, d_last_batch, 
                        d_final_ranks, rank);
                checkCuda(cudaMemcpy(h_ranks, d_final_ranks, 
                                     n*sizeof(int), cudaMemcpyDeviceToHost));
                int front_count = 0;
                for (int i = 0; i < n; i++) {
                    if (h_ranks[i] == rank) {
                        front_count += 1;
                    }

                }
                if (verbosity > 0) {
                    printf("Front %d: %d elements.\n", rank + 1, front_count);
                    if (verbosity > 1) {
                        for (int i = 0; i < n; i++) {
                            if (h_ranks[i] == rank) {
                                printf("%d ", i);
                            }

                        }
                        printf("\n");
                    }
                }
                rank++;
                num_sorted += front_count;
            }
        }
        checkCuda(cudaFree(d_dominating_idx));
        checkCuda(cudaFree(d_last_batch));
        checkCuda(cudaFree(d_final_ranks));
    }

    
    checkCuda(cudaFree(d_tr_pop));
    checkCuda(cudaFree(d_sorted_idx));
    checkCuda(cudaFree(d_best_m));
}

void show_info(int n, int m, char *filename, int verbosity) {
    printf("Parameters for this run:\n");
    printf("    Population size:      %d\n", n);
    printf("    Number of objectives: %d\n", m);
    if (filename != NULL) {
        printf("    Population data file: %s\n", filename);
    }
    printf("    Verbosity:            %d\n\n", verbosity);
}

#ifdef MATLAB_MEX_FILE
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
    cudaFree(0); // Trick to initalize CUDA context
    
    printf("GPU-BOS: CUDA implementation of the Best Order Sort algorithm\n\n");
    int verbosity = 1;
    if (nrhs < 1) {
        mexErrMsgIdAndTxt("nds:nrhs", "Required input: Population matrix.");
    } 
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("nds:nrhs", "Required output: Ranks array.");
    }
    if (!mxIsSingle(prhs[0]) || mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("nds:population", "Input population must be a single precision matrix.");
    }
    if (nrhs > 1) {
        if (!mxIsScalar(prhs[1])) {
            mexErrMsgIdAndTxt("nds:verbosity", "Verbosity must be a scalar.");
        }
        verbosity = mxGetScalar(prhs[1]);
        if (verbosity < 0 || verbosity > 2) {
            mexErrMsgIdAndTxt("nds:verbosity_level", "Verbosity must be 0, 1 or 2.");
        }
    }
    
    int n = mxGetN(prhs[0]);
    int m = mxGetM(prhs[0]);
    float *population = (float *) mxGetPr(prhs[0]);
    show_info(n, m, NULL, verbosity);

    double start_time = get_time();
    plhs[0] = mxCreateNumericMatrix(1, (mwSize)n, mxINT32_CLASS, mxREAL);
    int *ranks = (int *) mxGetPr(plhs[0]);
    nds(n, m, population, verbosity, ranks);
    double end_time = get_time();
    if (verbosity == 0) {
        printf("Elapsed time: %.9f ms.\n", end_time - start_time);
    }
}
#else
int main(int argc, char **argv) {
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaFree(0); // Trick to initalize CUDA context

    printf("GPU-BOS: CUDA implementation of the Best Order Sort algorithm\n\n");
    int verbosity = 1;

    int c, error;
    while ((c = getopt(argc, argv, "hv:")) != -1) {
        switch (c) {
            case 'h':
                send_help(argv[0]);
                exit(EXIT_SUCCESS);
            case 'v':
                error = parse_int(optarg, &verbosity);
                if (error) {
                    fprintf(stderr, "ERROR (-v): Invalid verbosity level.\n");
                    exit(EXIT_FAILURE);
                }
                break;
            default:
                send_help(argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    int n_opts = argc - optind;
    if (!n_opts) {
        send_help(argv[0]);
        exit(EXIT_FAILURE);
    }
    if (n_opts < 3) {
        fprintf(stderr, "ERROR: Missing required parameters.\n");
        exit(EXIT_FAILURE);
    }
    if (n_opts > 3) {
        fprintf(stderr, "WARNING: Too many non-optional arguments!\n");
    } 

    int n, m;
    error = parse_int(argv[optind++], &n);
    if (error) {
        fprintf(stderr, "ERROR (n): Invalid population size.\n");
        exit(EXIT_FAILURE);
    }

    error = parse_int(argv[optind++], &m);
    if (error) {
        fprintf(stderr, "ERROR (m): Invalid number of objectives.\n");
        exit(EXIT_FAILURE);
    }

    char filename[BUFSIZE];
    strcpy(filename, argv[optind++]);
    FILE *f = fopen(filename, "r");
    
    if (f == NULL) {
        fprintf(stderr, "ERROR (pop_file): Unable to open population file.\n");
        exit(EXIT_FAILURE);
    }

    // Read population data
    float *h_pop = (float *) malloc(n*m*sizeof(float));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if(!fscanf(f, "%f", &h_pop[i*m + j])) {
                fprintf(stderr, "ERROR: While reading population.");
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(f);

    show_info(n, m, filename, verbosity);

    double start_time = get_time();
    int *h_ranks = (int *) malloc(n*sizeof(int));
    nds(n, m, h_pop, verbosity, h_ranks);
    free(h_ranks);
    double end_time = get_time();
    if (verbosity == 0) {
        printf("Elapsed time: %.9f ms.\n", end_time - start_time);
    }

    free(h_pop);
	exit(EXIT_SUCCESS);
}
#endif
