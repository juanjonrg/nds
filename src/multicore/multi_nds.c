#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

#include "utils.h"

#define BUFSIZE 100
#define checkPth(rc) __checkPth(rc, __FILE__, __LINE__)
static inline void __checkPth(int rc, const char *file, const int line) {
    if (rc != 0) {
        fprintf(stderr, "PTH error at %s:%i: %d\n", file, line, rc);
        exit(1);
    }
}

int n, m;
float *population;
int *sorted_pop;
int ***all_fronts;
int **all_front_counts;
int *ranks;

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

int comp(const int a_idx, const int b_idx, const int j) {
    float a = population[a_idx*m + j];
    float b = population[b_idx*m + j];
    if (a < b)
    	return -1;
    else if (a > b)
    	return 1;
    return 0;
}

int compare(const void *a_in, const void *b_in, void *thunk_in) {
    int j = *((int*)thunk_in);
    int a_idx = *(int *) a_in;
    int b_idx = *(int *) b_in;
    int result = comp(a_idx, b_idx, j);

    if (result == 0) {
        for (int i = 1; i < m; i++) {
            result = comp(a_idx, b_idx, (j + i) % m);
            if (result != 0) {
                return result;
            }
        }
    }
    return result;
}

double get_time() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
        return ts.tv_sec * 1000 + ts.tv_nsec / 1000000.0;
    else
        return 0;
}

int is_dominated(int a, int b) {
    bool equal = true;
    for (int j = 0; j < m; j++) {
        if (population[a*m + j] < population[b*m + j]) {
            return 0;
        } else if (equal && population[a*m + j] > population[b*m + j]) {
            equal = false;
        }
    }
    return !equal;
}

int add_to_front(int idx, int f, int last_front, int **fronts, int *front_counts) {
    int count = front_counts[f];
    if (front_counts[f] == 0) {
        // Initialize front
        fronts[f] = malloc(n * sizeof(int));
    }
    fronts[f][count] = idx;
    ranks[idx] = f;
    front_counts[f]++;

    if (f > last_front) {
        return f;
    }
    return last_front;
}

void *threaded_nds(void *thread_id) {
    int j = (long) thread_id;

    // Initalize population indexes.
    for (int i = 0; i < n; i++) {
       sorted_pop[j*n + i] = i;
    }

    // Sort population by objective j.
    qsort_r(&sorted_pop[j*n], n, sizeof(int), compare, &j);

    // Initialize front structures.
    all_fronts[j] = malloc(n * sizeof(int *));
    all_front_counts[j] = calloc(n, sizeof(int));
    int **fronts = all_fronts[j];
    int *front_counts = all_front_counts[j];
    int last_front = 0;

    // Lets rank!
    for (int i = 0; i < n; i++) {
        int idx = sorted_pop[j*n + i];
        int rank = ranks[idx];

        if (rank < 0) {
            // Individual not ranked!
            bool check = true;
            for (int x = 0; x <= last_front; x++) {
                check = false;
                for (int y = 0; y < front_counts[x]; y++) {
                    int member = fronts[x][y];
                    check = is_dominated(idx, member);
                    if (check) {
                        // Not on this front!
                        break;
                    }
                }
                if (!check) {
                    // On this front!
                    last_front = add_to_front(idx, x, last_front, fronts, front_counts);
                    break;
                }
            }
            if (check) {
                // Dominated by the last front, on a new front!
                last_front = add_to_front(idx, last_front + 1, last_front, fronts, front_counts);
            }
        } else {
            // Individual ranked by other thread, add it to our fronts!
            last_front = add_to_front(idx, rank, last_front, fronts, front_counts);
        }
    }

    for (int i = 0; i <= last_front; i++) {
        free(fronts[i]);
    }
    free(fronts);

    return NULL;
}

void nds(int verbosity) {

    sorted_pop = malloc(n*m*sizeof(int));
    all_fronts = malloc(m*sizeof(int **));
    all_front_counts = malloc(m*sizeof(int *));
    ranks = malloc(n*sizeof(int));

    for (int i = 0; i < n; i++) {
        ranks[i] = -1;
    }

    int num_threads = m;
    pthread_t threads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        checkPth(pthread_create(&threads[i], NULL, threaded_nds, (void *) (intptr_t) i));
    }
    for (int i = 0; i < num_threads; i++) {
        checkPth(pthread_join(threads[i], NULL));
    }

    for (int i = 0; i < n; i++) {
        int count = all_front_counts[0][i];

        if (!count) {
            break;
        }
        if (verbosity > 0) {
            printf("Front %d: %d elements.\n", i+1, count);
            if (verbosity > 1) {
                for (int j = 0; j < n; j++) {
                    if (ranks[j] == i) {
                        printf("%d ", j);
                    }

                }
                printf("\n");
            }
        }
    }

    for (int i = 0; i < num_threads; i++) {
        free(all_front_counts[i]);
    }
    free(sorted_pop);
    free(all_fronts);
    free(all_front_counts);
    free(ranks);
}

int main(int argc, char **argv) {
    int verbosity = 1;

    printf("MC-BOS: Multicore implementation of the Best Order Sort algorithm\n\n");

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
    population = malloc(n*m*sizeof(float));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if(!fscanf(f, "%f", &population[i*m + j])) {
                fprintf(stderr, "ERROR: While reading population.");
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(f);

    printf("Parameters for this run:\n");
    printf("    Population size:      %d\n", n);
    printf("    Number of objectives: %d\n", m);
    printf("    Population data file: %s\n", filename);
    printf("    Verbosity:            %d\n\n", verbosity);

    double start_time = get_time();
    nds(verbosity);
    double end_time = get_time();
    if (verbosity == 0) {
        printf("Elapsed time: %.9f ms.\n", end_time - start_time);
    }

    free(population);
	exit(EXIT_SUCCESS);
}
