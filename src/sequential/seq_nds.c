#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

#include "utils.h"

#define BUFSIZE 100

int n, m;
float *population;
int *sorted_pop, *last_front, *ranks;
int **front_counts;
int ***fronts;

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

void add_to_front(int idx, int f, int j) {
    int count = front_counts[j][f];
    if (!count) {
        // Initialize front
        fronts[j][f] = malloc(n * sizeof(int));
    }
    fronts[j][f][count] = idx;
    front_counts[j][f]++;
}

void rank_population() {
    int n_ranked = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            int idx = sorted_pop[j*n + i];
            int rank = ranks[idx];

            if (rank < 0) {
                // Individual not ranked!
                bool check = true;
                for (int x = 0; x <= last_front[j]; x++) {
                    check = false;
                    for (int y = 0; y < front_counts[j][x]; y++) {
                        int member = fronts[j][x][y];
                        check = is_dominated(idx, member);
                        if (check) {
                            // Not on this front!
                            break;
                        }
                    }
                    if (!check) {
                        // On this front!
                        add_to_front(idx, x, j);
                        ranks[idx] = x;
                        n_ranked++;
                        break;
                    }
                }
                if (check) {
                    // Dominated by the last front, on a new front!
                    last_front[j]++;
                    add_to_front(idx, last_front[j], j);
                    ranks[idx] = last_front[j];
                    n_ranked++;
                }
            } else {
                // Individual previously ranked, add it to our fronts!
                add_to_front(idx, rank, j);
                if (rank > last_front[j]) {
                    last_front[j] = rank;
                }
            }
            if (n_ranked == n) {
                return;
            }
        }
    }
}


void nds(int verbosity) {

    sorted_pop = malloc(n*m*sizeof(int));
    fronts = malloc(m*sizeof(int **));
    front_counts = malloc(m*sizeof(int *));
    ranks = malloc(n*sizeof(int));
    last_front = calloc(m, sizeof(int));

    for (int i = 0; i < n; i++) {
        ranks[i] = -1;
    }

    for (int j = 0; j < m; j++) {
        // Initalize population indexes.
        for (int i = 0; i < n; i++) {
           sorted_pop[j*n + i] = i;
        }
        // Sort population by objective j.
        qsort_r(&sorted_pop[j*n], n, sizeof(int), compare, &j);
        fronts[j] = malloc(n * sizeof(int *));
        front_counts[j] = calloc(n, sizeof(int));
    }

    rank_population();

    int n_ranked = 0;
    int rank = 0;
    while (n_ranked < n) {
        int front_count = 0;
        for (int i = 0; i < n; i++) {
            if (ranks[i] == rank) {
                front_count++;
            }
        }
        if (verbosity > 0) {
            printf("Front %d: %d elements.\n", rank + 1, front_count);
            if (verbosity > 1) {
                for (int i = 0; i < n; i++) {
                    if (ranks[i] == rank) {
                        printf("%d ", i);
                    }

                }
                printf("\n");
            }
        }
        rank++;
        n_ranked += front_count;
    }

    for (int j = 0; j < m; j++) {
        for (int i = 0; i <= last_front[j]; i++) {
            free(fronts[j][i]);
        }
        free(fronts[j]);
        free(front_counts[j]);
    }
   
    free(last_front);
    free(sorted_pop);
    free(fronts);
    free(front_counts);
    free(ranks);
}

int main(int argc, char **argv) {
    int verbosity = 1;

    printf("BOS: Sequential implementation of the Best Order Sort algorithm\n\n");

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
