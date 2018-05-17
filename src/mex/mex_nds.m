% Compile MEX.
mex -v COPTIMFLAGS="-O3 -std=c99" ../sequential/seq_nds.c
mex -v COPTIMFLAGS="-O3 -std=c99" ../multicore/multi_nds.c
% The CUDA version requires the Parallel Computing Toolbox, CUDA SDK and CUB 1.7.4.
%mexcuda -v COPTIMFLAGS="-O3" -I"/opt/cub-1.7.4/" ../cuda/cuda_nds.cu

% Read population from file.
population = dlmread('../../test/pops/rand_100_4_20180502_201305.109.txt');
% Transpose population and convert it to single precision.
population = single(transpose(population));

% Run the NDS implementations.
% The second parameter is optional and controls the verbosity:
% 0 (only nds runtime), 1 (default, only front counts), 2 (front counts and contents).
ranks_seq = seq_nds(population, 2);
ranks_multi = multi_nds(population, 2);
%ranks_cuda = cuda_nds(population, 2);

% Check if they returned the same result
result = isequal(ranks_seq, ranks_multi)
