CC = nvcc
FLAGS = -O3 -arch=sm_37
#FLAGS = -O3 -arch=sm_20
#FLAGS += -g -G -lineinfo
INCLUDES = -I../utils -I/opt/cub-1.7.4
LIBS = ../utils/utils.o

MAIN = cuda_nds

all: $(MAIN)

$(MAIN): $(MAIN).o
	$(CC) $(FLAGS) $(LIBS) $^ -o $@

%.o : %.cu
	$(CC) $(FLAGS) -dc $< $(INCLUDES)

install:
	cp -v $(MAIN) ../../test/bin

clean:
	rm -vf *.o $(MAIN)
