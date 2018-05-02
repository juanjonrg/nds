CC = gcc
CFLAGS = -Wall -O3
INCLUDES = -I../utils/
LIBS = -lpthread ../utils/utils.o

MAIN = multi_nds
OBJS := $(patsubst %.c,%.o,$(wildcard *.c))

all: $(MAIN)

$(MAIN): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(MAIN) $(LIBS)

%.o : %.c
	$(CC) $(CFLAGS) -c $< $(INCLUDES)

install:
	cp -v $(MAIN) ../../test/bin

clean:
	rm -v $(OBJS) $(MAIN)


