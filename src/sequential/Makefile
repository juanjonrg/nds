CC = gcc
CFLAGS = -Wall -O3
INCLUDES = -I../utils/
LIBS = ../utils/utils.o

MAIN = seq_nds
OBJS = $(patsubst %.c,%.o,$(wildcard *.c))

all: $(MAIN)

$(MAIN): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(LIBS) -o $(MAIN)

%.o : %.c
	$(CC) $(CFLAGS) -c $< $(INCLUDES)

install:
	cp -v $(MAIN) ../../test/bin

clean:
	rm -v $(OBJS) $(MAIN)


