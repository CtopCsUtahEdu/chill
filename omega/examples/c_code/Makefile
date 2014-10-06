CC = g++
INC_DIR = -I../../include
LIB_DIR = -L../../lib

all: example library_example


example : example.o ../../lib/libomega.a
	${CC} -g $(LIB_DIR) example.o -lomega -o example

example.o : example.c
	${CC} -Wall -g $(INC_DIR) -c example.c

library_example : library_example.o ../../lib/libomega.a
	${CC} -g $(LIB_DIR) library_example.o -lomega -o library_example

library_example.o : library_example.c
	${CC} -Wall -g $(INC_DIR) -c library_example.c

clean:
	@rm -fr *.o example library_example
