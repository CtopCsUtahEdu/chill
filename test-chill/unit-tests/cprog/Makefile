OBJS = $(patsubst %.cc, %.o, $(wildcard *.cc))

.PHONY: all
all: sorter

$(OBJS): %.o: %.cc
	g++ -g -fprofile-arcs -ftest-coverage -c $< -o $@

.PHONY: sorter
sorter: $(OBJS)
	g++ -g -fprofile-arcs -ftest-coverage -o bin/sorter $(OBJS)

.PHONY: clean
clean:
	rm -f *.o
	rm -f *.gcno *.gcda *.gcov
	rm -f bin/sorter
