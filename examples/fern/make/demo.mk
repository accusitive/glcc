CC ?= cc
OUT = build/demo
SRC = demo.c

all: $(OUT)

$(OUT): $(SRC)
	mkdir -p build
	$(CC) $(SRC) -o $@

run: $(OUT)
	./$(OUT)

clean:
	rm -f $(OUT)

.PHONY: all run clean
