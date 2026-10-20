GLCC ?= $(firstword $(wildcard ../../target/debug/glcc ../../target/debug/glcc.exe) glcc)
OUT = build/demo
SRC = demo.c

all: $(OUT)

$(OUT): $(SRC)
	mkdir -p build
	$(GLCC) $(SRC) -o $@

run: $(OUT)
	./$(OUT)

clean:
	rm -f $(OUT)

.PHONY: all run clean
