# Prism shader examples: compile each C shader to the shader IR's
# canonical ISA text with `glcc -fprism --emit-isa`. The ISA text is
# the artifact (there is no executable to run); `make show` prints it.
# Prefers the repo's build tree; falls back to an installed glcc.
GLCC ?= $(firstword $(wildcard ../../target/debug/glcc) glcc)

SHADERS := vs frag rings

all: $(SHADERS:%=build/%.isa)

build/%.isa: %.c
	mkdir -p $(dir $@)
	$(GLCC) -fprism -O1 --emit-isa $< -o $@

show: all
	@for f in $(SHADERS:%=build/%.isa); do echo "== $$f =="; cat "$$f"; echo; done

clean:
	rm -rf build

.PHONY: all show clean
