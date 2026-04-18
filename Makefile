# caveLLMan — Makefile
# Self-evolving hieroglyphic language model (pure C + notorch).
#
# Targets:
#   make                       Build cavellman engine (default)
#   make cavellman             Build interactive engine with BLAS + pthreads
#   make cavellman-cpu         Build engine without BLAS (portable)
#   make train_cavellman       Build training binary
#   make train_diffusion       Build diffusion training binary
#   make train                 Build all training binaries
#   make weights               Train fresh weights/cavellman_v3.bin if missing
#   make test                  Run semantic tokenizer tests (node)
#   make clean                 Remove build artifacts
#   make help                  Show this help

CC = cc
CFLAGS = -O2 -Wall -Wextra -std=c11 -I.

# Detect platform for BLAS
UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
  BLAS_FLAGS = -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK -framework Accelerate
  BLAS_NAME = Accelerate
endif

ifeq ($(UNAME), Linux)
  BLAS_FLAGS = -DUSE_BLAS -lopenblas
  BLAS_NAME = OpenBLAS
endif

.PHONY: all cavellman cavellman-cpu train train_cavellman train_diffusion weights test clean help

all: cavellman

# ── Interactive engine ──────────────────────────────────────────────────

cavellman: cavellman.c ariannamethod/notorch.c ariannamethod/notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -Iariannamethod -o cavellman cavellman.c ariannamethod/notorch.c -lm -lpthread
	@echo "Compiled: cavellman (Hebbian + async learner + $(BLAS_NAME))"

cavellman-cpu: cavellman.c ariannamethod/notorch.c ariannamethod/notorch.h
	$(CC) $(CFLAGS) -Iariannamethod -o cavellman cavellman.c ariannamethod/notorch.c -lm -lpthread
	@echo "Compiled: cavellman (Hebbian + async learner, no BLAS)"

# ── Training binaries ───────────────────────────────────────────────────

train: train_cavellman train_diffusion

train_cavellman: ariannamethod/train_cavellman.c ariannamethod/notorch.c ariannamethod/notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -Iariannamethod -o train_cavellman ariannamethod/train_cavellman.c ariannamethod/notorch.c -lm
	@echo "Compiled: train_cavellman ($(BLAS_NAME))"

train_diffusion: ariannamethod/train_diffusion.c ariannamethod/notorch.c ariannamethod/notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -Iariannamethod -o train_diffusion ariannamethod/train_diffusion.c ariannamethod/notorch.c -lm
	@echo "Compiled: train_diffusion ($(BLAS_NAME))"

# ── Weights: train fresh if missing ─────────────────────────────────────
# The .bin file is gitignored. Run `make weights` to generate.

weights: weights/cavellman_v3.bin

weights/cavellman_v3.bin: train_cavellman data/cavellman_train_final.txt
	@mkdir -p weights
	./train_cavellman --dataset data/cavellman_train_final.txt --preset small --save weights/cavellman_v3.bin
	@echo "Trained: weights/cavellman_v3.bin — run ./cavellman --weights weights/cavellman_v3.bin --preset small"

# ── Tests ───────────────────────────────────────────────────────────────

test:
	@echo "── semantic tokenizer tests ──"
	@node tests/test_semantic_tokenizer.js

# ── Cleanup ─────────────────────────────────────────────────────────────

clean:
	rm -f cavellman train_cavellman train_diffusion *.o

help:
	@echo "caveLLMan — self-evolving hieroglyphic LM (pure C, notorch)"
	@echo ""
	@echo "  make                  Build cavellman engine (default)"
	@echo "  make cavellman        Build engine with BLAS + pthreads"
	@echo "  make cavellman-cpu    Build engine without BLAS"
	@echo "  make train            Build training binaries"
	@echo "  make weights          Train weights/cavellman_v3.bin (if missing)"
	@echo "  make test             Run semantic tokenizer tests (node)"
	@echo "  make clean            Remove build artifacts"
	@echo ""
	@echo "Train:"
	@echo "  ./train_cavellman --dataset data/cavellman_train_final.txt --preset small --steps 15000"
	@echo "  ./train_diffusion  --dataset data/cavellman_train_final.txt --steps 15000"
	@echo ""
	@echo "Run:"
	@echo "  ./cavellman --weights weights/cavellman_v3.bin --preset small"
	@echo ""
	@echo "Presets: tiny(18/3/2) micro(48/4/3) standard(64/8/3) small(96/8/4) medium(128/8/4)"
