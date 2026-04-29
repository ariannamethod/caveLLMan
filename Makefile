# caveLLMan — Makefile
# Self-evolving hieroglyphic language model (pure C + notorch).
# Dual-only: two caves in a ring. Human is optional, not central.
#
# Targets:
#   make                       Build cavellman ring engine (default)
#   make cavellman             Build ring engine with BLAS + pthreads
#   make cavellman-cpu         Build ring engine without BLAS (portable)
#   make train_cavellman       Build training binary
#   make train_diffusion       Build diffusion training binary
#   make train                 Build all training binaries
#   make weights               Train fresh weights/cavellman_v3.bin if missing
#   make clean                 Remove build artifacts
#   make help                  Show this help

CC = cc
CFLAGS = -O3 -Wall -Wextra -std=c11 -I.

# Detect platform for BLAS
UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
  BLAS_FLAGS = -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK -framework Accelerate
  BLAS_NAME = Accelerate
endif

ifeq ($(UNAME), Linux)
  BLAS_FLAGS = -DUSE_BLAS -lopenblas
  BLAS_NAME = OpenBLAS
  # -rdynamic + -g make backtrace_symbols print function names from the crash
  # trap installed in main(). -O3 + -march=native -mtune=native are the
  # second half of the Railway CPU recipe (first half — single-thread
  # OpenBLAS — already in Dockerfile env). On Railway Linux runners
  # with small preset dim (96/128) the inner-loop matmul outruns
  # OpenBLAS once the compiler vectorises it; recipe gave ~7× end-to-end
  # speedup on Henry session 2026-04-29 (310 → 142 min on 12K steps).
  # _GNU_SOURCE: sigaction, SA_RESETHAND, strdup, usleep, fileno on glibc.
  CFLAGS += -g -rdynamic -D_GNU_SOURCE -march=native -mtune=native
endif

.PHONY: all cavellman cavellman-cpu train train_cavellman train_diffusion weights clean help

all: cavellman

# ── Ring engine (dual-only) ─────────────────────────────────────────────

cavellman: cavellman.c ariannamethod/notorch.c ariannamethod/notorch.h
	$(CC) $(CFLAGS) -Iariannamethod -o cavellman cavellman.c ariannamethod/notorch.c -lm -lpthread $(BLAS_FLAGS)
	@echo "Compiled: cavellman (dual ring + Hebbian + async learner + $(BLAS_NAME))"

# 42/ — gravitational-dispatch fork (real sibling, not CLI flag).
# Modular split: cavellman_42.c (engine + ring + trinity) + predator.c
# (storm engine), shared types + extern API in 42/cavellman_42.h.
cavellman_42: 42/cavellman_42.c 42/predator.c 42/trinity.c 42/cavellman_42.h ariannamethod/notorch.c ariannamethod/notorch.h
	$(CC) $(CFLAGS) -Iariannamethod -I42 -o cavellman_42 42/cavellman_42.c 42/predator.c 42/trinity.c ariannamethod/notorch.c -lm -lpthread $(BLAS_FLAGS)
	@echo "Compiled: cavellman_42 (gravity-dispatch fork, modular: core + predator + trinity + $(BLAS_NAME))"

cavellman-cpu: cavellman.c ariannamethod/notorch.c ariannamethod/notorch.h
	$(CC) $(CFLAGS) -Iariannamethod -o cavellman cavellman.c ariannamethod/notorch.c -lm -lpthread
	@echo "Compiled: cavellman (dual ring + Hebbian + async learner, no BLAS)"

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

# ── Cleanup ─────────────────────────────────────────────────────────────

clean:
	rm -f cavellman train_cavellman train_diffusion *.o

help:
	@echo "caveLLMan — self-evolving hieroglyphic LM (pure C, notorch)"
	@echo "Dual-only: two caves in a ring. Human is optional, not central."
	@echo ""
	@echo "  make                  Build cavellman ring engine (default)"
	@echo "  make cavellman        Build ring engine with BLAS + pthreads"
	@echo "  make cavellman-cpu    Build ring engine without BLAS"
	@echo "  make train            Build training binaries"
	@echo "  make weights          Train weights/cavellman_v3.bin (if missing)"
	@echo "  make clean            Remove build artifacts"
	@echo ""
	@echo "Train:"
	@echo "  ./train_cavellman --dataset data/cavellman_train_final.txt --preset small --steps 15000"
	@echo "  ./train_diffusion  --dataset data/cavellman_train_final.txt --steps 15000"
	@echo ""
	@echo "Run:"
	@echo "  ./cavellman                                      # A=Dracula, B=Frankenstein"
	@echo "  ./cavellman --preset medium --weights weights/cavellman_medium.bin"
	@echo ""
	@echo "Presets: tiny(18/3/2) micro(48/4/3) standard(64/8/3) small(96/8/4) medium(128/8/4)"
