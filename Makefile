# emoLM — Makefile
# Train and run emoji language models using notorch (pure C, no Python)
#
# Usage:
#   make              Build with BLAS (fastest)
#   make cpu          Build without BLAS (portable)
#   make train        Build training binary with BLAS
#   make infer        Build inference binary
#   make test         Build and run all tests (C + Python)
#   make test_c       C smoke test (50 steps, check loss < 5.5)
#   make test_py      Python unit tests (131 tests)
#   make clean        Remove build artifacts

CC = cc
CFLAGS = -O2 -Wall -Wextra -std=c11 -I.

# Detect platform
UNAME := $(shell uname)

# ── macOS: Apple Accelerate (AMX/Neural Engine) ──
ifeq ($(UNAME), Darwin)
  BLAS_FLAGS = -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK -framework Accelerate
  BLAS_NAME = Accelerate
endif

# ── Linux: OpenBLAS ──
ifeq ($(UNAME), Linux)
  BLAS_FLAGS = -DUSE_BLAS -lopenblas
  BLAS_NAME = OpenBLAS
endif

.PHONY: all train cpu infer test test_c test_py clean help

# Default: build both binaries with BLAS
all: train_emolm infer_emolm
	@echo "Built train_emolm + infer_emolm with $(BLAS_NAME)"

# Training binary with BLAS
train: train_emolm
	@echo "Ready: ./train_emolm"

train_emolm: train_emolm.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o train_emolm train_emolm.c notorch.c -lm
	@echo "Compiled: train_emolm (CPU + $(BLAS_NAME))"

# Diffusion training binary
diffusion: train_diffusion
	@echo "Ready: ./train_diffusion"

train_diffusion: train_diffusion.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o train_diffusion train_diffusion.c notorch.c -lm
	@echo "Compiled: train_diffusion (CPU + $(BLAS_NAME))"

# Inference binary with BLAS
infer: infer_emolm
	@echo "Ready: ./infer_emolm"

infer_emolm: infer_emolm.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o infer_emolm infer_emolm.c notorch.c -lm
	@echo "Compiled: infer_emolm (CPU + $(BLAS_NAME))"

# CPU without BLAS (portable fallback) — builds both binaries
cpu: train_emolm.c infer_emolm.c notorch.c notorch.h
	$(CC) $(CFLAGS) -o train_emolm train_emolm.c notorch.c -lm
	$(CC) $(CFLAGS) -o infer_emolm infer_emolm.c notorch.c -lm
	@echo "Compiled: train_emolm + infer_emolm (CPU, no BLAS)"

# ── Tests ────────────────────────────────────────────────────────────────
# All tests
test: test_c test_py

# C smoke test: 50 steps, verify training completes
test_c: cpu
	@echo "── C smoke test (tiny, 50 steps) ──"
	@./train_emolm --preset tiny --steps 50 --no-save 2>&1 | tee /tmp/emolm_test.log
	@grep -q "Training complete" /tmp/emolm_test.log && echo "✅ C test passed" || (echo "❌ C test failed" && exit 1)

# Python unit tests
test_py:
	@echo "── Python unit tests ──"
	python3 -m unittest tests.test_emolm -v

clean:
	rm -f train_emolm infer_emolm *.o

help:
	@echo "emoLM — emoji language model (pure C, notorch)"
	@echo ""
	@echo "  make            Build all with BLAS"
	@echo "  make cpu        Build all without BLAS (portable)"
	@echo "  make train      Build training binary"
	@echo "  make infer      Build inference binary"
	@echo "  make test       Run all tests (C + Python)"
	@echo "  make test_c     C smoke test (50 steps)"
	@echo "  make test_py    Python unit tests"
	@echo "  make clean      Remove artifacts"
	@echo ""
	@echo "Train:"
	@echo "  ./train_emolm --preset standard --steps 4000"
	@echo "  ./train_emolm --preset tiny --steps 2000 --save weights/emolm.bin"
	@echo ""
	@echo "Infer:"
	@echo "  ./infer_emolm --weights weights/emolm.bin --preset tiny"
	@echo ""
	@echo "Presets: tiny(18/3/2) micro(48/4/3) standard(64/8/3) small(96/8/4) medium(128/8/4)"
