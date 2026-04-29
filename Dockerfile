# caveLLMan ring — continuous autonomous evolution.
# Built for Railway long-running deploy; works on any Linux with libopenblas.
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# BLAS path back on now that the actual root cause (preset dim mismatch
# between A/B trained at E=96 small and startCommand --preset medium
# making forwards read past wq/w_fc1 buffers) is fixed via shape detection
# in model_load. With correct dims, BLAS is happy too.
#
# Build both ring engine AND train_cavellman — the ring `fork+execlp`s
# `./train_cavellman` for microtraining on own speech + DNA samples
# (cavellman.c:2211, 42/cavellman_42.c:1693). Without this binary the
# child always exits 127 ("command not found") and Hebbian becomes the
# only adaptation path. Storm fork has the most cave threads → most
# observed failures (17/200 lines on v4-storm vs 0-4 on others).
RUN make cavellman train_cavellman

# Persistent state lives on the mounted volume at /data; spore goes under it.
ENV CAVELLMAN_SPORE_DIR=/data/spore
RUN mkdir -p /data/spore

# Force OpenBLAS into single-threaded mode. With multiple application pthreads
# (cave threads + Molly thread + orchestrator + learner) calling sgemv
# concurrently, OpenBLAS's internal thread pool race-condited and SIGSEGV'd
# inside sgemv_t_PRESCOTT after ~30-80 seconds on Railway Linux. Caught with
# the crash trap. Single-threaded BLAS makes each sgemv inline + serial,
# eliminating the contention entirely. App-level parallelism still works.
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV GOTO_NUM_THREADS=1

# Direct exec of cavellman — Docker default stdio is enough now that the
# tick loop fflush(stdout)es every 100 ms. No sh wrapper, no stdbuf, no
# stdin redirect: cavellman's fcntl(O_NONBLOCK) on a closed/disconnected
# stdin returns EOF immediately, ring proceeds as autonomous.
CMD ["./cavellman", "--preset", "medium", "--weights", "weights/cavellman_medium.bin", "--seed", "4242"]
