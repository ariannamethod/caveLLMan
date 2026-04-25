# caveLLMan ring — continuous autonomous evolution.
# Built for Railway long-running deploy; works on any Linux with libopenblas.
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Build cavellman WITHOUT BLAS (naive matvec fallback). Trinity SIGSEGV'd
# inside cblas_sgemv (caught by crash trap) — BLAS interaction with our
# 4-5 pthread setup is the suspect. Naive matvec is single-loop inline,
# eliminates all BLAS state. Performance hit is acceptable for the medium
# preset (E=128, L=4); ring/v1/v2 keep BLAS in their separate services.
RUN make cavellman-cpu

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
