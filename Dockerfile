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

# Build cavellman with BLAS path on Linux (Makefile auto-detects UNAME=Linux
# and links OpenBLAS via -lopenblas).
RUN make cavellman

# Persistent state lives on the mounted volume at /data; spore goes under it.
ENV CAVELLMAN_SPORE_DIR=/data/spore
RUN mkdir -p /data/spore

# Unbuffered stdout — belt and suspenders. cavellman.c calls setvbuf(_IONBF)
# in main(), but Docker stdio bridge can still buffer on first deploy; stdbuf
# -o0 forces unbuffered from outside the binary. exec replaces the sh process
# so cavellman becomes PID 1 inside the container (clean signal handling).
CMD ["sh", "-c", "exec stdbuf -o0 ./cavellman --preset medium --weights weights/cavellman_medium.bin --seed 4242 < /dev/null"]
