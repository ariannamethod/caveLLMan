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

# Direct exec of cavellman — Docker default stdio is enough now that the
# tick loop fflush(stdout)es every 100 ms. No sh wrapper, no stdbuf, no
# stdin redirect: cavellman's fcntl(O_NONBLOCK) on a closed/disconnected
# stdin returns EOF immediately, ring proceeds as autonomous.
CMD ["./cavellman", "--preset", "medium", "--weights", "weights/cavellman_medium.bin", "--seed", "4242"]
