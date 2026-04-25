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

# Unbuffered stdout so Railway log stream sees each line as it's printed.
ENV PYTHONUNBUFFERED=1

# /dev/null on stdin keeps the non-blocking try_read_user_line happy
# (returns 0 immediately, ring proceeds as autonomous).
CMD ["sh", "-c", "./cavellman --preset medium --weights weights/cavellman_medium.bin --seed 4242 < /dev/null"]
