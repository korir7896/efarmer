FROM python:3.12-slim-bookworm

# Standard command-line tooling a population geneticist would have on hand.
# Withholding it would be an artificial handicap, not a source of difficulty.
RUN apt-get update && apt-get install -y --no-install-recommends \
        bcftools \
        tabix \
        build-essential \
        curl \
        ca-certificates \
        less \
        procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        numpy==2.5.3 \
        scipy==1.16.3 \
        msprime==1.4.4 \
        tskit==1.0.3 \
        scikit-allel==1.3.13 \
        demes==0.2.3 \
        pandas==2.3.3

# Keep thread counts at or below the CPUs declared in task.toml.
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY data/ /app/data/

