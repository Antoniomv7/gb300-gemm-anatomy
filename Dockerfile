# gb300-gemm-anatomy Phase 0 environment.
# Digest-pinned CUDA 13.1.0 devel image (includes nvcc, ptxas, cuobjdump,
# nvdisasm, and Nsight Compute) plus CuTe DSL pinned at CUTLASS v4.6.1.
# Build args default to the values in VERSIONS.env; `make build-image` passes
# them explicitly from VERSIONS.env, and `make check-static` verifies the
# defaults stay consistent. No GPU is used or required at build time.

ARG BASE_IMAGE=nvidia/cuda:13.1.0-devel-ubuntu24.04@sha256:0725ed044e80c230fcd54218ae3edc2855897ef7813b20868bdb53b03b99ea1c

FROM ${BASE_IMAGE}

ARG CUDA_VERSION=13.1.0
ARG CUTLASS_VERSION=v4.6.1
ARG CUTLASS_COMMIT=e05f953a5b3d38adc240df2ff928e0421c2abba3
ARG MAX_BUILD_JOBS=2

LABEL org.opencontainers.image.title="gb300-gemm-anatomy-phase0" \
      org.opencontainers.image.description="Reproducible CUDA 13.1 + CuTe DSL v4.6.1 environment for BF16 GEMM anatomy on GB300" \
      org.opencontainers.image.licenses="BSD-3-Clause" \
      anatomy.cuda.version="${CUDA_VERSION}" \
      anatomy.cutlass.version="${CUTLASS_VERSION}" \
      anatomy.cutlass.commit="${CUTLASS_COMMIT}" \
      anatomy.cuda.arch="sm_103a" \
      anatomy.max.build.jobs="${MAX_BUILD_JOBS}"

# Cap every build system that honours these variables at two jobs.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MAX_JOBS=2 \
    MAKEFLAGS=-j2

# Minimal deterministic packages from the Ubuntu 24.04 archive.
# Python 3.12 (Ubuntu 24.04 default) is inside CuTe DSL's supported 3.10-3.14.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        python3 \
        python3-pip \
        python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Dedicated virtual environment (Ubuntu 24.04 system Python is
# PEP 668 externally managed, so nothing is installed system-wide).
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH}

# Fetch CUTLASS at exactly the pinned commit (the peeled target of tag
# v4.6.1) and fail the build if the checkout does not match.
RUN git init -q /opt/cutlass \
    && git -C /opt/cutlass remote add origin https://github.com/NVIDIA/cutlass.git \
    && git -C /opt/cutlass fetch --depth 1 origin "${CUTLASS_COMMIT}" \
    && git -C /opt/cutlass checkout -q FETCH_HEAD \
    && test "$(git -C /opt/cutlass rev-parse HEAD)" = "${CUTLASS_COMMIT}"

# Install CuTe DSL with the pinned commit's own installer, which resolves to
# the version-pinned wheel nvidia-cutlass-dsl[cu13]==4.6.1. The import check
# is host-side Python only; no GPU is touched during build.
RUN bash /opt/cutlass/python/CuTeDSL/setup.sh --cu13 \
    && python3 -c "import cutlass; v = cutlass.__version__; assert v == '4.6.1', f'unexpected CuTe DSL version {v}'; print('CuTeDSL', v)" \
    && pip cache purge 2>/dev/null || true

WORKDIR /workspace
