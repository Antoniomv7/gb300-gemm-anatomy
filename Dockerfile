# gb300-gemm-anatomy Phase 0 environment.
# Digest-pinned CUDA 13.1.0 devel image (includes nvcc, ptxas, cuobjdump,
# nvdisasm, and Nsight Compute) plus CuTe DSL pinned at CUTLASS v4.6.1.
# Build args default to the values in VERSIONS.env (global contract) and
# PHASE3_VERSIONS.env (Phase 3-only extension); `make build-image` passes them
# explicitly from those two files, and `make check-static` verifies the
# defaults stay consistent. No GPU is used or required at build time.

ARG BASE_IMAGE=nvidia/cuda:13.1.0-devel-ubuntu24.04@sha256:0725ed044e80c230fcd54218ae3edc2855897ef7813b20868bdb53b03b99ea1c

FROM ${BASE_IMAGE}

ARG CUDA_VERSION=13.1.0
ARG CUTLASS_VERSION=v4.6.1
ARG CUTLASS_COMMIT=e05f953a5b3d38adc240df2ff928e0421c2abba3
ARG MAX_BUILD_JOBS=2
# P3.1 auxiliary runtime dependencies (see PHASE3_VERSIONS.env and
# src/gemm/P3_1_PROTOCOL.md): NVIDIA's own CuTe DSL examples use PyTorch for
# allocation, DLPack interoperability, CUDA stream access, and CPU reference
# validation. These wheels do not replace the pinned CUDA 13.1 toolkit or the
# CuTe DSL v4.6.1 pin above, and no existing pin changes because of them.
# cuda-python and cuda-bindings are pinned coherently to the release torch
# 2.10.0+cu130 requires, so the environment's dependency graph is consistent.
ARG PYTORCH_VERSION=2.10.0+cu130
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG PYTORCH_CUDA_VERSION=13.0
ARG CUDA_PYTHON_VERSION=13.0.3
ARG CUDA_BINDINGS_VERSION=13.0.3

LABEL org.opencontainers.image.title="gb300-gemm-anatomy-phase0" \
      org.opencontainers.image.description="Reproducible CUDA 13.1 + CuTe DSL v4.6.1 environment for BF16 GEMM anatomy on GB300" \
      org.opencontainers.image.licenses="BSD-3-Clause" \
      anatomy.cuda.version="${CUDA_VERSION}" \
      anatomy.cutlass.version="${CUTLASS_VERSION}" \
      anatomy.cutlass.commit="${CUTLASS_COMMIT}" \
      anatomy.cuda.arch="sm_103a" \
      anatomy.max.build.jobs="${MAX_BUILD_JOBS}" \
      anatomy.pytorch.version="${PYTORCH_VERSION}"

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
# is host-side Python only; no GPU is touched during build. PIP_NO_CACHE_DIR=1
# already prevents cache writes, so no cache cleanup step is needed.
RUN bash /opt/cutlass/python/CuTeDSL/setup.sh --cu13 \
    && python3 -c "import cutlass; v = cutlass.__version__; assert v == '4.6.1', f'unexpected CuTe DSL version {v}'; print('CuTeDSL', v)"

# P3.1: the exact, non-floating auxiliary wheels, into the same virtual
# environment (never a second environment, never a nightly, never an
# unversioned install, never a dependency-skipping install). PyTorch is only an
# allocation/DLPack/stream/reference-check dependency of NVIDIA's own CuTe DSL
# examples. Importing torch and reading its version strings never initializes a
# device, so no GPU is used or required at build time.
#
# torch 2.10.0+cu130 requires cuda-bindings==13.0.3, while the CuTe DSL v4.6.1
# installer above resolves cuda-python 13.3.1 (which requires
# cuda-bindings~=13.3.1). Both pins below therefore name the coherent 13.0.3
# release, which satisfies torch and CuTe DSL 4.6.1's own cuda-python>=12.8
# constraint at the same time. No pin in VERSIONS.env is weakened, changed, or
# hidden; nothing is uninstalled to mask a conflict; the dependency graph is
# resolved, and the gate below proves it.
#
# The coherent cuda-python/cuda-bindings pair is resolved first, in one
# invocation, so the environment moves directly from the installer's default
# resolution to the pinned one and is never left transiently inconsistent;
# the pinned torch build then finds its exact cuda-bindings requirement already
# satisfied.
RUN python3 -m pip install --no-cache-dir \
        "cuda-python==${CUDA_PYTHON_VERSION}" \
        "cuda-bindings==${CUDA_BINDINGS_VERSION}" \
    && python3 -m pip install --no-cache-dir --index-url "${PYTORCH_INDEX_URL}" "torch==${PYTORCH_VERSION}"

# Hard build gate, run after every Python package is installed. It verifies the
# exact installed distribution versions through importlib.metadata, re-reads the
# runtime version strings of torch and CuTe DSL, and requires a fully consistent
# dependency graph. `pip check` is never suppressed, filtered, or allowed to
# fail: a broken requirement fails the image build. Still GPU-free.
RUN python3 -c "import os; from importlib.metadata import version; expected = {'torch': os.environ['PYTORCH_VERSION'], 'cuda-python': os.environ['CUDA_PYTHON_VERSION'], 'cuda-bindings': os.environ['CUDA_BINDINGS_VERSION'], 'nvidia-cutlass-dsl': os.environ['CUTLASS_VERSION'].lstrip('v')}; installed = {name: version(name) for name in expected}; assert installed == expected, f'installed distributions {installed} != pinned {expected}'; print('distributions OK:', installed)" \
    && python3 -c "import torch; v = torch.__version__; assert v == '${PYTORCH_VERSION}', f'unexpected torch version {v}'; c = torch.version.cuda; assert c == '${PYTORCH_CUDA_VERSION}', f'unexpected torch CUDA version {c}'; print('torch', v, 'cuda', c)" \
    && python3 -c "import cutlass; expected = '${CUTLASS_VERSION}'.lstrip('v'); v = cutlass.__version__; assert v == expected, f'CuTe DSL {v} != pinned {expected} after the auxiliary installs'; print('CuTeDSL', v, '(unchanged)')" \
    && echo "== pip check (hard gate: the whole environment must be consistent) ==" \
    && python3 -m pip check

WORKDIR /workspace
