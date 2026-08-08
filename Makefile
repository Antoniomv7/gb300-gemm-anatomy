# gb300-gemm-anatomy Makefile.
# Exposed targets: help, check-static, build-image, check-env, preflight,
# memory-ldgsts-build, memory-ldgsts-sass, memory-ldgsts-self-test,
# memory-ldgsts-smoke, memory-tma-build, memory-tma-sass,
# memory-tma-self-test, memory-tma-smoke, memory-paths-plan,
# memory-paths-check, memory-paths-smoke, memory-paths-p14-plan,
# memory-paths-p14-check, memory-paths-p14-pilot, memory-paths-p14-profile,
# memory-paths-p14-analyze, compute-umma-1sm-build, compute-umma-1sm-sass,
# compute-umma-1sm-check, compute-umma-1sm-self-test, compute-umma-1sm-smoke,
# compute-umma-2sm-build, compute-umma-2sm-sass, compute-umma-2sm-check,
# compute-umma-2sm-self-test, compute-umma-2sm-smoke, compute-umma-sweep-plan,
# compute-umma-sweep-check, compute-umma-sweep-smoke, compute-umma-p24-plan,
# compute-umma-p24-check, compute-umma-p24-pilot, compute-umma-p24-profile,
# compute-umma-p24-analyze, gemm-cutedsl-p31-check, gemm-cutedsl-p31-smoke,
# gemm-cutedsl-p32-check, gemm-cutedsl-p32-smoke, gemm-cublaslt-p33-check,
# gemm-cublaslt-p33-smoke, gemm-cutedsl-p34-check, gemm-cutedsl-p34-smoke.
# No target selects a GPU automatically, elevates privileges, or exceeds two
# build jobs.

# Two version contracts, never merged: VERSIONS.env is the closed global
# contract that the audited P1/P2 aggregators parse against their own closed key
# allowlist (an unknown key there fails a future P1/P2 campaign finalize), so
# every Phase 3-only pin lives in PHASE3_VERSIONS.env instead. PHASE3_VERSIONS.env
# extends the global contract and overrides nothing in it.
include VERSIONS.env
include PHASE3_VERSIONS.env

IMAGE_TAG ?= gb300-gemm-anatomy:phase0

# Derived pins: "13.1" from CUDA_VERSION=13.1.0, "4.6.1" from CUTLASS_VERSION=v4.6.1.
CUDA_SHORT_VERSION := $(basename $(CUDA_VERSION))
CUTEDSL_VERSION := $(patsubst v%,%,$(CUTLASS_VERSION))

MEMORY_LDGSTS_SRC := src/memory/ldgsts.cu
MEMORY_LDGSTS_BIN := build/memory/ldgsts
MEMORY_LDGSTS_SASS := build/memory/ldgsts.sass

MEMORY_TMA_SRC := src/memory/tma.cu
MEMORY_TMA_BIN := build/memory/tma
MEMORY_TMA_SASS := build/memory/tma.sass

EXP01_RUNNER := scripts/run_exp01_memory_paths.sh
EXP01_AGGREGATOR := scripts/aggregate_exp01_memory_paths.py

EXP01_P14_RUNNER := scripts/run_exp01_memory_paths_p14.sh
EXP01_P14_ANALYZER := scripts/analyze_exp01_memory_paths_p14.py
EXP01_P14_SAFE_CAPTURE := scripts/p14_safe_capture.py
EXP01_P14_NCU_BRIDGE := scripts/p14_ncu_bridge.py
EXP01_P14_PROTOCOL := src/memory/P1_4_PROTOCOL.md
EXP01_P14_RAW_ROOT := results/raw/exp01_memory_paths_p14

COMPUTE_UMMA_1SM_SRC := src/compute/umma_1sm.cu
COMPUTE_UMMA_1SM_BIN := build/compute/umma_1sm
COMPUTE_UMMA_1SM_SASS := build/compute/umma_1sm.sass
COMPUTE_UMMA_1SM_CHECKER := scripts/check_umma_1sm_sass.py
COMPUTE_UMMA_1SM_PROTOCOL := src/compute/P2_PROTOCOL.md
# nvcc's single-flag "-arch=sm_103a" shorthand does not propagate the "a"
# (architecture-specific) suffix to ptxas's SASS-generation target on this
# pinned CUDA 13.1.80 toolchain: it compiles P0/P1's LDGSTS/TMA code (which
# needs no sm_103a-only instruction) but fails every tcgen05 instruction with
# "not supported on .target 'sm_103'" (the "a" silently dropped; nvcc's own
# intermediate PTX file is even named "*.compute_103.ptx", confirming the "a"
# is lost before ptxas ever runs). Directly reproduced in this container: `nvcc
# -std=c++17 -O3 -lineinfo -arch=sm_103a -o ... src/compute/umma_1sm.cu` exits
# 255 with 3336 "not supported on .target 'sm_103'" errors and produces no
# binary, while the explicit split below (same pinned CUDA_ARCH value) exits 0
# and disassembles to all twelve expected UTCHMMA specializations (see
# src/compute/P2_PROTOCOL.md section 20 for the full recorded evidence).
# Splitting CUDA_ARCH into a virtual/real pair fixes it without changing
# VERSIONS.env or the pinned architecture string itself.
COMPUTE_UMMA_1SM_ARCH_FLAGS := -arch=compute_$(patsubst sm_%,%,$(CUDA_ARCH)) -code=$(CUDA_ARCH)

COMPUTE_UMMA_2SM_SRC := src/compute/umma_2sm.cu
COMPUTE_UMMA_2SM_BIN := build/compute/umma_2sm
COMPUTE_UMMA_2SM_SASS := build/compute/umma_2sm.sass
COMPUTE_UMMA_2SM_CHECKER := scripts/check_umma_2sm_sass.py
COMPUTE_UMMA_2SM_PROTOCOL := src/compute/P2_2_PROTOCOL.md
# Same pinned-toolchain requirement as P2.1 (see the COMPUTE_UMMA_1SM_ARCH_FLAGS
# comment above and src/compute/P2_PROTOCOL.md section 20): nvcc's single-flag
# "-arch=sm_103a" shorthand does not propagate the "a" suffix to ptxas's
# SASS-generation target on this pinned CUDA 13.1.80 toolchain, so every
# tcgen05 instruction fails to compile under that literal form. The explicit
# virtual/real split below, derived from the same pinned CUDA_ARCH value, is
# required and reproducibly verified for umma_2sm.cu as well.
COMPUTE_UMMA_2SM_ARCH_FLAGS := -arch=compute_$(patsubst sm_%,%,$(CUDA_ARCH)) -code=$(CUDA_ARCH)

EXP02_RUNNER := scripts/run_exp02_umma_throughput.sh
EXP02_AGGREGATOR := scripts/aggregate_exp02_umma_throughput.py
EXP02_PROTOCOL := src/compute/P2_3_PROTOCOL.md

EXP02_P24_RUNNER := scripts/run_exp02_umma_throughput_p24.sh
EXP02_P24_ANALYZER := scripts/analyze_exp02_umma_throughput_p24.py
EXP02_P24_SAFE_CAPTURE := scripts/p24_safe_capture.py
EXP02_P24_NCU_BRIDGE := scripts/p24_ncu_bridge.py
EXP02_P24_PROTOCOL := src/compute/P2_4_PROTOCOL.md
EXP02_P24_RAW_ROOT := results/raw/exp02_umma_throughput_p24

# P3.1: NVIDIA's own dense GEMM example, executed unmodified from the pinned
# CUTLASS checkout inside the image. This repository owns no GEMM source: the
# P3.1 files it adds are the protocol below and PHASE3_VERSIONS.env. Every
# provenance value comes from a pinned contract: CUTLASS_COMMIT from the global
# VERSIONS.env, and CUTEDSL_P31_EXAMPLE_PATH, CUTEDSL_P31_EXAMPLE_GIT_BLOB,
# CUTEDSL_P31_EXAMPLE_SHA256 from PHASE3_VERSIONS.env.
GEMM_P31_PROTOCOL := src/gemm/P3_1_PROTOCOL.md
GEMM_P31_EXAMPLE := /opt/cutlass/$(CUTEDSL_P31_EXAMPLE_PATH)

# P3.2: a thin, repository-owned orchestration wrapper around that same pinned
# example, executing one frozen BF16 shape and separating compile / first-launch
# / steady-state kernel time. It still vendors no NVIDIA GEMM source and adds no
# key to either version contract: it reuses P3.1's pins because it executes
# P3.1's file. See src/gemm/P3_2_PROTOCOL.md.
GEMM_P32_WRAPPER := src/gemm/cutedsl_gemm.py
GEMM_P32_CHECKER := scripts/check_cutedsl_gemm_p32.py
GEMM_P32_PROTOCOL := src/gemm/P3_2_PROTOCOL.md

# P3.3: the equivalent cuBLASLt baseline for exactly the same geometry and the
# same operands, issued through a direct cublasLtMatmul call. cuBLASLt already
# ships inside the pinned CUDA 13.1 image, so P3.3 adds no package, no image
# change, and no key to either version contract; the library's own runtime
# version is read with cublasLtGetVersion() instead of being pinned. The bridge
# owns no GEMM kernel and no NVIDIA source is copied, forked, patched, or
# vendored. See src/gemm/P3_3_PROTOCOL.md.
GEMM_P33_WRAPPER := src/gemm/cublaslt_gemm.py
GEMM_P33_BRIDGE := src/gemm/cublaslt_bridge.cu
GEMM_P33_CHECKER := scripts/check_cublaslt_gemm_p33.py
GEMM_P33_PROTOCOL := src/gemm/P3_3_PROTOCOL.md
# Container-private build output only: the repository is mounted read-only in
# the gate, and the wrapper looks the library up at exactly this fixed path
# (BRIDGE_LIBRARY_PATH), which is a constant and not a runtime control.
GEMM_P33_BRIDGE_DIR := /tmp/p33-bridge
GEMM_P33_BRIDGE_LIB := $(GEMM_P33_BRIDGE_DIR)/libp33_cublaslt_bridge.so
# Same pinned-toolchain requirement as P2.1/P2.2 (see COMPUTE_UMMA_1SM_ARCH_FLAGS
# above): nvcc's single-flag "-arch=sm_103a" shorthand does not propagate the
# "a" suffix to ptxas on this pinned CUDA 13.1.80 toolchain, so the explicit
# virtual/real split derived from the same pinned CUDA_ARCH value is used here
# too, unchanged and for the same reason.
GEMM_P33_ARCH_FLAGS := -arch=compute_$(patsubst sm_%,%,$(CUDA_ARCH)) -code=$(CUDA_ARCH)

# P3.4: the three frozen CuTe DSL execution variants at the same single shape.
# It reuses P3.1's pinned non-persistent example for the non-persistent variant
# and adds the official static-persistent example from the SAME pinned CUTLASS
# commit for the two persistent variants, so PHASE3_VERSIONS.env grows by
# exactly the three CUTEDSL_P34_* keys and VERSIONS.env is untouched. No NVIDIA
# GEMM source is copied, vendored, forked, or patched, no package or image
# changes, and no comparison of any kind is produced: P3.5 owns that. See
# src/gemm/P3_4_PROTOCOL.md.
GEMM_P34_WRAPPER := src/gemm/cutedsl_variants.py
GEMM_P34_CHECKER := scripts/check_cutedsl_variants_p34.py
GEMM_P34_PROTOCOL := src/gemm/P3_4_PROTOCOL.md
GEMM_P34_PERSISTENT_EXAMPLE := /opt/cutlass/$(CUTEDSL_P34_PERSISTENT_EXAMPLE_PATH)

# P3.5: the five frozen final shapes and the first explicit, descriptive
# comparison among the four candidates (the three P3.4 CuTe DSL execution
# variants plus the P3.3 cuBLASLt baseline). It reuses the SAME two already
# pinned official CUTLASS sources and the same cuBLASLt library that already
# ships in the pinned CUDA 13.1 image, so it adds no package, no image change,
# and NO key to either version contract. Its own C-ABI bridge is a separate
# translation unit from the closed P3.3 one, because P3.3 froze a single shape
# as compile-time constants while P3.5 must serve five - and it therefore
# refuses every geometry outside its own five-entry allowlist. No NVIDIA GEMM
# source is copied, vendored, forked, or patched. Every emitted row is
# publishable=false: the comparison fields are arithmetic, not a conclusion, and
# the pilot, the final campaigns, the statistics, and every interpretation are
# Phase 4 work. See src/gemm/P3_5_PROTOCOL.md.
GEMM_P35_WRAPPER := src/gemm/gemm_comparison.py
GEMM_P35_BRIDGE := src/gemm/cublaslt_bridge_p35.cu
GEMM_P35_CHECKER := scripts/check_gemm_comparison_p35.py
GEMM_P35_PROTOCOL := src/gemm/P3_5_PROTOCOL.md
# Container-private build output only: the repository is mounted read-only in
# the gate, and the wrapper looks the library up at exactly this fixed path
# (BRIDGE_LIBRARY_PATH), which is a constant and not a runtime control.
GEMM_P35_BRIDGE_DIR := /tmp/p35-bridge
GEMM_P35_BRIDGE_LIB := $(GEMM_P35_BRIDGE_DIR)/libp35_cublaslt_bridge.so
# Same pinned-toolchain requirement as P2.1/P2.2/P3.3: nvcc's single-flag
# "-arch=sm_103a" shorthand does not propagate the "a" suffix to ptxas on this
# pinned CUDA 13.1.80 toolchain, so the explicit virtual/real split derived from
# the same pinned CUDA_ARCH value is used here too, unchanged and for the same
# reason.
GEMM_P35_ARCH_FLAGS := -arch=compute_$(patsubst sm_%,%,$(CUDA_ARCH)) -code=$(CUDA_ARCH)

REQUIRED_FILES := \
	AGENTS.md README.md PLAN.md LICENSE .gitignore VERSIONS.env \
	PHASE3_VERSIONS.env \
	Dockerfile Makefile \
	scripts/run_container.sh scripts/preflight.sh scripts/check_ldgsts_sass.py \
	scripts/check_tma_sass.py \
	smoke/cuda_smoke.cu smoke/cutedsl_smoke.py \
	src/memory/ldgsts.cu src/memory/tma.cu src/memory/README.md \
	results/README.md \
	$(EXP01_RUNNER) $(EXP01_AGGREGATOR) \
	$(EXP01_P14_RUNNER) $(EXP01_P14_ANALYZER) $(EXP01_P14_SAFE_CAPTURE) $(EXP01_P14_NCU_BRIDGE) \
	$(EXP01_P14_PROTOCOL) \
	$(COMPUTE_UMMA_1SM_SRC) $(COMPUTE_UMMA_1SM_CHECKER) $(COMPUTE_UMMA_1SM_PROTOCOL) \
	$(COMPUTE_UMMA_2SM_SRC) $(COMPUTE_UMMA_2SM_CHECKER) $(COMPUTE_UMMA_2SM_PROTOCOL) \
	$(EXP02_RUNNER) $(EXP02_AGGREGATOR) $(EXP02_PROTOCOL) \
	$(EXP02_P24_RUNNER) $(EXP02_P24_ANALYZER) $(EXP02_P24_SAFE_CAPTURE) $(EXP02_P24_NCU_BRIDGE) \
	$(EXP02_P24_PROTOCOL) \
	$(GEMM_P31_PROTOCOL) \
	$(GEMM_P32_WRAPPER) $(GEMM_P32_CHECKER) $(GEMM_P32_PROTOCOL) \
	$(GEMM_P33_WRAPPER) $(GEMM_P33_BRIDGE) $(GEMM_P33_CHECKER) $(GEMM_P33_PROTOCOL) \
	$(GEMM_P34_WRAPPER) $(GEMM_P34_CHECKER) $(GEMM_P34_PROTOCOL) \
	$(GEMM_P35_WRAPPER) $(GEMM_P35_BRIDGE) $(GEMM_P35_CHECKER) $(GEMM_P35_PROTOCOL)

.DEFAULT_GOAL := help
.PHONY: help check-static build-image check-env preflight \
	memory-ldgsts-build memory-ldgsts-sass memory-ldgsts-self-test memory-ldgsts-smoke \
	memory-tma-build memory-tma-sass memory-tma-self-test memory-tma-smoke \
	memory-paths-plan memory-paths-check memory-paths-smoke \
	memory-paths-p14-plan memory-paths-p14-check memory-paths-p14-pilot \
	memory-paths-p14-profile memory-paths-p14-analyze \
	compute-umma-1sm-build compute-umma-1sm-sass compute-umma-1sm-check \
	compute-umma-1sm-self-test compute-umma-1sm-smoke \
	compute-umma-2sm-build compute-umma-2sm-sass compute-umma-2sm-check \
	compute-umma-2sm-self-test compute-umma-2sm-smoke \
	compute-umma-p24-plan compute-umma-p24-check compute-umma-p24-pilot \
	compute-umma-p24-profile compute-umma-p24-analyze \
	compute-umma-sweep-plan compute-umma-sweep-check compute-umma-sweep-smoke \
	gemm-cutedsl-p31-check gemm-cutedsl-p31-smoke \
	gemm-cutedsl-p32-check gemm-cutedsl-p32-smoke \
	gemm-cublaslt-p33-check gemm-cublaslt-p33-smoke \
	gemm-cutedsl-p34-check gemm-cutedsl-p34-smoke \
	gemm-comparison-p35-check gemm-comparison-p35-smoke

help:
	@echo "gb300-gemm-anatomy — Phase 0 + P1.1 (LDGSTS) + P1.2 (TMA) + P1.3 (sweep) targets"
	@echo ""
	@echo "  make help                     Show this help."
	@echo "  make check-static             Static validation: no Docker, no GPU, no network."
	@echo "  make build-image              Build the pinned image ($(IMAGE_TAG)). No GPU."
	@echo "  make check-env                Check tools, versions, and the Python dependency"
	@echo "                                graph (pip check) inside a GPU-less container."
	@echo "  make preflight                Run the single-GPU Phase 0 preflight. Requires"
	@echo "                                an explicit BLACKWELL_GPU_INDEX=<physical-index>;"
	@echo "                                never selects a GPU automatically."
	@echo ""
	@echo "  -- P1.1 LDGSTS (GPU-free build/SASS targets below; GPU targets require"
	@echo "     BLACKWELL_GPU_INDEX) --"
	@echo "  make memory-ldgsts-build      Compile the P1.1 LDGSTS microbenchmark. No GPU."
	@echo "  make memory-ldgsts-sass       Disassemble it and verify per-specialization"
	@echo "                                16-byte LDGSTS groups and commit/wait barriers."
	@echo "                                No GPU."
	@echo "  make memory-ldgsts-self-test  Validate all nine specializations on GPU (no"
	@echo "                                publishable numbers). Requires BLACKWELL_GPU_INDEX."
	@echo "  make memory-ldgsts-smoke      Self-test, then one short run_kind=smoke"
	@echo "                                measurement (NOT a final result). Requires"
	@echo "                                BLACKWELL_GPU_INDEX."
	@echo ""
	@echo "  -- P1.2 TMA (GPU-free build/SASS targets below; GPU targets require"
	@echo "     BLACKWELL_GPU_INDEX) --"
	@echo "  make memory-tma-build         Compile the P1.2 2D unicast TMA microbenchmark."
	@echo "                                No GPU."
	@echo "  make memory-tma-sass          Disassemble it and verify per-specialization"
	@echo "                                UTMALDG.2D loads and transaction-barrier"
	@echo "                                completion. No GPU."
	@echo "  make memory-tma-self-test     Validate all nine specializations on GPU (no"
	@echo "                                publishable numbers). Requires BLACKWELL_GPU_INDEX."
	@echo "  make memory-tma-smoke         Self-test, then one short run_kind=smoke"
	@echo "                                measurement (NOT a final result). Requires"
	@echo "                                BLACKWELL_GPU_INDEX."
	@echo ""
	@echo "  -- P1.3 joint LDGSTS/TMA sweep infrastructure (exp01_memory_paths) --"
	@echo "  GPU-free P1.3 planning/checking (no GPU, no Docker, no network):"
	@echo "  make memory-paths-plan       Print the deterministic 18-invocation plan."
	@echo "  make memory-paths-check      Shell/Python syntax, executable bits, GPU-free"
	@echo "                               synthetic tests, and exact 18-way plan validation."
	@echo "  GPU-executing 18-way functional smoke (uses the P1.1/P1.2 CUDA build/SASS"
	@echo "  gates above, then both binary self-tests, then all 18 smoke configurations;"
	@echo "  requires BLACKWELL_GPU_INDEX; never selects a GPU automatically):"
	@echo "  make memory-paths-smoke      run_kind=smoke only; functional verification,"
	@echo "                               NOT a publishable performance result."
	@echo "  P1.3 never runs Nsight Compute and never collects run_kind=benchmark data;"
	@echo "  P1.4 (below) owns the pilot, profiling, and interpretation."
	@echo ""
	@echo "  -- P1.4 profiling, HBM validation, analysis, pilot (see"
	@echo "     src/memory/P1_4_PROTOCOL.md; implemented, audited, verified on GB300;"
	@echo "     reviewed pilot complete, publishable=false) --"
	@echo "  GPU-free P1.4 planning/checking (no GPU, no Docker, no network):"
	@echo "  make memory-paths-p14-plan     Print the frozen P1.3 18-invocation pilot plan"
	@echo "                                 and the frozen P1.4 six-case NCU plan."
	@echo "  make memory-paths-p14-check    Shell/Python syntax, executable bits, GPU-free"
	@echo "                                 synthetic/adversarial tests, and exact-plan"
	@echo "                                 validation (18-way P1.3, six-way P1.4)."
	@echo "  GPU-executing (requires BLACKWELL_GPU_INDEX, P1_4_CAMPAIGN_ID, and"
	@echo "  P1_4_PREFLIGHT_SUMMARY; never selects a GPU automatically):"
	@echo "  make memory-paths-p14-pilot    The frozen 18-configuration run_kind=benchmark"
	@echo "                                 pilot, through the unmodified P1.3 runner."
	@echo "  make memory-paths-p14-profile  Nsight Compute on exactly the six frozen cases"
	@echo "                                 against an already-PILOT_COMPLETE campaign."
	@echo "  GPU-free P1.4 analysis (requires P1_4_CAMPAIGN_ID; a completed pilot+profile):"
	@echo "  make memory-paths-p14-analyze  Validate and analyze a COMPLETE P1.4 campaign;"
	@echo "                                 all outputs remain publishable=false."
	@echo ""
	@echo "  -- P2.1 1-SM BF16 UMMA (tcgen05.mma, kind::f16, cta_group::1; see"
	@echo "     src/compute/P2_PROTOCOL.md; implemented, independently audited,"
	@echo "     functionally verified on GB300, no publishable result;"
	@echo "     P2.2 implemented; independently audited; GB300 verification passed."
	@echo "     P2.3 implemented; independently audited; GB300 verification passed."
	@echo "     P2.4 implemented; independently audited; GB300 verification passed. No"
	@echo "     publishable P2.2/P2.3/P2.4 result exists.) --"
	@echo "  GPU-free build/SASS/check (no GPU, no network):"
	@echo "  make compute-umma-1sm-build    Compile the twelve P2.1 specializations. No GPU."
	@echo "  make compute-umma-1sm-sass     Disassemble and verify the real cubin: exactly"
	@echo "                                 twelve UTCHMMA bursts of depth instructions each,"
	@echo "                                 a full TMEM lifecycle, no forbidden/2-SM"
	@echo "                                 instruction. No GPU."
	@echo "  make compute-umma-1sm-check    Python syntax, checker self-test, exactly-twelve-"
	@echo "                                 specializations and forbidden-pattern source"
	@echo "                                 checks, honest status reporting, plus the real"
	@echo "                                 cubin SASS gate above. No GPU, no network."
	@echo "  GPU-executing (requires BLACKWELL_GPU_INDEX; never selects a GPU automatically):"
	@echo "  make compute-umma-1sm-self-test  Validate all twelve specializations on GPU (no"
	@echo "                                   publishable numbers)."
	@echo "  make compute-umma-1sm-smoke      Self-test, then one short run_kind=smoke"
	@echo "                                   measurement (NOT a final result)."
	@echo ""
	@echo "  -- P2.2 2-SM BF16 UMMA (tcgen05.mma, kind::f16, cta_group::2, one static"
	@echo "     two-CTA cluster; see src/compute/P2_2_PROTOCOL.md; implemented,"
	@echo "     independently audited, verified on GB300, no publishable result) --"
	@echo "  GPU-free build/SASS/check (no GPU, no network):"
	@echo "  make compute-umma-2sm-build    Compile the twelve P2.2 specializations. No GPU."
	@echo "  make compute-umma-2sm-sass     Disassemble and verify the real cubin: exactly"
	@echo "                                 twelve UTCHMMA.2CTA bursts of depth instructions"
	@echo "                                 each, a full collective TMEM lifecycle, ELF"
	@echo "                                 two-CTA cluster attributes, no forbidden/1-SM-"
	@echo "                                 fallback instruction. No GPU."
	@echo "  make compute-umma-2sm-check    Python syntax, checker self-test, exactly-twelve-"
	@echo "                                 specializations and forbidden-pattern source"
	@echo "                                 checks, honest status reporting, plus the real"
	@echo "                                 cubin SASS gate above. No GPU, no network."
	@echo "  GPU-executing (requires BLACKWELL_GPU_INDEX; never selects a GPU automatically):"
	@echo "  make compute-umma-2sm-self-test  Validate all twelve specializations on GPU (no"
	@echo "                                   publishable numbers)."
	@echo "  make compute-umma-2sm-smoke      Self-test, then one short run_kind=smoke"
	@echo "                                   measurement (NOT a final result)."
	@echo ""
	@echo "  -- P2.3 joint 1-SM/2-SM BF16 UMMA sweep infrastructure (exp02_umma_throughput;"
	@echo "     see src/compute/P2_3_PROTOCOL.md; implemented; independently audited;"
	@echo "     verified on GB300. Reuses the audited P2.1/P2.2 binaries"
	@echo "     and CLIs unmodified; introduces no new CUDA kernel; no Nsight Compute) --"
	@echo "  GPU-free P2.3 planning/checking (no GPU, no Docker, no network):"
	@echo "  make compute-umma-sweep-plan   Print the deterministic 24-invocation plan."
	@echo "  make compute-umma-sweep-check  Shell/Python syntax, executable bits, GPU-free"
	@echo "                                 synthetic tests, exact 24-way plan validation,"
	@echo "                                 plus the existing P2.1/P2.2 build/SASS gates."
	@echo "  GPU-executing 24-way functional smoke (uses the compute-umma-1sm-sass/"
	@echo "  compute-umma-2sm-sass gates above, then both binaries' full self-tests, then"
	@echo "  all 24 smoke configurations; requires BLACKWELL_GPU_INDEX; never selects a"
	@echo "  GPU automatically):"
	@echo "  make compute-umma-sweep-smoke  run_kind=smoke only; functional verification,"
	@echo "                                 NOT a publishable performance result."
	@echo "  P2.3 never runs Nsight Compute and never computes TFLOP/s, an empirical"
	@echo "  ceiling, 1-SM/2-SM speedup, scaling efficiency, or saturation; P2.4 owns"
	@echo "  that interpretation."
	@echo ""
	@echo "  -- P2.4 profiling and empirical BF16 UMMA per-SM ceiling candidate"
	@echo "     (exp02_umma_throughput_p24; see src/compute/P2_4_PROTOCOL.md; implemented;"
	@echo "     independently audited and verified on GB300. Drives the"
	@echo "     unmodified P2.3 runner for one frozen 24-configuration pilot, profiles the"
	@echo "     same 24 configurations with Nsight Compute, and computes deterministic"
	@echo "     TFLOP/s, 1-SM/2-SM scaling, candidate saturation, and an empirical"
	@echo "     per-SM ceiling candidate. Campaign 20260805T102759Z reached ANALYZED;"
	@echo "     Phase 2 is closed; no publishable result exists.) --"
	@echo "  GPU-free P2.4 planning/checking (no GPU, no Docker, no network):"
	@echo "  make compute-umma-p24-plan     Print the frozen 24-case profile plan (the"
	@echo "                                 same P2.3 configurations, plus kernel_symbol)."
	@echo "  make compute-umma-p24-check    Shell/Python syntax, executable bits, GPU-free"
	@echo "                                 synthetic/adversarial tests, exact 24-way plan"
	@echo "                                 validation, plus the existing P2.1/P2.2/P2.3 gates."
	@echo "  GPU-executing (requires BLACKWELL_GPU_INDEX, P2_4_CAMPAIGN_ID, and"
	@echo "  P2_4_PREFLIGHT_SUMMARY; never selects a GPU automatically):"
	@echo "  make compute-umma-p24-pilot    The frozen 24-configuration run_kind=benchmark"
	@echo "                                 pilot, through the unmodified P2.3 runner."
	@echo "  make compute-umma-p24-profile  Nsight Compute on the same 24 configurations"
	@echo "                                 against an already-PILOT_COMPLETE campaign."
	@echo "  GPU-free P2.4 analysis (requires P2_4_CAMPAIGN_ID; a completed pilot+profile):"
	@echo "  make compute-umma-p24-analyze  Validate and analyze a COMPLETE P2.4 campaign;"
	@echo "                                 all outputs remain publishable=false; state"
	@echo "                                 becomes ANALYZED or, if the mandatory SM-clock"
	@echo "                                 metric could not be trusted for every"
	@echo "                                 configuration, INCONCLUSIVE (no TFLOP/s emitted)."
	@echo ""
	@echo "  -- P3.1 pinned official CuTe DSL example (see src/gemm/P3_1_PROTOCOL.md;"
	@echo "     implemented, independently audited, and verified on GB300; closed. Executes"
	@echo "     NVIDIA's own unmodified dense_gemm.py from the pinned /opt/cutlass"
	@echo "     checkout: BF16 x BF16 -> FP32, (M,N,K,L)=(256,256,512,1), non-persistent,"
	@echo "     1-CTA MMA group, mma tiler (128,128), cluster (1,1), TMA store. This"
	@echo "     repository owns no GEMM source and P3.1 produces NO performance result.) --"
	@echo "  GPU-free P3.1 provenance/environment gate (no GPU, no network):"
	@echo "  make gemm-cutedsl-p31-check    Verify /opt/cutlass HEAD, checkout cleanliness,"
	@echo "                                 the example's regular-file identity, Git blob"
	@echo "                                 SHA and SHA-256, the CuTe DSL/PyTorch/cuda-python/"
	@echo "                                 cuda-bindings pins, a consistent dependency graph"
	@echo "                                 (pip check), and that the example's own --help"
	@echo "                                 runs GPU-free."
	@echo "  GPU-executing (requires BLACKWELL_GPU_INDEX; never selects a GPU automatically):"
	@echo "  make gemm-cutedsl-p31-smoke    Re-check the upstream commit and SHA-256 inside"
	@echo "                                 the GPU container, then run the frozen official"
	@echo "                                 command with mandatory reference validation."
	@echo "                                 Functional smoke check only, NOT a performance"
	@echo "                                 result; any internally computed timing is"
	@echo "                                 discarded."
	@echo ""
	@echo "  -- P3.2 one-shape wrapper (see src/gemm/P3_2_PROTOCOL.md; implemented,"
	@echo "     independently audited and verified on GB300; closed. Drives the same"
	@echo "     pinned NVIDIA example through a repository-owned wrapper at the frozen"
	@echo "     BF16 shape (M,N,K,L)=(4096,4096,4096,1), non-persistent, 1-CTA MMA group,"
	@echo "     mma tiler (128,128), cluster (1,1), TMA store, seed 1111, and separates"
	@echo "     compile_time_ms / first_launch_ms / kernel_time_ms. Correctness is"
	@echo "     mandatory and always precedes any timing. Every emitted row is"
	@echo "     publishable=false: P3.2 produces NO experimental result and NO"
	@echo "     cuBLASLt comparison.) --"
	@echo "  GPU-free P3.2 contract gate (no GPU, no network; runs the P3.1 gate first):"
	@echo "  make gemm-cutedsl-p32-check    Re-verify the upstream commit, blob, SHA-256,"
	@echo "                                 installed versions and dependency consistency,"
	@echo "                                 then compile both P3.2 files and run the"
	@echo "                                 wrapper's GPU-free --help and --self-test plus"
	@echo "                                 the checker and its own self-test."
	@echo "  GPU-executing (requires BLACKWELL_GPU_INDEX; never selects a GPU automatically):"
	@echo "  make gemm-cutedsl-p32-smoke    Re-check the upstream commit and SHA-256 inside"
	@echo "                                 the GPU container, then run the frozen one-shape"
	@echo "                                 wrapper with 2 warm-ups and 10 measured launches."
	@echo "                                 Emits one non-publishable CSV row of functional"
	@echo "                                 evidence, NOT an experimental result."
	@echo ""
	@echo "  -- P3.3 cuBLASLt baseline (see src/gemm/P3_3_PROTOCOL.md; implemented,"
	@echo "     independently audited and verified on GB300; closed. Runs the SAME frozen"
	@echo "     BF16 geometry as P3.2 -- (M,N,K,L)=(4096,4096,4096,1), C = A x B^T, seed"
	@echo "     1111, hot reused operands -- on the SAME operands, through a direct"
	@echo "     cublasLtMatmul call: A row-major MxK lda=K, B row-major NxK ldb=K, C/D"
	@echo "     row-major MxN ldc=ldd=N, OP_N/OP_T, CUDA_R_16BF in, CUDA_R_32F out,"
	@echo "     CUBLAS_COMPUTE_32F, host pointer mode, default identity epilogue, alpha=1,"
	@echo "     beta=0. Fixed non-autotuned policy: 64 MiB workspace limit, 32 heuristic"
	@echo "     results requested, CUBLASLT_SEARCH_BEST_FIT, first supported result taken"
	@echo "     and re-validated with cublasLtMatmulAlgoCheck; no candidate is ever"
	@echo "     benchmarked. Correctness is mandatory and always precedes any timing."
	@echo "     Separates setup_time_ms / first_launch_ms / kernel_time_ms -- setup is NOT"
	@echo "     compilation and the P3.2 field name is never reused. Every emitted row is"
	@echo "     publishable=false: P3.3 produces NO experimental result and NO CuTe-versus-"
	@echo "     cuBLASLt comparison, which belongs to P3.5.) --"
	@echo "  GPU-free P3.3 contract gate (no GPU, no network; runs the P3.2 gate first):"
	@echo "  make gemm-cublaslt-p33-check   Compile the C-ABI cuBLASLt bridge for $(CUDA_ARCH)"
	@echo "                                 into container-private /tmp, inspect its ELF"
	@echo "                                 symbols to prove the measured path references"
	@echo "                                 cublasLtMatmul and no fallback GEMM API, then run"
	@echo "                                 the wrapper's GPU-free --help and --self-test"
	@echo "                                 plus the checker and its own self-test."
	@echo "  GPU-executing (requires BLACKWELL_GPU_INDEX; never selects a GPU automatically):"
	@echo "  make gemm-cublaslt-p33-smoke   Compile the bridge inside the already-selected"
	@echo "                                 GPU container, re-check the upstream commit and"
	@echo "                                 SHA-256 there, then run the frozen cuBLASLt"
	@echo "                                 baseline with 2 warm-ups and 10 measured"
	@echo "                                 launches. Emits one non-publishable CSV row of"
	@echo "                                 functional evidence, NOT an experimental result"
	@echo "                                 and NOT a performance comparison."
	@echo ""
	@echo "  -- P3.4 three execution variants (see src/gemm/P3_4_PROTOCOL.md; implemented,"
	@echo "     independently audited, and verified on GB300. Runs exactly three"
	@echo "     frozen candidates at the SAME single shape (M,N,K,L)=(4096,4096,4096,1) on"
	@echo "     ONE shared, immutable operand set and one shared untimed FP32 oracle:"
	@echo "       nonpersistent_1cta  DenseGemmKernel            tiler (128,128) cluster (1,1)"
	@echo "       persistent_1cta     PersistentDenseGemmKernel  tiler (128,128) cluster (1,1)"
	@echo "       persistent_2cta     PersistentDenseGemmKernel  tiler (256,128) cluster (2,1)"
	@echo "     The 2-CTA row uses an M tile of 256 so each participating CTA keeps a local"
	@echo "     M extent of 128, matching P2.2's two-SM geometry. Both kernels come from"
	@echo "     pinned unmodified official NVIDIA examples in the same pinned CUTLASS"
	@echo "     commit; max_active_clusters comes from the official hardware helper and is"
	@echo "     never guessed. Correctness is mandatory and precedes every timing, per"
	@echo "     variant, and the four output lines appear only if ALL THREE variants pass."
	@echo "     Every row is publishable=false: P3.4 produces NO experimental result, NO"
	@echo "     ranking, and NO variant or cuBLASLt comparison, which belong to P3.5.) --"
	@echo "  GPU-free P3.4 contract gate (no GPU, no network; runs the P3.3 gate first):"
	@echo "  make gemm-cutedsl-p34-check    Revalidate the CUTLASS checkout and BOTH pinned"
	@echo "                                 official sources, verify the pinned package"
	@echo "                                 versions and dependency graph, then compile the"
	@echo "                                 P3.4 files and run the wrapper's GPU-free --help"
	@echo "                                 and --self-test plus the checker and its own"
	@echo "                                 self-test."
	@echo "  GPU-executing (requires BLACKWELL_GPU_INDEX; never selects a GPU automatically):"
	@echo "  make gemm-cutedsl-p34-smoke    Re-check both upstream sources inside the GPU"
	@echo "                                 container, then run all three frozen variants"
	@echo "                                 with 2 warm-ups and 10 measured launches each."
	@echo "                                 Emits four CSV lines of functional evidence, NOT"
	@echo "                                 an experimental result and NOT a comparison."
	@echo ""
	@echo "  -- P3.5 five shapes and comparison (see src/gemm/P3_5_PROTOCOL.md; closed as"
	@echo "     YES / YES / YES after independent audit and GB300 verification."
	@echo "     Runs the SAME four frozen candidates on EACH of the five frozen final"
	@echo "     shapes, always in"
	@echo "     shape-major order, on one shared immutable operand set and one untimed FP32"
	@echo "     oracle per shape:"
	@echo "       shapes      (4096,4096,4096,1) (8192,8192,8192,1) (16384,512,4096,1)"
	@echo "                   (32768,512,4096,1) (512,16384,4096,1)"
	@echo "       candidates  nonpersistent_1cta  persistent_1cta  persistent_2cta"
	@echo "                   cublaslt/heuristic_first_supported (the comparison baseline)"
	@echo "     No arbitrary shape is reachable from the CLI, the environment, a config"
	@echo "     file, or an input CSV: the wrapper and the C bridge freeze the same five"
	@echo "     independently and must agree. The cuBLASLt policy is exactly P3.3's and"
	@echo "     never changes; only which supported algorithm the vendor heuristic returns"
	@echo "     may differ per shape. Correctness is mandatory and precedes every timing,"
	@echo "     per candidate, and the 21 output lines appear only if ALL 20 measurements"
	@echo "     pass. The comparison is descriptive only: exact 2*M*N*K FLOP counts,"
	@echo "     TFLOP/s, a ratio and a signed gap against the cuBLASLt baseline (a negative"
	@echo "     gap means faster and is NEVER clamped), a rank, and the best CuTe DSL"
	@echo "     variant. NO confidence interval, p-value, outlier removal, roofline,"
	@echo "     bandwidth, or causal interpretation is computed. Every row is"
	@echo "     publishable=false: P3.5 is NOT a campaign and NOT a final result, and"
	@echo "     beating cuBLASLt is NOT a success criterion.) --"
	@echo "  GPU-free P3.5 contract gate (no GPU, no network; runs the P3.4 gate first):"
	@echo "  make gemm-comparison-p35-check Revalidate the CUTLASS checkout and BOTH pinned"
	@echo "                                 official sources, verify the pinned package"
	@echo "                                 versions and dependency graph, compile the P3.5"
	@echo "                                 cuBLASLt bridge into container-private /tmp and"
	@echo "                                 inspect its ELF symbols and dynamic"
	@echo "                                 dependencies (cublasLtMatmul present, no"
	@echo "                                 fallback GEMM API present), then run the"
	@echo "                                 wrapper's GPU-free --help and --self-test plus"
	@echo "                                 the checker and its own self-test."
	@echo "  GPU-executing (requires BLACKWELL_GPU_INDEX; never selects a GPU automatically):"
	@echo "  make gemm-comparison-p35-smoke Re-check both upstream sources inside the GPU"
	@echo "                                 container, compile the bridge into private /tmp,"
	@echo "                                 then run all five shapes x four candidates with"
	@echo "                                 2 warm-ups and 10 measured launches each. Emits"
	@echo "                                 21 CSV lines of functional comparison evidence,"
	@echo "                                 NOT an experimental result, NOT a statistical"
	@echo "                                 conclusion, and NOT a Phase 4 interpretation."
	@echo ""
	@echo "Pinned global contract (VERSIONS.env, unchanged since Phase 0 and consumed"
	@echo "unmodified by the closed P1/P2 aggregators): CUDA $(CUDA_VERSION), CUTLASS"
	@echo "$(CUTLASS_VERSION), arch $(CUDA_ARCH), max build jobs $(MAX_BUILD_JOBS)."
	@echo "Phase 3 extension (PHASE3_VERSIONS.env): auxiliary PyTorch $(PYTORCH_VERSION)"
	@echo "(CUDA $(PYTORCH_CUDA_VERSION)), cuda-python $(CUDA_PYTHON_VERSION), cuda-bindings $(CUDA_BINDINGS_VERSION)."

check-static:
	@echo "== required files present =="
	@missing=0; for f in $(REQUIRED_FILES); do \
		if [ ! -f "$$f" ]; then echo "MISSING: $$f"; missing=1; fi; \
	done; [ "$$missing" -eq 0 ]
	@echo "== bash syntax =="
	bash -n scripts/run_container.sh
	bash -n scripts/preflight.sh
	@echo "== scripts are executable =="
	@test -x scripts/run_container.sh
	@test -x scripts/preflight.sh
	@echo "== python syntax =="
	python3 -m py_compile smoke/cutedsl_smoke.py
	@rm -rf smoke/__pycache__
	@echo "== version contract format =="
	@grep -Eq '^CUDA_VERSION=13\.1\.0$$' VERSIONS.env
	@grep -Eq '^CUDA_IMAGE_DIGEST=sha256:[0-9a-f]{64}$$' VERSIONS.env
	@grep -Eq '^CUDA_IMAGE_PLATFORM=linux/(amd64|arm64)$$' VERSIONS.env
	@grep -Eq '^CUTLASS_COMMIT=[0-9a-f]{40}$$' VERSIONS.env
	@grep -Eq '^CUDA_ARCH=sm_103a$$' VERSIONS.env
	@grep -Eq '^MAX_BUILD_JOBS=2$$' VERSIONS.env
	@echo "== VERSIONS.env stays the closed global contract (no Phase 3 keys) =="
	@! grep -nE '^(PYTORCH_|CUDA_PYTHON_VERSION|CUDA_BINDINGS_VERSION|CUTEDSL_P31_)' VERSIONS.env
	@echo "== the real P1/P2 version parsers accept the real VERSIONS.env =="
	python3 -c 'import sys; sys.path.insert(0, "scripts"); import aggregate_exp01_memory_paths as p1, aggregate_exp02_umma_throughput as p2; p1.parse_versions_env(); p2.parse_versions_env(); print("P1/P2 VERSIONS.env compatibility: PASS")'
	@rm -rf scripts/__pycache__
	@echo "== P3.1 version contract format (PHASE3_VERSIONS.env, exact non-floating pins) =="
	@grep -Eq '^PYTORCH_VERSION=2\.10\.0\+cu130$$' PHASE3_VERSIONS.env
	@grep -Eq '^PYTORCH_INDEX_URL=https://download\.pytorch\.org/whl/cu130$$' PHASE3_VERSIONS.env
	@grep -Eq '^PYTORCH_CUDA_VERSION=13\.0$$' PHASE3_VERSIONS.env
	@grep -Eq '^CUDA_PYTHON_VERSION=13\.0\.3$$' PHASE3_VERSIONS.env
	@grep -Eq '^CUDA_BINDINGS_VERSION=13\.0\.3$$' PHASE3_VERSIONS.env
	@grep -Eq '^CUTEDSL_P31_EXAMPLE_PATH=examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm\.py$$' PHASE3_VERSIONS.env
	@grep -Eq '^CUTEDSL_P31_EXAMPLE_GIT_BLOB=[0-9a-f]{40}$$' PHASE3_VERSIONS.env
	@grep -Eq '^CUTEDSL_P31_EXAMPLE_SHA256=[0-9a-f]{64}$$' PHASE3_VERSIONS.env
	@echo "== PHASE3_VERSIONS.env extends, never redefines, the global contract =="
	@! grep -nE '^(CUDA_VERSION|CUDA_IMAGE|CUDA_IMAGE_DIGEST|CUDA_IMAGE_PLATFORM|CUTLASS_VERSION|CUTLASS_COMMIT|CUDA_ARCH|MAX_BUILD_JOBS)=' PHASE3_VERSIONS.env
	@echo "== Dockerfile consistent with VERSIONS.env and PHASE3_VERSIONS.env =="
	@grep -Fq "$(CUDA_IMAGE)@$(CUDA_IMAGE_DIGEST)" Dockerfile
	@grep -Fq "CUTLASS_COMMIT=$(CUTLASS_COMMIT)" Dockerfile
	@grep -Fq "PYTORCH_VERSION=$(PYTORCH_VERSION)" Dockerfile
	@grep -Fq "PYTORCH_INDEX_URL=$(PYTORCH_INDEX_URL)" Dockerfile
	@grep -Fq "PYTORCH_CUDA_VERSION=$(PYTORCH_CUDA_VERSION)" Dockerfile
	@grep -Fq "CUDA_PYTHON_VERSION=$(CUDA_PYTHON_VERSION)" Dockerfile
	@grep -Fq "CUDA_BINDINGS_VERSION=$(CUDA_BINDINGS_VERSION)" Dockerfile
	@echo "== the image build gates on a consistent dependency graph, unsuppressed =="
	@grep -Fq 'python3 -m pip check' Dockerfile
	@! grep -nE 'pip check[^&|]*(\|\||;[[:space:]]*true|\|[[:space:]]*(grep|sed|awk))' Dockerfile Makefile
	@! grep -nE -- '--no-deps' Dockerfile
	@echo "== preflight targets pinned architecture =="
	@grep -Fq -- "-arch=$(CUDA_ARCH)" scripts/preflight.sh
	@echo "== forbidden patterns absent from scripts, Dockerfile, smoke, memory =="
	@pat='--gpus[ =]+all|NVIDIA_VISIBLE_DEVICES=all|--privileged|--pid[ =]+host|docker\.sock|--cap-add|SYS_ADMIN|set -x'; \
	pat="$$pat|\bs""udo\b|\$$\(np""roc\)|nvidia-smi[^|]*(-pm|--persistence-mode|-lgc|--lock-gpu-clocks|-pl|--power-limit)"; \
	! grep -nE -- "$$pat" scripts/run_container.sh scripts/preflight.sh Dockerfile \
		smoke/cuda_smoke.cu smoke/cutedsl_smoke.py \
		src/memory/ldgsts.cu scripts/check_ldgsts_sass.py \
		src/memory/tma.cu scripts/check_tma_sass.py
	@! grep -nE "s""udo|np""roc" Makefile
	@echo "== LDGSTS source uses the frozen PTX path (P1.1 contract) =="
	@grep -Fq 'cp.async.cg.shared.global' src/memory/ldgsts.cu
	@grep -Fq 'cp.async.commit_group' src/memory/ldgsts.cu
	@grep -Fq 'cp.async.wait_group' src/memory/ldgsts.cu
	@! grep -nE 'cuda::memcpy_async|cooperative_groups::memcpy_async|__pipeline_memcpy_async|cp\.async\.bulk' src/memory/ldgsts.cu
	@echo "== LDGSTS Makefile target pins the contract architecture =="
	@grep -Fq -- '-arch=$$(CUDA_ARCH)' Makefile
	@echo "== LDGSTS SASS checker syntax and synthetic contract tests =="
	python3 -m py_compile scripts/check_ldgsts_sass.py
	python3 scripts/check_ldgsts_sass.py --self-test
	@rm -rf scripts/__pycache__
	@test -x scripts/check_ldgsts_sass.py
	@echo "== TMA source uses the frozen 2D unicast TMA path (P1.2 contract) =="
	@grep -Fq 'cp_async_bulk_tensor' src/memory/tma.cu
	@grep -Fq 'mbarrier_arrive_expect_tx' src/memory/tma.cu
	@grep -Fq 'mbarrier_try_wait_parity' src/memory/tma.cu
	@grep -Fq 'elect_sync' src/memory/tma.cu
	@grep -Fq 'cuTensorMapEncodeTiled' src/memory/tma.cu
	@grep -Fq 'cudaGetDriverEntryPointByVersion' src/memory/tma.cu
	@echo "== TMA source absent of prohibited 1D/multicast/cluster/LDGSTS transfer paths =="
	@! grep -nE 'cp\.async\.cg\.shared\.global|cuda::memcpy_async|cooperative_groups::memcpy_async|__pipeline_memcpy_async|cp_async_bulk\(' src/memory/tma.cu
	@! grep -nE 'space_cluster|cta_group|multicast|MULTICAST|UBLKCP' src/memory/tma.cu
	@echo "== TMA Makefile target pins the contract architecture =="
	@grep -Fq -- '-arch=$$(CUDA_ARCH)' Makefile
	@echo "== TMA SASS checker syntax and synthetic contract tests =="
	python3 -m py_compile scripts/check_tma_sass.py
	python3 scripts/check_tma_sass.py --self-test
	@rm -rf scripts/__pycache__
	@test -x scripts/check_tma_sass.py
	@echo "== TMA geometry regression gate (P1.2 remediation) =="
	@! grep -nE '\(COPIES\)\s*\*\s*\(kTileWidthBytes\s*/\s*kVectorBytes\)' src/memory/tma.cu
	@grep -Fq 'compute_stage_bytes' src/memory/tma.cu
	@grep -Fq 'compute_tile_height' src/memory/tma.cu
	@grep -Fq 'static_assert(geometry_table_is_correct()' src/memory/tma.cu
	@grep -Fq 'compute_tile_height(compute_stage_bytes(1)) == 8' src/memory/tma.cu
	@grep -Fq 'compute_tile_height(compute_stage_bytes(2)) == 16' src/memory/tma.cu
	@grep -Fq 'compute_tile_height(compute_stage_bytes(4)) == 32' src/memory/tma.cu
	@grep -Fq 'compute_tile_height(compute_stage_bytes(8)) == 64' src/memory/tma.cu
	@grep -Fq 'compute_tile_height(compute_stage_bytes(16)) == 128' src/memory/tma.cu
	@echo "== TMA mbarrier invalidation present, source and SASS checker (P1.2 remediation) =="
	@grep -Fq 'tma_invalidate_barrier' src/memory/tma.cu
	@grep -Fq 'mbarrier.inval.shared.b64' src/memory/tma.cu
	@grep -Fq 'SYNCS.CCTL.IV' scripts/check_tma_sass.py
	@echo "== documentation reports P1.2 as implemented, not unimplemented (P1.2 remediation) =="
	@! grep -rnF 'has not been started' README.md src/memory/README.md
	@! grep -rnF 'no TMA code exists yet' README.md src/memory/README.md
	@! grep -nF 'P1.2 and experiments' README.md
	@echo "== P1.3 required files present, executable, and syntactically valid =="
	@test -x $(EXP01_RUNNER)
	@test -x $(EXP01_AGGREGATOR)
	bash -n $(EXP01_RUNNER)
	python3 -m py_compile $(EXP01_AGGREGATOR)
	@rm -rf scripts/__pycache__
	@echo "== P1.3 GPU-free synthetic tests (self-test) =="
	python3 $(EXP01_AGGREGATOR) --self-test
	@echo "== P1.3 exact 18-way plan validation =="
	@test "$$(python3 $(EXP01_AGGREGATOR) plan --format lines | wc -l | tr -d ' ')" -eq 18
	@echo "== P1.3 forbidden patterns absent from the new scripts =="
	@pat='--gpus[ =]+all|NVIDIA_VISIBLE_DEVICES=all|--privileged|--pid[ =]+host|docker\.sock|--cap-add|SYS_ADMIN|set -x'; \
	pat="$$pat|\bs""udo\b|\$$\(np""roc\)|\bncu\b|nvidia-smi[^|]*(-pm|--persistence-mode|-lgc|--lock-gpu-clocks|-pl|--power-limit)"; \
	! grep -nE -- "$$pat" $(EXP01_RUNNER) $(EXP01_AGGREGATOR)
	@echo "== P1.3 runner uses the audited launcher and never selects a GPU automatically =="
	@grep -Fq 'run_container.sh' $(EXP01_RUNNER)
	@grep -Fq 'BLACKWELL_GPU_INDEX' $(EXP01_RUNNER)
	@echo "== P1.3 runner records progress after every validated case =="
	@grep -Fq 'write_manifest_progress' $(EXP01_RUNNER)
	@grep -Fq 'configurations_completed=$$((p_index + 1))' $(EXP01_RUNNER)
	@grep -Fq 'samples_completed=$$((configurations_completed * REPETITIONS))' $(EXP01_RUNNER)
	@echo "== P1.3 raw campaign output is git-ignored =="
	@grep -Fq 'results/raw/' .gitignore
	@echo "== truthful P1.1-P1.4 status assertions =="
	@grep -Fq 'P1.1 | Standalone LDGSTS baseline | YES | YES | YES |' PLAN.md
	@grep -Fq 'P1.2 | Equivalent TMA path | YES | YES | YES |' PLAN.md
	@grep -Fq 'P1.3 | Joint sweep (≤18 configurations) | YES | YES | YES |' PLAN.md
	@grep -Fq 'P1.4 | Profiling, validation, analysis, pilot | YES | YES | YES |' PLAN.md
	@grep -Fq 'The Phase 2 gate has passed' PLAN.md
	@! grep -F 'P1.3, P1.4, and experiments 2' README.md
	@! grep -F 'P1.3 (the joint LDGSTS/TMA sweep) has not started' README.md
	@! grep -F 'P1.3 (the joint sweep) and P1.4' src/memory/README.md
	@! grep -rnF 'P1.2 is implemented but unaudited' README.md src/memory/README.md
	@! grep -nF 'Status: implemented, pending audit and GB300 verification (see' src/memory/README.md
	@! grep -nF 'not a comparison against LDGSTS (P1.3 is the' src/memory/README.md
	@grep -Fq 'remediation completed' README.md
	@grep -Fq 'remediation completed' src/memory/README.md
	@echo "== P1.4 required files present, executable, and syntactically valid =="
	@test -x $(EXP01_P14_RUNNER)
	@test -x $(EXP01_P14_ANALYZER)
	@test -x $(EXP01_P14_SAFE_CAPTURE)
	@test -x $(EXP01_P14_NCU_BRIDGE)
	bash -n $(EXP01_P14_RUNNER)
	python3 -m py_compile $(EXP01_P14_ANALYZER)
	python3 -m py_compile $(EXP01_P14_SAFE_CAPTURE)
	python3 -m py_compile $(EXP01_P14_NCU_BRIDGE)
	@rm -rf scripts/__pycache__
	@echo "== P1.4 GPU-free synthetic/adversarial tests (self-test) =="
	python3 $(EXP01_P14_ANALYZER) --self-test
	python3 $(EXP01_P14_SAFE_CAPTURE) --self-test
	python3 $(EXP01_P14_NCU_BRIDGE) --self-test
	$(EXP01_P14_RUNNER) --self-test
	@echo "== P1.4 exact plan validation (18-way P1.3 pilot, six-way P1.4 NCU) =="
	@test "$$(python3 $(EXP01_AGGREGATOR) plan --format lines | wc -l | tr -d ' ')" -eq 18
	@test "$$(python3 $(EXP01_P14_ANALYZER) plan --format lines | wc -l | tr -d ' ')" -eq 6
	@echo "== P1.4 forbidden patterns absent (ncu itself is expected/required in P1.4) =="
	@pat='--gpus[ =]+all|NVIDIA_VISIBLE_DEVICES=all|--privileged|--pid[ =]+host|docker\.sock|--cap-add|SYS_ADMIN|set -x'; \
	pat="$$pat|\bs""udo\b|\$$\(np""roc\)|--force-overwrite|--set[ =]+full"; \
	pat="$$pat|clock-control[ =]+base|clock-control[ =]+boost|clock-control[ =]+force-boost"; \
	pat="$$pat|nvidia-smi[^|]*(-pm|--persistence-mode|-lgc|--lock-gpu-clocks|-pl|--power-limit)"; \
	! grep -nE -- "$$pat" $(EXP01_P14_RUNNER) $(EXP01_P14_ANALYZER) $(EXP01_P14_SAFE_CAPTURE) $(EXP01_P14_NCU_BRIDGE)
	@echo "== P1.4 runner uses the audited launcher/P1.3 runner and never selects a GPU automatically =="
	@grep -Fq 'run_container.sh' $(EXP01_P14_RUNNER)
	@grep -Fq 'BLACKWELL_GPU_INDEX' $(EXP01_P14_RUNNER)
	@grep -Fq '$(EXP01_RUNNER)' $(EXP01_P14_RUNNER)
	@grep -Fq -- '--working-set-mib 512' $(EXP01_P14_RUNNER)
	@grep -Fq -- '--passes 32' $(EXP01_P14_RUNNER)
	@grep -Fq -- '--warmup-ms 2000' $(EXP01_P14_RUNNER)
	@grep -Fq -- '--repetitions 30' $(EXP01_P14_RUNNER)
	@echo "== P2.1 required files present, executable, and syntactically valid =="
	@test -x $(COMPUTE_UMMA_1SM_CHECKER)
	python3 -m py_compile $(COMPUTE_UMMA_1SM_CHECKER)
	@rm -rf scripts/__pycache__
	@echo "== P2.1 SASS checker GPU-free synthetic self-test =="
	python3 $(COMPUTE_UMMA_1SM_CHECKER) --self-test
	@echo "== P2.1 source declares exactly twelve specializations =="
	@test "$$(grep -oE 'UMMA_1SM_DEFINE_KERNEL\([0-9]+, [0-9]+\)' $(COMPUTE_UMMA_1SM_SRC) | wc -l | tr -d ' ')" -eq 12
	@test "$$(grep -oE 'UMMA_1SM_SPEC_ENTRY\([0-9]+, [0-9]+\)' $(COMPUTE_UMMA_1SM_SRC) | wc -l | tr -d ' ')" -eq 12
	@echo "== P2.1 source uses the frozen tcgen05.mma kind::f16 cta_group::1 contract =="
	@grep -Fq 'tcgen05.mma.cta_group::1.kind::f16' $(COMPUTE_UMMA_1SM_SRC)
	@grep -Fq 'tcgen05.wait::ld.sync.aligned' $(COMPUTE_UMMA_1SM_SRC)
	@grep -Fq 'tcgen05.fence::after_thread_sync' $(COMPUTE_UMMA_1SM_SRC)
	@grep -Fq 'tcgen05.alloc.cta_group::1' $(COMPUTE_UMMA_1SM_SRC)
	@grep -Fq 'tcgen05.dealloc.cta_group::1' $(COMPUTE_UMMA_1SM_SRC)
	@grep -Fq 'tcgen05.relinquish_alloc_permit.cta_group::1' $(COMPUTE_UMMA_1SM_SRC)
	@grep -Fq 'tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64' $(COMPUTE_UMMA_1SM_SRC)
	@echo "== P2.1 forbidden patterns absent (no 2-SM, cluster, sparse, block-scaled, or"
	@echo "   non-kind::f16 form; no P0/P1-style forbidden shell patterns) =="
	@echo "   (checked against code with '//' comments stripped, so a comment explaining"
	@echo "   why e.g. cta_group::2 is absent cannot itself trip these checks)"
	@! sed 's#//.*##' $(COMPUTE_UMMA_1SM_SRC) | grep -nE 'cta_group::2|__cluster_dims__|multicast|block_scale|\.sp\b'
	@! sed 's#//.*##' $(COMPUTE_UMMA_1SM_SRC) | grep -nE '\.kind::(tf32|f8f6f4|mxf8f6f4|mxf4|mxf4nvf4|i8)\b'
	@pat='--gpus[ =]+all|NVIDIA_VISIBLE_DEVICES=all|--privileged|--pid[ =]+host|docker\.sock|--cap-add|SYS_ADMIN|set -x'; \
	pat="$$pat|\bs""udo\b|\$$\(np""roc\)"; \
	pat="$$pat|nvidia-smi[^|]*(-pm|--persistence-mode|-lgc|--lock-gpu-clocks|-pl|--power-limit)"; \
	! sed 's#//.*##' $(COMPUTE_UMMA_1SM_SRC) | grep -nE -- "$$pat" && \
	! grep -nE -- "$$pat" $(COMPUTE_UMMA_1SM_CHECKER)
	@echo "== P2.1 Makefile target derives its arch/code flags from the pinned CUDA_ARCH =="
	@grep -Fq 'COMPUTE_UMMA_1SM_ARCH_FLAGS := -arch=compute_$$(patsubst sm_%,%,$$(CUDA_ARCH)) -code=$$(CUDA_ARCH)' Makefile
	@echo "== P2.1 documentation reports audited and GB300-verified closure honestly =="
	@grep -Fq 'P2.1 | 1-SM UMMA | YES | YES | YES |' PLAN.md
	@! grep -nF 'publishable P2.1 result' README.md PLAN.md $(COMPUTE_UMMA_1SM_PROTOCOL)
	@grep -Fq '* Independent audit: **passed**.' $(COMPUTE_UMMA_1SM_PROTOCOL)
	@grep -Fq '* GB300 verification: **passed**.' $(COMPUTE_UMMA_1SM_PROTOCOL)
	@grep -Fq '* Publishable result: **none**.' $(COMPUTE_UMMA_1SM_PROTOCOL)
	@echo "== P2.2 required files present, executable, and syntactically valid =="
	@test -x $(COMPUTE_UMMA_2SM_CHECKER)
	python3 -m py_compile $(COMPUTE_UMMA_2SM_CHECKER)
	@rm -rf scripts/__pycache__
	@echo "== P2.2 SASS checker GPU-free synthetic self-test =="
	python3 $(COMPUTE_UMMA_2SM_CHECKER) --self-test
	@echo "== P2.2 source declares exactly the twelve-configuration matrix =="
	@test "$$(grep -oE 'UMMA_2SM_DEFINE_KERNEL\([0-9]+, [0-9]+\)' $(COMPUTE_UMMA_2SM_SRC) | wc -l | tr -d ' ')" -eq 12
	@test "$$(grep -oE 'UMMA_2SM_SPEC_ENTRY\([0-9]+, [0-9]+\)' $(COMPUTE_UMMA_2SM_SRC) | wc -l | tr -d ' ')" -eq 12
	@for n in 64 128 256; do \
		for d in 4 16 64 256; do \
			grep -Fq "UMMA_2SM_DEFINE_KERNEL($$n, $$d)" $(COMPUTE_UMMA_2SM_SRC) \
				|| { echo "check-static: MISSING UMMA_2SM_DEFINE_KERNEL($$n, $$d)"; exit 1; }; \
			grep -Fq "UMMA_2SM_SPEC_ENTRY($$n, $$d)" $(COMPUTE_UMMA_2SM_SRC) \
				|| { echo "check-static: MISSING UMMA_2SM_SPEC_ENTRY($$n, $$d)"; exit 1; }; \
		done; \
	done
	@echo "== P2.2 source uses the frozen tcgen05 cta_group::2 CTA-pair contract =="
	@grep -Fq 'tcgen05.mma.cta_group::2.kind::f16' $(COMPUTE_UMMA_2SM_SRC)
	@grep -Fq 'tcgen05.wait::ld.sync.aligned' $(COMPUTE_UMMA_2SM_SRC)
	@grep -Fq 'tcgen05.fence::after_thread_sync' $(COMPUTE_UMMA_2SM_SRC)
	@grep -Fq 'tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32' $(COMPUTE_UMMA_2SM_SRC)
	@grep -Fq 'tcgen05.dealloc.cta_group::2.sync.aligned.b32' $(COMPUTE_UMMA_2SM_SRC)
	@grep -Fq 'tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned' $(COMPUTE_UMMA_2SM_SRC)
	@grep -Fq 'tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64' $(COMPUTE_UMMA_2SM_SRC)
	@grep -Fq '__cluster_dims__(2, 1, 1)' $(COMPUTE_UMMA_2SM_SRC)
	@grep -Fq 'get_sreg_cluster_ctarank' $(COMPUTE_UMMA_2SM_SRC)
	@grep -Fq 'get_sreg_cluster_nctarank' $(COMPUTE_UMMA_2SM_SRC)
	@grep -Fq 'barrier_cluster_arrive' $(COMPUTE_UMMA_2SM_SRC)
	@grep -Fq 'barrier_cluster_wait' $(COMPUTE_UMMA_2SM_SRC)
	@grep -Fq '0x0003' $(COMPUTE_UMMA_2SM_SRC)
	@echo "== P2.2 forbidden patterns absent (no 1-SM cta_group::1, sparse, block-scaled, or"
	@echo "   non-kind::f16 form; no P0/P1-style forbidden shell patterns) =="
	@echo "   (checked against code with '//' comments stripped, so a comment explaining"
	@echo "   why e.g. cta_group::1 is absent cannot itself trip these checks)"
	@! sed 's#//.*##' $(COMPUTE_UMMA_2SM_SRC) | grep -nE 'cta_group::1\b'
	@! sed 's#//.*##' $(COMPUTE_UMMA_2SM_SRC) | grep -nE '\.kind::(tf32|f8f6f4|mxf8f6f4|mxf4nvf4|mxf4|i8)\b'
	@! sed 's#//.*##' $(COMPUTE_UMMA_2SM_SRC) | grep -nE '\.sp\b|block_scale'
	@pat='--gpus[ =]+all|NVIDIA_VISIBLE_DEVICES=all|--privileged|--pid[ =]+host|docker\.sock|--cap-add|SYS_ADMIN|set -x'; \
	pat="$$pat|\bs""udo\b|\$$\(np""roc\)"; \
	pat="$$pat|nvidia-smi[^|]*(-pm|--persistence-mode|-lgc|--lock-gpu-clocks|-pl|--power-limit)"; \
	! sed 's#//.*##' $(COMPUTE_UMMA_2SM_SRC) | grep -nE -- "$$pat" && \
	! grep -nE -- "$$pat" $(COMPUTE_UMMA_2SM_CHECKER)
	@echo "== P2.2 Makefile target derives its arch/code flags from the pinned CUDA_ARCH =="
	@grep -Fq 'COMPUTE_UMMA_2SM_ARCH_FLAGS := -arch=compute_$$(patsubst sm_%,%,$$(CUDA_ARCH)) -code=$$(CUDA_ARCH)' Makefile
	@echo "== P2.2 documentation reports audited and GB300-verified closure honestly =="
	@grep -Fq 'P2.2 | 2-SM UMMA | YES | YES | YES |' PLAN.md
	@grep -Fq 'independently audited: YES; verified on GB300: YES; publishable results:' README.md
	@grep -Fq '* P2.2: **implemented and closed**' $(COMPUTE_UMMA_2SM_PROTOCOL)
	@! grep -nF 'publishable P2.2 result' README.md PLAN.md $(COMPUTE_UMMA_2SM_PROTOCOL)
	@grep -Fq '* Independent audit: **passed**.' $(COMPUTE_UMMA_2SM_PROTOCOL)
	@grep -Fq '* GB300 verification: **passed**.' $(COMPUTE_UMMA_2SM_PROTOCOL)
	@grep -Fq '* Publishable result: **none**.' $(COMPUTE_UMMA_2SM_PROTOCOL)
	@grep -Fq 'P2.2 = YES / YES / YES' $(COMPUTE_UMMA_2SM_PROTOCOL)
	@! grep -nF 'P2.2 = YES / NO / NO' PLAN.md README.md $(COMPUTE_UMMA_1SM_PROTOCOL) $(COMPUTE_UMMA_2SM_PROTOCOL)
	@! grep -nF 'Independent audit: **pending**.' PLAN.md README.md $(COMPUTE_UMMA_1SM_PROTOCOL) $(COMPUTE_UMMA_2SM_PROTOCOL)
	@! grep -nF 'GB300 verification: **pending**.' PLAN.md README.md $(COMPUTE_UMMA_1SM_PROTOCOL) $(COMPUTE_UMMA_2SM_PROTOCOL)
	@echo "== P2.2 repair (audit round 1): stale 'unimplemented' phrases cannot silently return =="
	@grep -Fq 'P2.2 implemented; independently audited; GB300 verification passed.' Makefile
	@! grep -F "P2.2's unimplemented scope" $(COMPUTE_UMMA_1SM_PROTOCOL)
	@! grep -F 'P2.2 (unimplemented, 12 configs)' $(COMPUTE_UMMA_1SM_PROTOCOL)
	@! grep -F 'P2.2 (2-SM), P2.3 (joint sweep), and P2.4 (profiling/ceiling) remain' $(COMPUTE_UMMA_1SM_PROTOCOL)
	@! grep -F '* P2.2, P2.3, P2.4: **not implemented**.' $(COMPUTE_UMMA_1SM_PROTOCOL)
	@echo "== P2.2 repair (audit round 1): mbarrier-init fence cannot silently regress =="
	@grep -Fq 'fence_mbarrier_init_release_cluster' $(COMPUTE_UMMA_2SM_SRC)
	@grep -Fq 'fence.mbarrier_init.release.cluster' $(COMPUTE_UMMA_2SM_SRC)
	@grep -Fq 'fence_mbarrier_init_release_cluster' $(COMPUTE_UMMA_2SM_PROTOCOL)
	@grep -Fq 'fence.mbarrier_init.release.cluster' $(COMPUTE_UMMA_2SM_PROTOCOL)
	@echo "== P2.2 repair (audit round 1): per-phase CTA-pair handshake cannot silently regress =="
	@grep -Fq 'is_leader && cta_rank == 0 && timing_mode == TimingMode::kTimed' $(COMPUTE_UMMA_2SM_SRC)
	@test "$$(grep -coE 'if \(is_leader && cta_rank == 0 && timing_mode == TimingMode::kTimed\)' $(COMPUTE_UMMA_2SM_SRC))" -eq 2
	@echo "== P2.3 required files present, executable, and syntactically valid =="
	@test -x $(EXP02_RUNNER)
	@test -x $(EXP02_AGGREGATOR)
	bash -n $(EXP02_RUNNER)
	python3 -m py_compile $(EXP02_AGGREGATOR)
	@rm -rf scripts/__pycache__
	@echo "== P2.3 GPU-free synthetic tests (self-test) =="
	python3 $(EXP02_AGGREGATOR) --self-test
	@echo "== P2.3 exact 24-way plan validation =="
	@test "$$(python3 $(EXP02_AGGREGATOR) plan --format lines | wc -l | tr -d ' ')" -eq 24
	@echo "== P2.3 forbidden patterns absent from the new scripts =="
	@pat='--gpus[ =]+all|NVIDIA_VISIBLE_DEVICES=all|--privileged|--pid[ =]+host|docker\.sock|--cap-add|SYS_ADMIN|set -x'; \
	pat="$$pat|\bs""udo\b|\$$\(np""roc\)|\bncu\b|nvidia-smi[^|]*(-pm|--persistence-mode|-lgc|--lock-gpu-clocks|-pl|--power-limit)"; \
	! grep -nE -- "$$pat" $(EXP02_RUNNER) $(EXP02_AGGREGATOR)
	@echo "== P2.3 runner uses the audited launcher and never selects a GPU automatically =="
	@grep -Fq 'run_container.sh' $(EXP02_RUNNER)
	@grep -Fq 'BLACKWELL_GPU_INDEX' $(EXP02_RUNNER)
	@echo "== P2.3 runner reuses the audited P2.1/P2.2 binaries unmodified (no new kernel) =="
	@grep -Fq 'build/compute/umma_1sm' $(EXP02_RUNNER)
	@grep -Fq 'build/compute/umma_2sm' $(EXP02_RUNNER)
	@grep -Fq 'compute-umma-1sm-sass compute-umma-2sm-sass' $(EXP02_RUNNER)
	@echo "== P2.3 runner records progress after every validated case =="
	@grep -Fq 'write_manifest_progress' $(EXP02_RUNNER)
	@grep -Fq 'configurations_completed=$$((p_index + 1))' $(EXP02_RUNNER)
	@grep -Fq 'samples_completed=$$((configurations_completed * REPETITIONS))' $(EXP02_RUNNER)
	@echo "== P2.3 raw campaign output is git-ignored (shared results/raw/ rule) =="
	@grep -Fq 'results/raw/' .gitignore
	@echo "== truthful P2.3 status assertions =="
	@grep -Fq 'P2.3 | Sweep (≤24 configurations) | YES | YES | YES |' PLAN.md
	@grep -Fq 'P2.3 = YES / YES / YES' $(EXP02_PROTOCOL)
	@grep -Fq 'Phase 2 is closed' PLAN.md
	@grep -Fq 'Phase 2 is **closed**' $(EXP02_PROTOCOL)
	@! grep -rnF 'P2.3 (joint sweep) has not been started' PLAN.md README.md
	@! grep -F 'No runner, no campaign, no sweep script exists.' $(COMPUTE_UMMA_1SM_PROTOCOL) $(COMPUTE_UMMA_2SM_PROTOCOL)
	@echo "== P2.4 required files present, executable, and syntactically valid =="
	@test -x $(EXP02_P24_RUNNER)
	@test -x $(EXP02_P24_ANALYZER)
	@test -x $(EXP02_P24_SAFE_CAPTURE)
	@test -x $(EXP02_P24_NCU_BRIDGE)
	bash -n $(EXP02_P24_RUNNER)
	python3 -m py_compile $(EXP02_P24_ANALYZER) $(EXP02_P24_SAFE_CAPTURE) $(EXP02_P24_NCU_BRIDGE)
	@rm -rf scripts/__pycache__
	@echo "== P2.4 GPU-free synthetic/adversarial tests (self-test) =="
	python3 $(EXP02_P24_ANALYZER) --self-test
	python3 $(EXP02_P24_SAFE_CAPTURE) --self-test
	python3 $(EXP02_P24_NCU_BRIDGE) --self-test
	$(EXP02_P24_RUNNER) --self-test
	@echo "== P2.4 exact plan validation (24-way P2.3 pilot, 24-way P2.4 profile) =="
	@test "$$(python3 $(EXP02_AGGREGATOR) plan --format lines | wc -l | tr -d ' ')" -eq 24
	@test "$$(python3 $(EXP02_P24_ANALYZER) plan --format lines | wc -l | tr -d ' ')" -eq 24
	@echo "== P2.4 forbidden patterns absent (ncu itself is expected/required in P2.4) =="
	@pat='--gpus[ =]+all|NVIDIA_VISIBLE_DEVICES=all|--privileged|--pid[ =]+host|docker\.sock|--cap-add|SYS_ADMIN|set -x'; \
	pat="$$pat|\bs""udo\b|\$$\(np""roc\)|--force-overwrite|--set[ =]+full"; \
	pat="$$pat|clock-control[ =]+base|clock-control[ =]+boost|clock-control[ =]+force-boost"; \
	pat="$$pat|nvidia-smi[^|]*(-pm|--persistence-mode|-lgc|--lock-gpu-clocks|-pl|--power-limit)"; \
	! grep -nE -- "$$pat" $(EXP02_P24_RUNNER) $(EXP02_P24_ANALYZER) $(EXP02_P24_SAFE_CAPTURE) $(EXP02_P24_NCU_BRIDGE)
	@echo "== P2.4 runner uses the audited launcher/P2.3 runner and never selects a GPU automatically =="
	@grep -Fq 'run_container.sh' $(EXP02_P24_RUNNER)
	@grep -Fq 'BLACKWELL_GPU_INDEX' $(EXP02_P24_RUNNER)
	@grep -Fq '$(EXP02_RUNNER)' $(EXP02_P24_RUNNER)
	@grep -Fq -- '--iterations 1000' $(EXP02_P24_RUNNER)
	@grep -Fq -- '--warmup-iterations 10' $(EXP02_P24_RUNNER)
	@grep -Fq -- '--repetitions 30' $(EXP02_P24_RUNNER)
	@echo "== P2.4 profile invocations use the frozen warmup-iterations 0 / repetitions 1 contract =="
	@grep -Fq -- '--warmup-iterations 0 --repetitions 1' $(EXP02_P24_RUNNER)
	@echo "== P2.4 profile invocation uses an exact kernel-name filter and the second (timed) launch =="
	@grep -Fq -- '--launch-skip' $(EXP02_P24_NCU_BRIDGE)
	@grep -Fq -- '--kernel-name-base' $(EXP02_P24_NCU_BRIDGE)
	@echo "== P2.4 repair: the P2.4 Make gate actually executes the P2.1/P2.2/P2.3 GPU-free gates =="
	@grep -Eq '^compute-umma-p24-check: compute-umma-1sm-check compute-umma-2sm-check compute-umma-sweep-check$$' Makefile
	@echo "== P2.4 raw campaign output is git-ignored (shared results/raw/ rule) =="
	@grep -Fq 'results/raw/' .gitignore
	@echo "== truthful P2.4 status assertions =="
	@grep -Fq 'P2.4 | Profiling and empirical ceiling | YES | YES | YES |' PLAN.md
	@grep -Fq 'P2.4 | Profiling and empirical ceiling | YES | YES | YES |' $(EXP02_P24_PROTOCOL)
	@grep -Fq '* P2.4: **implemented, independently audited, and verified on GB300**' $(COMPUTE_UMMA_1SM_PROTOCOL)
	@grep -Fq '* P2.4: **implemented, independently audited, and verified on GB300**' $(COMPUTE_UMMA_2SM_PROTOCOL)
	@grep -Fq 'Phase 2 is closed' PLAN.md
	@grep -Fq 'Phase 2: CLOSED' README.md
	@! grep -rnF 'P2.4 remains entirely unimplemented' PLAN.md README.md $(COMPUTE_UMMA_1SM_PROTOCOL) $(COMPUTE_UMMA_2SM_PROTOCOL)
	@! grep -rnF 'P2.4 (profiling and empirical ceiling) remains entirely **not implemented**' README.md
	@grep -Fq 'P2.4 is also `YES / YES / YES`' $(EXP02_PROTOCOL)
	@echo "== P3.1 protocol present; no NVIDIA GEMM source is vendored into this repository =="
	@test -f $(GEMM_P31_PROTOCOL)
	@! grep -nE '^(import|from|def|class) ' $(GEMM_P31_PROTOCOL)
	@! test -e src/gemm/dense_gemm.py
	@echo "== P3.1 resolves the official example from the pinned VERSIONS.env values =="
	@grep -Fq 'GEMM_P31_EXAMPLE := /opt/cutlass/$$(CUTEDSL_P31_EXAMPLE_PATH)' Makefile
	@grep -Fq '$(CUTEDSL_P31_EXAMPLE_GIT_BLOB)' $(GEMM_P31_PROTOCOL)
	@grep -Fq '$(CUTEDSL_P31_EXAMPLE_SHA256)' $(GEMM_P31_PROTOCOL)
	@grep -Fq '$(CUTLASS_COMMIT)' $(GEMM_P31_PROTOCOL)
	@echo "== P3.1 frozen functional configuration cannot silently change =="
	@grep -Fq -- '--mnkl 256,256,512,1' Makefile
	@grep -Fq -- '--ab_dtype BFloat16 --c_dtype Float32 --acc_dtype Float32' Makefile
	@grep -Fq -- '--a_major k --b_major k --c_major n' Makefile
	@grep -Fq -- '--mma_tiler_mn 128,128 --cluster_shape_mn 1,1' Makefile
	@grep -Fq -- '--use_tma_store' Makefile
	@grep -Fq -- '--warmup_iterations 0 --iterations 1' Makefile
	@echo "== P3.1 never adds 2-CTA instructions, skips reference checking, uses cold L2,"
	@echo "   or executes the persistent example (those are later units) =="
	@# These option spellings must not appear anywhere in the Makefile at all.
	@pat='--use_2cta'; pat="$$pat""_instrs|--skip_ref""_check|--use_cold""_l2"; \
	! grep -nE -- "$$pat" Makefile
	@# The persistent example, however, IS legitimately referenced now that P3.4
	@# exists, so the invariant is scoped to P3.1 itself: P3.1's pinned path and
	@# its derived Make variable must never name the persistent file. A
	@# whole-file ban would have failed the moment a later unit landed.
	@# (These three patterns are anchored at column 0, so none of them can match
	@#  the tab-indented recipe lines that contain them.)
	@! grep -nE '^CUTEDSL_P31_EXAMPLE_PATH=.*persistent' PHASE3_VERSIONS.env
	@! grep -nE '^GEMM_P31_EXAMPLE[[:space:]]*:=.*persistent' Makefile
	@! grep -nE '^gemm-cutedsl-p31-(check|smoke):.*persistent' Makefile
	@grep -Eq '^CUTEDSL_P31_EXAMPLE_PATH=.*/dense_gemm\.py$$' PHASE3_VERSIONS.env
	@echo "== P3.1 smoke validates BLACKWELL_GPU_INDEX before any Docker prerequisite =="
	@grep -Eq '^gemm-cutedsl-p31-smoke:$$' Makefile
	@grep -Fq 'scripts/run_container.sh' Makefile
	@echo "== truthful P3.1 status assertions =="
	@grep -Fq 'P3.1 | Pinned official CuTe DSL example | YES | YES | YES |' PLAN.md
	@# P3.4 and P3.5 are later units that progress on their own, so P3.1's own
	@# gate only requires that each still has a status row - never that it is
	@# pinned to a particular value, which would fail the moment that unit is
	@# truthfully implemented, audited, or verified. Each unit's own section
	@# below asserts its own truthful status.
	@grep -Fq 'P3.4 | Three execution variants |' PLAN.md
	@grep -Fq 'P3.5 | Five shapes and comparison |' PLAN.md
	@grep -Fq 'P3.1 = YES / YES / YES' $(GEMM_P31_PROTOCOL)
	@grep -Fq 'P3.1 produces no experimental result' $(GEMM_P31_PROTOCOL)
	@grep -Fq 'non-persistent' $(GEMM_P31_PROTOCOL)
	@grep -Fq 'P3.1 (pinned official CuTe DSL example)' README.md
	@! grep -nF 'P3.1 | Pinned official CuTe DSL example | YES | NO | NO |' PLAN.md
	@echo "== P3.2 files present, executable, and still vendoring no NVIDIA GEMM source =="
	@test -f $(GEMM_P32_WRAPPER)
	@test -f $(GEMM_P32_CHECKER)
	@test -f $(GEMM_P32_PROTOCOL)
	@test -x $(GEMM_P32_WRAPPER)
	@test -x $(GEMM_P32_CHECKER)
	@! grep -nE '^(import|from|def|class) ' $(GEMM_P32_PROTOCOL)
	@echo "== P3.2 python syntax, GPU-free self-tests, and the full contract check =="
	python3 -m py_compile $(GEMM_P32_WRAPPER) $(GEMM_P32_CHECKER)
	python3 $(GEMM_P32_WRAPPER) --self-test
	python3 $(GEMM_P32_CHECKER) --self-test
	python3 $(GEMM_P32_CHECKER) .
	@rm -rf src/gemm/__pycache__ scripts/__pycache__
	@echo "== P3.2 frozen one-shape configuration cannot silently change =="
	@grep -Eq '^FROZEN_M = 4096$$' $(GEMM_P32_WRAPPER)
	@grep -Eq '^FROZEN_N = 4096$$' $(GEMM_P32_WRAPPER)
	@grep -Eq '^FROZEN_K = 4096$$' $(GEMM_P32_WRAPPER)
	@grep -Eq '^FROZEN_L = 1$$' $(GEMM_P32_WRAPPER)
	@grep -Eq '^FROZEN_AB_DTYPE = "BFloat16"$$' $(GEMM_P32_WRAPPER)
	@grep -Eq '^FROZEN_ACC_DTYPE = "Float32"$$' $(GEMM_P32_WRAPPER)
	@grep -Eq '^FROZEN_C_DTYPE = "Float32"$$' $(GEMM_P32_WRAPPER)
	@grep -Eq '^FROZEN_A_MAJOR = "k"$$' $(GEMM_P32_WRAPPER)
	@grep -Eq '^FROZEN_B_MAJOR = "k"$$' $(GEMM_P32_WRAPPER)
	@grep -Eq '^FROZEN_C_MAJOR = "n"$$' $(GEMM_P32_WRAPPER)
	@grep -Eq '^FROZEN_MMA_TILER_MN = \(128, 128\)$$' $(GEMM_P32_WRAPPER)
	@grep -Eq '^FROZEN_CLUSTER_SHAPE_MN = \(1, 1\)$$' $(GEMM_P32_WRAPPER)
	@grep -Eq '^FROZEN_USE_2CTA_INSTRS = False$$' $(GEMM_P32_WRAPPER)
	@grep -Eq '^FROZEN_USE_TMA_STORE = True$$' $(GEMM_P32_WRAPPER)
	@grep -Eq '^FROZEN_SEED = 1111$$' $(GEMM_P32_WRAPPER)
	@grep -Eq '^SCHEMA_VERSION = "p32.v1"$$' $(GEMM_P32_WRAPPER)
	@grep -Eq '^RUN_KIND = "smoke"$$' $(GEMM_P32_WRAPPER)
	@grep -Eq '^VARIANT = "nonpersistent_1cta"$$' $(GEMM_P32_WRAPPER)
	@grep -Eq '^REFERENCE = "torch_cuda_fp32_ieee"$$' $(GEMM_P32_WRAPPER)
	@grep -Eq '^CACHE_MODE = "hot"$$' $(GEMM_P32_WRAPPER)
	@grep -Eq '^PUBLISHABLE = "false"$$' $(GEMM_P32_WRAPPER)
	@echo "== P3.2 exposes no shape/variant control and can never skip the reference check =="
	@pat='--mnkl|--ab'; \
	pat="$$pat""_dtype|--c_dtype|--acc_dtype|--a_major|--b_major|--c_major"; \
	pat="$$pat|--mma_tiler|--cluster_shape|--tolerance|--use_tma_store"; \
	pat="$$pat|--skip-ref""-check|--skip_ref""_check|--use_2cta""_instrs|--use_cold""_l2"; \
	pat="$$pat|--use-cold""-l2|--persistent|dense_gemm""_persistent"; \
	! grep -nE -- "$$pat" $(GEMM_P32_WRAPPER)
	@echo "   (the checker names those spellings on purpose, in order to ban them)"
	@echo "== P3.2 writes no result file and creates no campaign directory =="
	@! grep -nE 'results/raw|results/preflight' $(GEMM_P32_WRAPPER)
	@echo "   (the tokenized identifier ban for TFLOP/s, speedup, cuBLASLt, Nsight"
	@echo "    Compute, autotuning, and campaign trees lives in $(GEMM_P32_CHECKER))"
	@echo "== P3.2 adds no key to either version contract =="
	@! grep -nE '^CUTEDSL_P32_' PHASE3_VERSIONS.env VERSIONS.env
	@echo "== P3.2 reuses the audited launcher and never invokes Docker for GPU work =="
	@grep -Eq '^gemm-cutedsl-p32-smoke:$$' Makefile
	@grep -Fq 'scripts/run_container.sh' Makefile
	@echo "== P3.2 GPU-free gate actually executes the existing P3.1 gate =="
	@grep -Eq '^gemm-cutedsl-p32-check: gemm-cutedsl-p31-check$$' Makefile
	@echo "== P3.2 smoke runs exactly the frozen non-publishable iteration counts =="
	@grep -Fq -- '--warmup-iterations 2 \' Makefile
	@grep -Fq -- '--iterations 10' Makefile
	@echo "== truthful P3.2 status assertions =="
	@grep -Fq 'P3.2 | One-shape wrapper | YES | YES | YES |' PLAN.md
	@grep -Fq 'P3.2 = YES / YES / YES' $(GEMM_P32_PROTOCOL)
	@grep -Fq 'P3.2 creates no publishable performance result' $(GEMM_P32_PROTOCOL)
	@grep -Fq 'P3.2 (one-shape wrapper)' README.md
	@! grep -nF 'P3.2 | One-shape wrapper | NO | NO | NO |' PLAN.md
	@! grep -nF 'P3.2 | One-shape wrapper | YES | NO | NO |' PLAN.md
	@! grep -nF 'P3.2 | One-shape wrapper | YES | YES | NO |' PLAN.md
	@! grep -nF 'P3.2 | One-shape wrapper | YES | NO | YES |' PLAN.md
	@echo "== P3.3 files present, executable, and still vendoring no NVIDIA GEMM source =="
	@test -f $(GEMM_P33_WRAPPER)
	@test -f $(GEMM_P33_BRIDGE)
	@test -f $(GEMM_P33_CHECKER)
	@test -f $(GEMM_P33_PROTOCOL)
	@test -x $(GEMM_P33_WRAPPER)
	@test -x $(GEMM_P33_CHECKER)
	@! grep -nE '^(import|from|def|class) ' $(GEMM_P33_PROTOCOL)
	@echo "== P3.3 python syntax, GPU-free self-tests, and the full contract check =="
	python3 -m py_compile $(GEMM_P33_WRAPPER) $(GEMM_P33_CHECKER)
	python3 $(GEMM_P33_WRAPPER) --self-test
	python3 $(GEMM_P33_CHECKER) --self-test
	python3 $(GEMM_P33_CHECKER) .
	@rm -rf src/gemm/__pycache__ scripts/__pycache__
	@echo "== P3.3 frozen geometry and descriptor contract cannot silently change =="
	@grep -Eq '^FROZEN_M = 4096$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_N = 4096$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_K = 4096$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_L = 1$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_AB_DTYPE = "BFloat16"$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_ACC_DTYPE = "Float32"$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_C_DTYPE = "Float32"$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_A_MAJOR = "k"$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_B_MAJOR = "k"$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_C_MAJOR = "n"$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_LDA = FROZEN_K$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_LDB = FROZEN_K$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_LDC = FROZEN_N$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_LDD = FROZEN_N$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_TRANSA = "CUBLAS_OP_N"$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_TRANSB = "CUBLAS_OP_T"$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_ORDER = "CUBLASLT_ORDER_ROW"$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_AB_CUDA_TYPE = "CUDA_R_16BF"$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_CD_CUDA_TYPE = "CUDA_R_32F"$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_COMPUTE_TYPE = "CUBLAS_COMPUTE_32F"$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_SCALE_TYPE = "CUDA_R_32F"$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_POINTER_MODE = "CUBLASLT_POINTER_MODE_HOST"$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_EPILOGUE = "CUBLASLT_EPILOGUE_DEFAULT"$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_ALPHA = 1\.0$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_BETA = 0\.0$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_SEED = 1111$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^SCHEMA_VERSION = "p33.v1"$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^RUN_KIND = "smoke"$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^METHOD = "cublaslt"$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^VARIANT = "heuristic_first_supported"$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^REFERENCE = "torch_cuda_fp32_ieee"$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^CACHE_MODE = "hot"$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^PUBLISHABLE = "false"$$' $(GEMM_P33_WRAPPER)
	@echo "== P3.3 algorithm policy is fixed, never autotuned, never benchmarked =="
	@grep -Eq '^FROZEN_WORKSPACE_LIMIT_BYTES = 67108864$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_HEURISTIC_REQUESTED = 32$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^FROZEN_SEARCH_MODE = "CUBLASLT_SEARCH_BEST_FIT"$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^static const uint64_t P33_WORKSPACE_LIMIT_BYTES = 67108864ULL;$$' $(GEMM_P33_BRIDGE)
	@grep -Eq '^static const int P33_HEURISTIC_REQUESTED = 32;$$' $(GEMM_P33_BRIDGE)
	@echo "== P3.3 reuses no P3.2 schema field and never calls setup compilation =="
	@# Structural, not substring: the wrapper deliberately NAMES compile_time_ms
	@# and p32.v1 in prose and in negative self-test assertions that prove both
	@# are rejected. What must not exist is a schema entry or a schema version.
	@! grep -nE '^[[:space:]]+"compile_time_ms",$$' $(GEMM_P33_WRAPPER)
	@! grep -nE '^SCHEMA_VERSION = "p32\.v1"$$' $(GEMM_P33_WRAPPER)
	@grep -Eq '^[[:space:]]+"setup_time_ms",$$' $(GEMM_P33_WRAPPER)
	@echo "== the P3.3 measured path is cublasLtMatmul, with no fallback GEMM API =="
	@grep -Eq 'cublasLtMatmul[[:space:]]*\(' $(GEMM_P33_BRIDGE)
	@# Call sites, not bare names: both files deliberately NAME the forbidden
	@# entry points in prose that explains they are never used. The rigorous
	@# comment-stripped scan lives in $(GEMM_P33_CHECKER).
	@pat='cublasGemmEx|cublasGemmStridedBatchedEx|cublasGemmBatchedEx'; \
	pat="$$pat|cublasSgemm|cublasHgemm|cublasLtMatmulAlgoGetIds|cublasLtMatmulAlgoInit"; \
	! grep -nE -- "($$pat)[[:space:]]*\(" $(GEMM_P33_WRAPPER) $(GEMM_P33_BRIDGE)
	@echo "   (the checker names those spellings on purpose, in order to ban them)"
	@echo "== the P3.3 bridge owns no GEMM kernel and prints nothing =="
	@! grep -nE '__global__|__device__' $(GEMM_P33_BRIDGE)
	@! grep -nE '(^|[^a-zA-Z0-9_])(printf|puts|fputs)[[:space:]]*\(' $(GEMM_P33_BRIDGE)
	@! grep -nE 'std::(cout|cerr|clog)' $(GEMM_P33_BRIDGE)
	@! grep -nE 'cudaEventRecord|cudaEventElapsedTime|std::chrono' $(GEMM_P33_BRIDGE)
	@grep -Fq 'extern "C"' $(GEMM_P33_BRIDGE)
	@grep -Fq 'catch (...)' $(GEMM_P33_BRIDGE)
	@echo "== P3.3 exposes no descriptor/policy control and can never skip the reference check =="
	@pat='--mnkl|--shape|--lda|--ldb|--ldc|--ldd|--transa|--transb|--alpha|--beta'; \
	pat="$$pat|--epilogue|--workspace|--heuristic|--algo|--tile|--stages|--split"; \
	pat="$$pat|--cluster|--order|--autotune|--search|--cache-mode|--publish"; \
	pat="$$pat|--skip-ref""-check|--skip_ref""_check|--use_cold""_l2|--persistent"; \
	! grep -nE -- "$$pat" $(GEMM_P33_WRAPPER)
	@echo "== P3.3 writes no result file and creates no campaign directory =="
	@! grep -nE 'results/raw|results/preflight' $(GEMM_P33_WRAPPER)
	@echo "== P3.3 adds no key to either version contract =="
	@! grep -nE '^(CUBLAS|CUBLASLT|P33_)' PHASE3_VERSIONS.env VERSIONS.env
	@echo "== P3.3 reuses the audited launcher and never invokes Docker for GPU work =="
	@grep -Eq '^gemm-cublaslt-p33-smoke:$$' Makefile
	@grep -Fq 'scripts/run_container.sh' Makefile
	@echo "== P3.3 GPU-free gate actually executes the existing P3.2 gate =="
	@grep -Eq '^gemm-cublaslt-p33-check: gemm-cutedsl-p32-check$$' Makefile
	@echo "== P3.3 smoke runs exactly the frozen non-publishable iteration counts =="
	@grep -Fq -- '--warmup-iterations 2 \' Makefile
	@grep -Fq -- '--iterations 10' Makefile
	@echo "== truthful P3.3 status assertions =="
	@grep -Fq 'P3.3 | cuBLASLt baseline | YES | YES | YES |' PLAN.md
	@grep -Fq 'P3.3 = YES / YES / YES' $(GEMM_P33_PROTOCOL)
	@grep -Fq 'P3.3 creates no publishable performance result' $(GEMM_P33_PROTOCOL)
	@grep -Fq 'P3.3: CLOSED' README.md
	@! grep -nF 'P3.3 | cuBLASLt baseline | NO | NO | NO |' PLAN.md
	@! grep -nF 'P3.3 | cuBLASLt baseline | YES | NO | NO |' PLAN.md
	@! grep -nF 'P3.3 | cuBLASLt baseline | YES | NO | YES |' PLAN.md
	@! grep -nF 'P3.3 | cuBLASLt baseline | YES | YES | NO |' PLAN.md
	@echo "== P3.3 introduces no P3.4/P3.5 functionality and no comparison =="
	@! grep -nE '^(P34|P35)_' $(GEMM_P33_WRAPPER)
	@! grep -nE '^(FROZEN_)?(USE_2CTA|PERSISTENT|SWEEP|COMPARISON)' $(GEMM_P33_WRAPPER)
	@echo "   (the tokenized identifier ban for TFLOP/s, speedup, efficiency, bandwidth,"
	@echo "    utilization, winner labels, Nsight Compute, autotuning, and campaign trees"
	@echo "    lives in $(GEMM_P33_CHECKER), which scans Python NAME tokens so that prose"
	@echo "    explaining what P3.3 does NOT compute stays legal while code does not)"
	@echo "== P3.4 files present, executable, and still vendoring no NVIDIA GEMM source =="
	@test -f $(GEMM_P34_WRAPPER)
	@test -f $(GEMM_P34_CHECKER)
	@test -f $(GEMM_P34_PROTOCOL)
	@test -x $(GEMM_P34_WRAPPER)
	@test -x $(GEMM_P34_CHECKER)
	@! grep -nE '^(import|from|def|class) ' $(GEMM_P34_PROTOCOL)
	@echo "== P3.4 python syntax, GPU-free self-tests, and the full contract check =="
	python3 -m py_compile $(GEMM_P34_WRAPPER) $(GEMM_P34_CHECKER)
	python3 $(GEMM_P34_WRAPPER) --self-test
	python3 $(GEMM_P34_CHECKER) --self-test
	python3 $(GEMM_P34_CHECKER) .
	@rm -rf src/gemm/__pycache__ scripts/__pycache__
	@echo "== P3.4 frozen single shape and scientific contract cannot silently change =="
	@grep -Eq '^FROZEN_M = 4096$$' $(GEMM_P34_WRAPPER)
	@grep -Eq '^FROZEN_N = 4096$$' $(GEMM_P34_WRAPPER)
	@grep -Eq '^FROZEN_K = 4096$$' $(GEMM_P34_WRAPPER)
	@grep -Eq '^FROZEN_L = 1$$' $(GEMM_P34_WRAPPER)
	@grep -Eq '^FROZEN_AB_DTYPE = "BFloat16"$$' $(GEMM_P34_WRAPPER)
	@grep -Eq '^FROZEN_ACC_DTYPE = "Float32"$$' $(GEMM_P34_WRAPPER)
	@grep -Eq '^FROZEN_C_DTYPE = "Float32"$$' $(GEMM_P34_WRAPPER)
	@grep -Eq '^FROZEN_A_MAJOR = "k"$$' $(GEMM_P34_WRAPPER)
	@grep -Eq '^FROZEN_B_MAJOR = "k"$$' $(GEMM_P34_WRAPPER)
	@grep -Eq '^FROZEN_C_MAJOR = "n"$$' $(GEMM_P34_WRAPPER)
	@grep -Eq '^FROZEN_USE_TMA_STORE = True$$' $(GEMM_P34_WRAPPER)
	@grep -Eq '^FROZEN_SEED = 1111$$' $(GEMM_P34_WRAPPER)
	@grep -Eq '^SCHEMA_VERSION = "p34.v1"$$' $(GEMM_P34_WRAPPER)
	@grep -Eq '^RUN_KIND = "smoke"$$' $(GEMM_P34_WRAPPER)
	@grep -Eq '^METHOD = "cutedsl"$$' $(GEMM_P34_WRAPPER)
	@grep -Eq '^REFERENCE = "torch_cuda_fp32_ieee"$$' $(GEMM_P34_WRAPPER)
	@grep -Eq '^CACHE_MODE = "hot"$$' $(GEMM_P34_WRAPPER)
	@grep -Eq '^PUBLISHABLE = "false"$$' $(GEMM_P34_WRAPPER)
	@echo "== P3.4 declares exactly the three frozen variants, with the frozen geometry =="
	@grep -Eq '^VARIANT_NONPERSISTENT_1CTA = "nonpersistent_1cta"$$' $(GEMM_P34_WRAPPER)
	@grep -Eq '^VARIANT_PERSISTENT_1CTA = "persistent_1cta"$$' $(GEMM_P34_WRAPPER)
	@grep -Eq '^VARIANT_PERSISTENT_2CTA = "persistent_2cta"$$' $(GEMM_P34_WRAPPER)
	@grep -Eq '^SCHEDULER_NONPERSISTENT = "nonpersistent"$$' $(GEMM_P34_WRAPPER)
	@grep -Eq '^SCHEDULER_STATIC_PERSISTENT = "static_persistent"$$' $(GEMM_P34_WRAPPER)
	@grep -Eq '^UPSTREAM_CLASS_NONPERSISTENT = "DenseGemmKernel"$$' $(GEMM_P34_WRAPPER)
	@grep -Eq '^UPSTREAM_CLASS_PERSISTENT = "PersistentDenseGemmKernel"$$' $(GEMM_P34_WRAPPER)
	@grep -Fq '"mma_tiler_mn": (128, 128),' $(GEMM_P34_WRAPPER)
	@grep -Fq '"mma_tiler_mn": (256, 128),' $(GEMM_P34_WRAPPER)
	@grep -Fq '"cluster_shape_mn": (2, 1),' $(GEMM_P34_WRAPPER)
	@echo "   (the exact per-variant class/scheduler/tiler/cluster/2-CTA mapping and the"
	@echo "    256/2=128 per-CTA M extent are enforced structurally by $(GEMM_P34_CHECKER))"
	@echo "== P3.4 pins the second official source in PHASE3_VERSIONS.env, not VERSIONS.env =="
	@grep -Eq '^CUTEDSL_P34_PERSISTENT_EXAMPLE_PATH=' PHASE3_VERSIONS.env
	@grep -Eq '^CUTEDSL_P34_PERSISTENT_EXAMPLE_GIT_BLOB=[0-9a-f]{40}$$' PHASE3_VERSIONS.env
	@grep -Eq '^CUTEDSL_P34_PERSISTENT_EXAMPLE_SHA256=[0-9a-f]{64}$$' PHASE3_VERSIONS.env
	@grep -Fq 'dense_gemm_persistent.py' PHASE3_VERSIONS.env
	@! grep -nE '^CUTEDSL_P3[0-9]' VERSIONS.env
	@echo "== P3.4 exposes no frozen scientific parameter and can never skip correctness =="
	@pat='--mnkl|--shape|--variant|--scheduler|--persistent|--nonpersistent'; \
	pat="$$pat|--mma_tiler|--mma-tiler|--cluster_shape|--cluster-shape|--use_2cta|--use-2cta"; \
	pat="$$pat|--ab_dtype|--c_dtype|--acc_dtype|--a_major|--b_major|--c_major"; \
	pat="$$pat|--use_tma_store|--use-tma-store|--seed|--tolerance|--atol|--rtol"; \
	pat="$$pat|--skip-ref""-check|--skip_ref""_check|--use_cold""_l2|--example|--source-path"; \
	! grep -nE -- "$$pat" $(GEMM_P34_WRAPPER)
	@echo "   (the checker names those spellings on purpose, in order to ban them)"
	@echo "== P3.4 writes no result file and creates no campaign directory =="
	@! grep -nE 'results/raw|results/preflight' $(GEMM_P34_WRAPPER)
	@echo "== P3.4 reuses the audited launcher and never invokes Docker for GPU work =="
	@grep -Eq '^gemm-cutedsl-p34-smoke:$$' Makefile
	@grep -Fq 'scripts/run_container.sh' Makefile
	@echo "== P3.4 GPU-free gate actually executes the existing P3.3 gate =="
	@grep -Eq '^gemm-cutedsl-p34-check: gemm-cublaslt-p33-check$$' Makefile
	@echo "== P3.4 smoke runs exactly the frozen non-publishable iteration counts =="
	@grep -Fq -- '--warmup-iterations 2 \' Makefile
	@grep -Fq -- '--iterations 10' Makefile
	@echo "== truthful P3.4 status assertions =="
	@grep -Fq 'P3.4 | Three execution variants | YES | YES | YES |' PLAN.md
	@grep -Fq 'P3.4 = YES / YES / YES' $(GEMM_P34_PROTOCOL)
	@grep -Fq 'P3.4 creates no publishable performance result' $(GEMM_P34_PROTOCOL)
	@grep -Fq 'P3.4 (three execution variants)' README.md
	@! grep -nF 'P3.4 | Three execution variants | NO | NO | NO |' PLAN.md
	@! grep -nF 'P3.4 | Three execution variants | YES | NO | NO |' PLAN.md
	@! grep -nF 'P3.4 | Three execution variants | YES | YES | NO |' PLAN.md
	@! grep -nF 'P3.4 | Three execution variants | YES | NO | YES |' PLAN.md
	@echo "== P3.4 itself still implements no P3.5 functionality and no comparison =="
	@! grep -nE '^(P35)_' $(GEMM_P34_WRAPPER)
	@echo "== P3.5 files present, executable, and still vendoring no NVIDIA GEMM source =="
	@test -f $(GEMM_P35_WRAPPER)
	@test -f $(GEMM_P35_BRIDGE)
	@test -f $(GEMM_P35_CHECKER)
	@test -f $(GEMM_P35_PROTOCOL)
	@test -x $(GEMM_P35_WRAPPER)
	@test -x $(GEMM_P35_CHECKER)
	@! grep -nE '^(import|from|def|class) ' $(GEMM_P35_PROTOCOL)
	@! test -e src/gemm/dense_gemm.py
	@! test -e src/gemm/dense_gemm_persistent.py
	@echo "== P3.5 python syntax, GPU-free self-tests, and the full contract check =="
	python3 -m py_compile $(GEMM_P35_WRAPPER) $(GEMM_P35_CHECKER)
	python3 $(GEMM_P35_WRAPPER) --self-test
	python3 $(GEMM_P35_CHECKER) --self-test
	python3 $(GEMM_P35_CHECKER) .
	@rm -rf src/gemm/__pycache__ scripts/__pycache__
	@echo "== P3.5 declares exactly the five frozen shapes, in the frozen order =="
	@grep -Fq '    (4096, 4096, 4096, 1),' $(GEMM_P35_WRAPPER)
	@grep -Fq '    (8192, 8192, 8192, 1),' $(GEMM_P35_WRAPPER)
	@grep -Fq '    (16384, 512, 4096, 1),' $(GEMM_P35_WRAPPER)
	@grep -Fq '    (32768, 512, 4096, 1),' $(GEMM_P35_WRAPPER)
	@grep -Fq '    (512, 16384, 4096, 1),' $(GEMM_P35_WRAPPER)
	@grep -Fq '    {4096, 4096, 4096},' $(GEMM_P35_BRIDGE)
	@grep -Fq '    {8192, 8192, 8192},' $(GEMM_P35_BRIDGE)
	@grep -Fq '    {16384, 512, 4096},' $(GEMM_P35_BRIDGE)
	@grep -Fq '    {32768, 512, 4096},' $(GEMM_P35_BRIDGE)
	@grep -Fq '    {512, 16384, 4096},' $(GEMM_P35_BRIDGE)
	@echo "   (the exact five-shape order, the four-candidate order, and the fact that the"
	@echo "    wrapper and the C bridge must agree are enforced structurally by"
	@echo "    $(GEMM_P35_CHECKER))"
	@echo "== P3.5 frozen scientific contract cannot silently change =="
	@grep -Eq '^FROZEN_L = 1$$' $(GEMM_P35_WRAPPER)
	@grep -Eq '^FROZEN_AB_DTYPE = "BFloat16"$$' $(GEMM_P35_WRAPPER)
	@grep -Eq '^FROZEN_ACC_DTYPE = "Float32"$$' $(GEMM_P35_WRAPPER)
	@grep -Eq '^FROZEN_C_DTYPE = "Float32"$$' $(GEMM_P35_WRAPPER)
	@grep -Eq '^FROZEN_A_MAJOR = "k"$$' $(GEMM_P35_WRAPPER)
	@grep -Eq '^FROZEN_B_MAJOR = "k"$$' $(GEMM_P35_WRAPPER)
	@grep -Eq '^FROZEN_C_MAJOR = "n"$$' $(GEMM_P35_WRAPPER)
	@grep -Eq '^FROZEN_USE_TMA_STORE = True$$' $(GEMM_P35_WRAPPER)
	@grep -Eq '^FROZEN_SEED = 1111$$' $(GEMM_P35_WRAPPER)
	@grep -Eq '^SCHEMA_VERSION = "p35.v1"$$' $(GEMM_P35_WRAPPER)
	@grep -Eq '^RUN_KIND = "smoke"$$' $(GEMM_P35_WRAPPER)
	@grep -Eq '^REFERENCE = "torch_cuda_fp32_ieee"$$' $(GEMM_P35_WRAPPER)
	@grep -Eq '^CACHE_MODE = "hot"$$' $(GEMM_P35_WRAPPER)
	@grep -Eq '^PUBLISHABLE = "false"$$' $(GEMM_P35_WRAPPER)
	@grep -Eq '^NOT_APPLICABLE = "not_applicable"$$' $(GEMM_P35_WRAPPER)
	@grep -Eq '^FLOPS_PER_MAC = 2$$' $(GEMM_P35_WRAPPER)
	@echo "== P3.5 keeps the closed P3.3 cuBLASLt policy, unchanged =="
	@grep -Eq '^FROZEN_WORKSPACE_LIMIT_BYTES = 67108864$$' $(GEMM_P35_WRAPPER)
	@grep -Eq '^FROZEN_HEURISTIC_REQUESTED = 32$$' $(GEMM_P35_WRAPPER)
	@grep -Eq '^FROZEN_SEARCH_MODE = "CUBLASLT_SEARCH_BEST_FIT"$$' $(GEMM_P35_WRAPPER)
	@grep -Eq '^FROZEN_TRANSA = "CUBLAS_OP_N"$$' $(GEMM_P35_WRAPPER)
	@grep -Eq '^FROZEN_TRANSB = "CUBLAS_OP_T"$$' $(GEMM_P35_WRAPPER)
	@grep -Eq '^FROZEN_ORDER = "CUBLASLT_ORDER_ROW"$$' $(GEMM_P35_WRAPPER)
	@grep -Fq 'static const uint64_t P35_WORKSPACE_LIMIT_BYTES = 67108864ULL;' $(GEMM_P35_BRIDGE)
	@grep -Fq 'static const int P35_HEURISTIC_REQUESTED = 32;' $(GEMM_P35_BRIDGE)
	@echo "== the P3.5 measured path is cublasLtMatmul, with no fallback GEMM API =="
	@grep -Eq 'cublasLtMatmul[[:space:]]*\(' $(GEMM_P35_BRIDGE)
	@grep -Eq 'cublasLtMatmulAlgoCheck[[:space:]]*\(' $(GEMM_P35_BRIDGE)
	@grep -Eq 'cublasLtMatmulAlgoGetHeuristic[[:space:]]*\(' $(GEMM_P35_BRIDGE)
	@# Call sites, not bare names: both files deliberately NAME the forbidden
	@# entry points in prose that explains they are never used. The rigorous
	@# comment-stripped scan, and the exactly-one-cublasLtMatmul-call-site rule,
	@# live in $(GEMM_P35_CHECKER).
	@pat='cublasGemmEx|cublasGemmStridedBatchedEx|cublasGemmBatchedEx'; \
	pat="$$pat|cublasSgemm|cublasHgemm|cublasLtMatmulAlgoGetIds|cublasLtMatmulAlgoInit"; \
	! grep -nE -- "($$pat)[[:space:]]*\(" $(GEMM_P35_WRAPPER) $(GEMM_P35_BRIDGE)
	@echo "   (the checker names those spellings on purpose, in order to ban them)"
	@echo "== the P3.5 bridge owns no GEMM kernel, prints nothing, and times nothing =="
	@! grep -nE '__global__|__device__' $(GEMM_P35_BRIDGE)
	@! grep -nE '(^|[^a-zA-Z0-9_])(printf|puts|fputs)[[:space:]]*\(' $(GEMM_P35_BRIDGE)
	@! grep -nE 'std::(cout|cerr|clog)' $(GEMM_P35_BRIDGE)
	@! grep -nE 'cudaEventRecord|cudaEventElapsedTime|std::chrono|clock_gettime|gettimeofday' $(GEMM_P35_BRIDGE)
	@grep -Fq 'extern "C"' $(GEMM_P35_BRIDGE)
	@grep -Fq 'catch (...)' $(GEMM_P35_BRIDGE)
	@echo "== the P3.5 bridge validates every derived size against overflow =="
	@grep -Fq 'INT64_MAX' $(GEMM_P35_BRIDGE)
	@grep -Fq 'SIZE_MAX' $(GEMM_P35_BRIDGE)
	@grep -Eq 'p35_shape_index_of[[:space:]]*\(' $(GEMM_P35_BRIDGE)
	@echo "== P3.5 leaves the closed P3.3 bridge and its ABI untouched =="
	@! grep -nE 'p35_' $(GEMM_P33_BRIDGE)
	@grep -Fq 'p33_plan_create' $(GEMM_P33_BRIDGE)
	@echo "== P3.5 exposes no frozen scientific parameter and can never skip correctness =="
	@pat='--mnkl|--shape|--shapes|--variant|--candidate|--method|--scheduler'; \
	pat="$$pat|--persistent|--nonpersistent|--mma_tiler|--mma-tiler|--cluster_shape"; \
	pat="$$pat|--cluster-shape|--use_2cta|--use-2cta|--ab_dtype|--c_dtype|--acc_dtype"; \
	pat="$$pat|--a_major|--b_major|--c_major|--use_tma_store|--use-tma-store|--seed"; \
	pat="$$pat|--tolerance|--atol|--rtol|--workspace|--algo|--heuristic|--search"; \
	pat="$$pat|--skip-ref""-check|--skip_ref""_check|--use_cold""_l2|--example|--source-path"; \
	pat="$$pat|--output|--out-file|--csv|--publish|--partial|--only|--input|--config"; \
	! grep -nE -- "$$pat" $(GEMM_P35_WRAPPER)
	@echo "   (the checker names those spellings on purpose, in order to ban them)"
	@echo "== P3.5 writes no result file and creates no campaign directory =="
	@! grep -nE 'results/raw|results/preflight' $(GEMM_P35_WRAPPER)
	@echo "== P3.5 adds no key to either version contract =="
	@! grep -nE '^(CUTEDSL_)?P35_' PHASE3_VERSIONS.env VERSIONS.env
	@! grep -nE '^CUTEDSL_P35' PHASE3_VERSIONS.env VERSIONS.env
	@echo "== P3.5 reuses the audited launcher and never invokes Docker for GPU work =="
	@grep -Eq '^gemm-comparison-p35-smoke:$$' Makefile
	@grep -Fq 'scripts/run_container.sh' Makefile
	@echo "== P3.5 GPU-free gate actually executes the existing P3.4 gate =="
	@grep -Eq '^gemm-comparison-p35-check: gemm-cutedsl-p34-check$$' Makefile
	@echo "== P3.5 smoke runs exactly the frozen non-publishable iteration counts =="
	@grep -Fq -- '--warmup-iterations 2 \' Makefile
	@grep -Fq -- '--iterations 10' Makefile
	@echo "== truthful P3.5 status assertions =="
	@grep -Fq 'P3.5 | Five shapes and comparison | YES | YES | YES |' PLAN.md
	@grep -Fq 'P3.5 = YES / YES / YES' $(GEMM_P35_PROTOCOL)
	@grep -Fq 'P3.5 creates no publishable performance result' $(GEMM_P35_PROTOCOL)
	@grep -Fq 'P3.5 (five shapes and comparison)' README.md
	@! grep -nF 'P3.5 | Five shapes and comparison | NO | NO | NO |' PLAN.md
	@! grep -nF 'P3.5 | Five shapes and comparison | YES | YES | NO |' PLAN.md
	@! grep -nF 'P3.5 | Five shapes and comparison | YES | NO | YES |' PLAN.md
	@! grep -nF 'P3.5 | Five shapes and comparison | YES | NO | NO |' PLAN.md
	@grep -Fq 'P3.5: CLOSED' README.md
	@grep -Fq 'Phase 3: CLOSED' README.md
	@echo "== P3.5 introduces no Phase 4 functionality and no statistical treatment =="
	@grep -Fq 'P4.1 | Orchestrator | NO | NO | NO |' PLAN.md
	@grep -Fq 'P4.2 | Pilot plus three final campaigns | NO | NO | NO |' PLAN.md
	@grep -Fq 'P4.3 | Integrated analysis, documentation, audit | NO | NO | NO |' PLAN.md
	@! grep -nE '^(P4|P41|P42|P43)_' $(GEMM_P35_WRAPPER)
	@echo "   (the tokenized identifier ban for confidence intervals, bootstraps, outlier"
	@echo "    removal, rooflines, bandwidth, utilization, Nsight Compute, autotuning,"
	@echo "    plots, and campaign trees lives in $(GEMM_P35_CHECKER), which scans Python"
	@echo "    NAME tokens so that prose explaining what P3.5 does NOT compute stays legal"
	@echo "    while code does not)"
	@echo "check-static: OK"

build-image:
	docker build \
		--platform "$(CUDA_IMAGE_PLATFORM)" \
		--build-arg BASE_IMAGE="$(CUDA_IMAGE)@$(CUDA_IMAGE_DIGEST)" \
		--build-arg CUDA_VERSION="$(CUDA_VERSION)" \
		--build-arg CUTLASS_VERSION="$(CUTLASS_VERSION)" \
		--build-arg CUTLASS_COMMIT="$(CUTLASS_COMMIT)" \
		--build-arg MAX_BUILD_JOBS="$(MAX_BUILD_JOBS)" \
		--build-arg PYTORCH_VERSION="$(PYTORCH_VERSION)" \
		--build-arg PYTORCH_INDEX_URL="$(PYTORCH_INDEX_URL)" \
		--build-arg PYTORCH_CUDA_VERSION="$(PYTORCH_CUDA_VERSION)" \
		--build-arg CUDA_PYTHON_VERSION="$(CUDA_PYTHON_VERSION)" \
		--build-arg CUDA_BINDINGS_VERSION="$(CUDA_BINDINGS_VERSION)" \
		--tag "$(IMAGE_TAG)" \
		.

check-env:
	docker run --rm \
		--network none \
		--security-opt no-new-privileges \
		--cap-drop ALL \
		-e CUDA_SHORT_VERSION="$(CUDA_SHORT_VERSION)" \
		-e CUTEDSL_VERSION="$(CUTEDSL_VERSION)" \
		-e PYTORCH_VERSION="$(PYTORCH_VERSION)" \
		-e PYTORCH_CUDA_VERSION="$(PYTORCH_CUDA_VERSION)" \
		-e CUDA_PYTHON_VERSION="$(CUDA_PYTHON_VERSION)" \
		-e CUDA_BINDINGS_VERSION="$(CUDA_BINDINGS_VERSION)" \
		"$(IMAGE_TAG)" \
		bash -c 'set -euo pipefail; \
			for tool in nvcc ptxas cuobjdump nvdisasm ncu python3; do \
				command -v "$$tool" >/dev/null 2>&1 \
					|| { echo "check-env: MISSING tool: $$tool" >&2; exit 1; }; \
			done; \
			nvcc_v="$$(nvcc --version | grep -i release)"; \
			[ -n "$$nvcc_v" ] || { echo "check-env: empty nvcc version output" >&2; exit 1; }; \
			echo "nvcc: $$nvcc_v"; \
			case "$$nvcc_v" in \
				*"release $${CUDA_SHORT_VERSION}"*) ;; \
				*) echo "check-env: nvcc is not CUDA $${CUDA_SHORT_VERSION}: $$nvcc_v" >&2; exit 1;; \
			esac; \
			ptxas_v="$$(ptxas --version | grep -i release)"; \
			[ -n "$$ptxas_v" ] || { echo "check-env: empty ptxas version output" >&2; exit 1; }; \
			echo "ptxas: $$ptxas_v"; \
			case "$$ptxas_v" in \
				*"release $${CUDA_SHORT_VERSION}"*) ;; \
				*) echo "check-env: ptxas is not CUDA $${CUDA_SHORT_VERSION}: $$ptxas_v" >&2; exit 1;; \
			esac; \
			cuobjdump_v="$$(cuobjdump --version | grep -i release)"; \
			[ -n "$$cuobjdump_v" ] || { echo "check-env: empty cuobjdump version output" >&2; exit 1; }; \
			echo "cuobjdump: $$cuobjdump_v"; \
			nvdisasm_v="$$(nvdisasm --version | grep -i release)"; \
			[ -n "$$nvdisasm_v" ] || { echo "check-env: empty nvdisasm version output" >&2; exit 1; }; \
			echo "nvdisasm: $$nvdisasm_v"; \
			ncu_v="$$(ncu --version | grep -i version)"; \
			[ -n "$$ncu_v" ] || { echo "check-env: empty ncu version output" >&2; exit 1; }; \
			echo "ncu: $$ncu_v"; \
			py_v="$$(python3 --version)"; \
			[ -n "$$py_v" ] || { echo "check-env: empty python3 version output" >&2; exit 1; }; \
			echo "python3: $$py_v"; \
			python3 -c "import os, cutlass; v = cutlass.__version__; expected = os.environ[\"CUTEDSL_VERSION\"]; assert v == expected, f\"CuTeDSL {v} != pinned {expected}\"; print(\"cutedsl:\", v)"; \
			python3 -c "import os, torch; v = torch.__version__; expected = os.environ[\"PYTORCH_VERSION\"]; assert v == expected, f\"torch {v} != pinned {expected}\"; c = torch.version.cuda; expected_cuda = os.environ[\"PYTORCH_CUDA_VERSION\"]; assert c == expected_cuda, f\"torch CUDA {c} != pinned {expected_cuda}\"; print(\"torch:\", v, \"cuda:\", c)"; \
			python3 -c "import os; from importlib.metadata import version; expected = {\"cuda-python\": os.environ[\"CUDA_PYTHON_VERSION\"], \"cuda-bindings\": os.environ[\"CUDA_BINDINGS_VERSION\"]}; installed = {name: version(name) for name in expected}; assert installed == expected, f\"installed distributions {installed} != pinned {expected}\"; print(\"cuda distributions:\", installed)"; \
			echo "== pip check: the dependency graph must be consistent =="; \
			python3 -m pip check; \
			echo "check-env: OK"'

preflight:
	@if [ -z "$${BLACKWELL_GPU_INDEX:-}" ]; then \
		echo "ERROR: BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index."; \
		echo "       Example: BLACKWELL_GPU_INDEX=3 make preflight"; \
		echo "       This project never selects a GPU automatically."; \
		exit 2; \
	fi
	scripts/run_container.sh bash scripts/preflight.sh

# --- P1.1: standalone LDGSTS microbenchmark ---------------------------------
# memory-ldgsts-build and memory-ldgsts-sass never touch a GPU: they compile
# and disassemble inside the pinned, network-less, unprivileged image, same
# secure pattern as check-env. memory-ldgsts-self-test and memory-ldgsts-smoke
# execute on GPU and therefore go exclusively through scripts/run_container.sh,
# which requires an explicit BLACKWELL_GPU_INDEX and proves the device is free.

memory-ldgsts-build:
	@mkdir -p build/memory
	docker run --rm \
		--network none \
		--security-opt no-new-privileges \
		--cap-drop ALL \
		--user "$$(id -u):$$(id -g)" \
		-e HOME=/tmp \
		-v "$(CURDIR):/workspace" \
		-w /workspace \
		"$(IMAGE_TAG)" \
		nvcc -std=c++17 -O3 -lineinfo -arch=$(CUDA_ARCH) \
			-o $(MEMORY_LDGSTS_BIN) $(MEMORY_LDGSTS_SRC)

memory-ldgsts-sass: memory-ldgsts-build
	@mkdir -p build/memory
	docker run --rm \
		--network none \
		--security-opt no-new-privileges \
		--cap-drop ALL \
		--user "$$(id -u):$$(id -g)" \
		-e HOME=/tmp \
		-v "$(CURDIR):/workspace" \
		-w /workspace \
		"$(IMAGE_TAG)" \
		python3 scripts/check_ldgsts_sass.py $(MEMORY_LDGSTS_BIN) $(MEMORY_LDGSTS_SASS)

memory-ldgsts-self-test: memory-ldgsts-build
	@if [ -z "$${BLACKWELL_GPU_INDEX:-}" ]; then \
		echo "ERROR: BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index."; \
		echo "       Example: BLACKWELL_GPU_INDEX=3 make memory-ldgsts-self-test"; \
		echo "       This project never selects a GPU automatically."; \
		exit 2; \
	fi
	scripts/run_container.sh $(MEMORY_LDGSTS_BIN) --self-test

memory-ldgsts-smoke: memory-ldgsts-build
	@if [ -z "$${BLACKWELL_GPU_INDEX:-}" ]; then \
		echo "ERROR: BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index."; \
		echo "       Example: BLACKWELL_GPU_INDEX=3 make memory-ldgsts-smoke"; \
		echo "       This project never selects a GPU automatically."; \
		exit 2; \
	fi
	@echo "== memory-ldgsts-smoke: self-test =="
	scripts/run_container.sh $(MEMORY_LDGSTS_BIN) --self-test
	@echo "== memory-ldgsts-smoke: short run_kind=smoke measurement (NOT a final result) =="
	scripts/run_container.sh $(MEMORY_LDGSTS_BIN) \
		--stages 4 --bytes-in-flight-kib 32 --run-kind smoke \
		--working-set-mib 64 --passes 2 --warmup-ms 200 --repetitions 5
	@echo "=============================================================================="
	@echo "The run_kind=smoke output above is a functional smoke check only. It is NOT a"
	@echo "final experimental result and must not be cited as a performance number."
	@echo "=============================================================================="

# --- P1.2: standalone 2D unicast TMA microbenchmark -------------------------
# memory-tma-build and memory-tma-sass never touch a GPU: they compile and
# disassemble inside the pinned, network-less, unprivileged image, same
# secure pattern as memory-ldgsts-build/sass. memory-tma-self-test and
# memory-tma-smoke execute on GPU and therefore go exclusively through
# scripts/run_container.sh, which requires an explicit BLACKWELL_GPU_INDEX
# and proves the device is free.

memory-tma-build:
	@mkdir -p build/memory
	docker run --rm \
		--network none \
		--security-opt no-new-privileges \
		--cap-drop ALL \
		--user "$$(id -u):$$(id -g)" \
		-e HOME=/tmp \
		-v "$(CURDIR):/workspace" \
		-w /workspace \
		"$(IMAGE_TAG)" \
		nvcc -std=c++17 -O3 -lineinfo -arch=$(CUDA_ARCH) \
			-o $(MEMORY_TMA_BIN) $(MEMORY_TMA_SRC)

memory-tma-sass: memory-tma-build
	@mkdir -p build/memory
	docker run --rm \
		--network none \
		--security-opt no-new-privileges \
		--cap-drop ALL \
		--user "$$(id -u):$$(id -g)" \
		-e HOME=/tmp \
		-v "$(CURDIR):/workspace" \
		-w /workspace \
		"$(IMAGE_TAG)" \
		python3 scripts/check_tma_sass.py $(MEMORY_TMA_BIN) $(MEMORY_TMA_SASS)

memory-tma-self-test: memory-tma-build
	@if [ -z "$${BLACKWELL_GPU_INDEX:-}" ]; then \
		echo "ERROR: BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index."; \
		echo "       Example: BLACKWELL_GPU_INDEX=3 make memory-tma-self-test"; \
		echo "       This project never selects a GPU automatically."; \
		exit 2; \
	fi
	scripts/run_container.sh $(MEMORY_TMA_BIN) --self-test

memory-tma-smoke: memory-tma-build
	@if [ -z "$${BLACKWELL_GPU_INDEX:-}" ]; then \
		echo "ERROR: BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index."; \
		echo "       Example: BLACKWELL_GPU_INDEX=3 make memory-tma-smoke"; \
		echo "       This project never selects a GPU automatically."; \
		exit 2; \
	fi
	@echo "== memory-tma-smoke: self-test =="
	scripts/run_container.sh $(MEMORY_TMA_BIN) --self-test
	@echo "== memory-tma-smoke: short run_kind=smoke measurement (NOT a final result) =="
	scripts/run_container.sh $(MEMORY_TMA_BIN) \
		--stages 4 --bytes-in-flight-kib 32 --run-kind smoke \
		--working-set-mib 64 --passes 2 --warmup-ms 200 --repetitions 5
	@echo "=============================================================================="
	@echo "The run_kind=smoke output above is a functional smoke check only. It is NOT a"
	@echo "final experimental result and must not be cited as a performance number."
	@echo "=============================================================================="

# --- P1.3: joint LDGSTS/TMA sweep infrastructure (exp01_memory_paths) -------
# memory-paths-plan and memory-paths-check never touch a GPU or Docker: they
# only exercise scripts/run_exp01_memory_paths.sh's and
# scripts/aggregate_exp01_memory_paths.py's own GPU-free CLI paths and
# synthetic self-tests. memory-paths-smoke is the only P1.3 target that
# executes on GPU; it requires an explicit BLACKWELL_GPU_INDEX, reuses the
# memory-ldgsts-sass/memory-tma-sass gates above, then runs both binaries'
# full --self-test through scripts/run_container.sh before any of the 18
# smoke configurations. No P1.3 target invokes Nsight Compute or collects
# run_kind=benchmark data; that remains explicit P1.4 work.

memory-paths-plan:
	$(EXP01_RUNNER) --print-plan

memory-paths-check:
	bash -n $(EXP01_RUNNER)
	@test -x $(EXP01_RUNNER)
	@test -x $(EXP01_AGGREGATOR)
	python3 -m py_compile $(EXP01_AGGREGATOR)
	@rm -rf scripts/__pycache__
	python3 $(EXP01_AGGREGATOR) --self-test
	@test "$$(python3 $(EXP01_AGGREGATOR) plan --format lines | wc -l | tr -d ' ')" -eq 18
	$(EXP01_RUNNER) --self-test
	@echo "memory-paths-check: OK"

memory-paths-smoke: memory-ldgsts-sass memory-tma-sass
	@if [ -z "$${BLACKWELL_GPU_INDEX:-}" ]; then \
		echo "ERROR: BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index."; \
		echo "       Example: BLACKWELL_GPU_INDEX=3 make memory-paths-smoke"; \
		echo "       This project never selects a GPU automatically."; \
		exit 2; \
	fi
	@echo "== memory-paths-smoke: both binary self-tests, then all 18 smoke configurations =="
	$(EXP01_RUNNER) --run-kind smoke \
		--working-set-mib 64 --passes 2 --warmup-ms 200 --repetitions 2
	@echo "=============================================================================="
	@echo "The run_kind=smoke output above is functional verification of the P1.3 sweep"
	@echo "infrastructure only. It is NOT a final experimental result, computes no"
	@echo "speedup, and must not be cited as a performance number."
	@echo "=============================================================================="

# --- P1.4: profiling, HBM validation, analysis, pilot (exp01_memory_paths_p14) -
# memory-paths-p14-plan and memory-paths-p14-check never touch a GPU, Docker,
# or the network: they only exercise scripts/run_exp01_memory_paths_p14.sh's
# and scripts/analyze_exp01_memory_paths_p14.py's own GPU-free CLI paths and
# synthetic/adversarial self-tests. memory-paths-p14-pilot and
# memory-paths-p14-profile are the only P1.4 targets that execute on GPU;
# each requires an explicit BLACKWELL_GPU_INDEX, P1_4_CAMPAIGN_ID (a canonical
# UTC timestamp), and P1_4_PREFLIGHT_SUMMARY, and never selects a GPU
# automatically. memory-paths-p14-analyze is GPU-free and validates/analyzes
# an already-COMPLETE P1.4 campaign; it requires only P1_4_CAMPAIGN_ID.

memory-paths-p14-plan:
	$(EXP01_P14_RUNNER) --print-plan

memory-paths-p14-check:
	bash -n $(EXP01_P14_RUNNER)
	@test -x $(EXP01_P14_RUNNER)
	@test -x $(EXP01_P14_ANALYZER)
	@test -x $(EXP01_P14_SAFE_CAPTURE)
	@test -x $(EXP01_P14_NCU_BRIDGE)
	python3 -m py_compile $(EXP01_P14_ANALYZER)
	python3 -m py_compile $(EXP01_P14_SAFE_CAPTURE)
	python3 -m py_compile $(EXP01_P14_NCU_BRIDGE)
	@rm -rf scripts/__pycache__
	python3 $(EXP01_P14_ANALYZER) --self-test
	python3 $(EXP01_P14_SAFE_CAPTURE) --self-test
	python3 $(EXP01_P14_NCU_BRIDGE) --self-test
	@test "$$(python3 $(EXP01_AGGREGATOR) plan --format lines | wc -l | tr -d ' ')" -eq 18
	@test "$$(python3 $(EXP01_P14_ANALYZER) plan --format lines | wc -l | tr -d ' ')" -eq 6
	$(EXP01_P14_RUNNER) --self-test
	@echo "memory-paths-p14-check: OK"

memory-paths-p14-pilot:
	@if [ -z "$${BLACKWELL_GPU_INDEX:-}" ]; then \
		echo "ERROR: BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index."; \
		echo "       Example: BLACKWELL_GPU_INDEX=3 make memory-paths-p14-pilot"; \
		echo "       This project never selects a GPU automatically."; \
		exit 2; \
	fi
	@if [ -z "$${P1_4_CAMPAIGN_ID:-}" ]; then \
		echo "ERROR: P1_4_CAMPAIGN_ID must be set explicitly to a canonical UTC timestamp"; \
		echo "       YYYYMMDDTHHMMSSZ. Example: P1_4_CAMPAIGN_ID=$$(date -u +%Y%m%dT%H%M%SZ)"; \
		exit 2; \
	fi
	@if [ -z "$${P1_4_PREFLIGHT_SUMMARY:-}" ]; then \
		echo "ERROR: P1_4_PREFLIGHT_SUMMARY must be set explicitly to a fresh preflight"; \
		echo "       summary.json path (see 'make preflight')."; \
		exit 2; \
	fi
	$(EXP01_P14_RUNNER) --pilot

memory-paths-p14-profile:
	@if [ -z "$${BLACKWELL_GPU_INDEX:-}" ]; then \
		echo "ERROR: BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index."; \
		echo "       Example: BLACKWELL_GPU_INDEX=3 make memory-paths-p14-profile"; \
		echo "       This project never selects a GPU automatically."; \
		exit 2; \
	fi
	@if [ -z "$${P1_4_CAMPAIGN_ID:-}" ]; then \
		echo "ERROR: P1_4_CAMPAIGN_ID must be set explicitly to the same canonical UTC"; \
		echo "       timestamp used for 'make memory-paths-p14-pilot'."; \
		exit 2; \
	fi
	@if [ -z "$${P1_4_PREFLIGHT_SUMMARY:-}" ]; then \
		echo "ERROR: P1_4_PREFLIGHT_SUMMARY must be set explicitly to a fresh preflight"; \
		echo "       summary.json path (see 'make preflight')."; \
		exit 2; \
	fi
	$(EXP01_P14_RUNNER) --profile

memory-paths-p14-analyze:
	@if [ -z "$${P1_4_CAMPAIGN_ID:-}" ]; then \
		echo "ERROR: P1_4_CAMPAIGN_ID must be set explicitly to the campaign to analyze."; \
		exit 2; \
	fi
	python3 $(EXP01_P14_ANALYZER) analyze \
		--campaign-dir $(EXP01_P14_RAW_ROOT)/$${P1_4_CAMPAIGN_ID} \
		--analyzed-at-utc "$$(date -u +%Y%m%dT%H%M%SZ)"

# --- P2.1: 1-SM BF16 UMMA microbenchmark (tcgen05.mma, kind::f16, cta_group::1) ---
# compute-umma-1sm-build and compute-umma-1sm-sass never touch a GPU: they
# compile and disassemble inside the pinned, network-less, unprivileged image,
# the same secure pattern as memory-ldgsts-build/sass. compute-umma-1sm-check
# is also GPU-free (Python syntax, the checker's own synthetic self-test,
# source-level contract checks) but depends on compute-umma-1sm-sass to also
# validate the real compiled cubin. compute-umma-1sm-self-test and
# compute-umma-1sm-smoke are the only P2.1 targets that execute on GPU; each
# requires an explicit BLACKWELL_GPU_INDEX and goes exclusively through
# scripts/run_container.sh. See src/compute/P2_PROTOCOL.md for the complete
# frozen contract; P2.1 is implemented, independently audited, and
# functionally verified on GB300, and produces no publishable result.

compute-umma-1sm-build:
	@mkdir -p build/compute
	docker run --rm \
		--network none \
		--security-opt no-new-privileges \
		--cap-drop ALL \
		--user "$$(id -u):$$(id -g)" \
		-e HOME=/tmp \
		-v "$(CURDIR):/workspace" \
		-w /workspace \
		"$(IMAGE_TAG)" \
		nvcc -std=c++17 -O3 -lineinfo $(COMPUTE_UMMA_1SM_ARCH_FLAGS) \
			-o $(COMPUTE_UMMA_1SM_BIN) $(COMPUTE_UMMA_1SM_SRC)

compute-umma-1sm-sass: compute-umma-1sm-build
	@mkdir -p build/compute
	docker run --rm \
		--network none \
		--security-opt no-new-privileges \
		--cap-drop ALL \
		--user "$$(id -u):$$(id -g)" \
		-e HOME=/tmp \
		-v "$(CURDIR):/workspace" \
		-w /workspace \
		"$(IMAGE_TAG)" \
		python3 $(COMPUTE_UMMA_1SM_CHECKER) $(COMPUTE_UMMA_1SM_BIN) $(COMPUTE_UMMA_1SM_SASS)

compute-umma-1sm-check: compute-umma-1sm-sass
	@test -x $(COMPUTE_UMMA_1SM_CHECKER)
	python3 -m py_compile $(COMPUTE_UMMA_1SM_CHECKER)
	@rm -rf scripts/__pycache__
	python3 $(COMPUTE_UMMA_1SM_CHECKER) --self-test
	@test "$$(grep -oE 'UMMA_1SM_DEFINE_KERNEL\([0-9]+, [0-9]+\)' $(COMPUTE_UMMA_1SM_SRC) | wc -l | tr -d ' ')" -eq 12
	@echo "== P2.1 CLI contract: --help exits 0 and -h is rejected, both without GPU access =="
	docker run --rm \
		--network none \
		--security-opt no-new-privileges \
		--cap-drop ALL \
		--user "$$(id -u):$$(id -g)" \
		-e HOME=/tmp \
		-v "$(CURDIR):/workspace" \
		-w /workspace \
		"$(IMAGE_TAG)" \
		sh -c '$(COMPUTE_UMMA_1SM_BIN) --help >/dev/null && echo "--help: exit 0 (OK)"'
	docker run --rm \
		--network none \
		--security-opt no-new-privileges \
		--cap-drop ALL \
		--user "$$(id -u):$$(id -g)" \
		-e HOME=/tmp \
		-v "$(CURDIR):/workspace" \
		-w /workspace \
		"$(IMAGE_TAG)" \
		sh -c '$(COMPUTE_UMMA_1SM_BIN) -h >/dev/null 2>&1; test $$? -eq 2 && echo "-h: exit 2 (rejected, OK)"'
	@echo "compute-umma-1sm-check: OK"

compute-umma-1sm-self-test: compute-umma-1sm-sass
	@if [ -z "$${BLACKWELL_GPU_INDEX:-}" ]; then \
		echo "ERROR: BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index."; \
		echo "       Example: BLACKWELL_GPU_INDEX=3 make compute-umma-1sm-self-test"; \
		echo "       This project never selects a GPU automatically."; \
		exit 2; \
	fi
	scripts/run_container.sh $(COMPUTE_UMMA_1SM_BIN) --self-test

compute-umma-1sm-smoke: compute-umma-1sm-sass
	@if [ -z "$${BLACKWELL_GPU_INDEX:-}" ]; then \
		echo "ERROR: BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index."; \
		echo "       Example: BLACKWELL_GPU_INDEX=3 make compute-umma-1sm-smoke"; \
		echo "       This project never selects a GPU automatically."; \
		exit 2; \
	fi
	@echo "== compute-umma-1sm-smoke: self-test =="
	scripts/run_container.sh $(COMPUTE_UMMA_1SM_BIN) --self-test
	@echo "== compute-umma-1sm-smoke: short run_kind=smoke measurement (NOT a final result) =="
	scripts/run_container.sh $(COMPUTE_UMMA_1SM_BIN) \
		--run-kind smoke --n 128 --depth 16 \
		--iterations 20 --warmup-iterations 5 --repetitions 3
	@echo "=============================================================================="
	@echo "The run_kind=smoke output above is a functional smoke check only. It is NOT a"
	@echo "final experimental result, is not a TFLOP/s or saturation claim, and must not"
	@echo "be cited as a performance number."
	@echo "=============================================================================="

# --- P2.2: 2-SM BF16 UMMA microbenchmark (tcgen05.mma, kind::f16, cta_group::2, ---
# --- one static two-CTA cluster) -----------------------------------------------
# compute-umma-2sm-build and compute-umma-2sm-sass never touch a GPU: they
# compile and disassemble inside the pinned, network-less, unprivileged image,
# the same secure pattern as compute-umma-1sm-build/sass. compute-umma-2sm-check
# is also GPU-free (Python syntax, the checker's own synthetic self-test,
# source-level contract checks) but depends on compute-umma-2sm-sass to also
# validate the real compiled cubin. compute-umma-2sm-self-test and
# compute-umma-2sm-smoke are the only P2.2 targets that execute on GPU; each
# requires an explicit BLACKWELL_GPU_INDEX and goes exclusively through
# scripts/run_container.sh. See src/compute/P2_2_PROTOCOL.md for the complete
# frozen contract; P2.2 is implemented, independently audited, functionally
# verified on GB300, and produces no publishable result.

compute-umma-2sm-build:
	@mkdir -p build/compute
	docker run --rm \
		--network none \
		--security-opt no-new-privileges \
		--cap-drop ALL \
		--user "$$(id -u):$$(id -g)" \
		-e HOME=/tmp \
		-v "$(CURDIR):/workspace" \
		-w /workspace \
		"$(IMAGE_TAG)" \
		nvcc -std=c++17 -O3 -lineinfo $(COMPUTE_UMMA_2SM_ARCH_FLAGS) \
			-o $(COMPUTE_UMMA_2SM_BIN) $(COMPUTE_UMMA_2SM_SRC)

compute-umma-2sm-sass: compute-umma-2sm-build
	@mkdir -p build/compute
	docker run --rm \
		--network none \
		--security-opt no-new-privileges \
		--cap-drop ALL \
		--user "$$(id -u):$$(id -g)" \
		-e HOME=/tmp \
		-v "$(CURDIR):/workspace" \
		-w /workspace \
		"$(IMAGE_TAG)" \
		python3 $(COMPUTE_UMMA_2SM_CHECKER) $(COMPUTE_UMMA_2SM_BIN) $(COMPUTE_UMMA_2SM_SASS)

compute-umma-2sm-check: compute-umma-2sm-sass
	@test -x $(COMPUTE_UMMA_2SM_CHECKER)
	python3 -m py_compile $(COMPUTE_UMMA_2SM_CHECKER)
	@rm -rf scripts/__pycache__
	python3 $(COMPUTE_UMMA_2SM_CHECKER) --self-test
	@test "$$(grep -oE 'UMMA_2SM_DEFINE_KERNEL\([0-9]+, [0-9]+\)' $(COMPUTE_UMMA_2SM_SRC) | wc -l | tr -d ' ')" -eq 12
	@echo "== P2.2 CLI contract: --help exits 0 and -h is rejected, both without GPU access =="
	docker run --rm \
		--network none \
		--security-opt no-new-privileges \
		--cap-drop ALL \
		--user "$$(id -u):$$(id -g)" \
		-e HOME=/tmp \
		-v "$(CURDIR):/workspace" \
		-w /workspace \
		"$(IMAGE_TAG)" \
		sh -c '$(COMPUTE_UMMA_2SM_BIN) --help >/dev/null && echo "--help: exit 0 (OK)"'
	docker run --rm \
		--network none \
		--security-opt no-new-privileges \
		--cap-drop ALL \
		--user "$$(id -u):$$(id -g)" \
		-e HOME=/tmp \
		-v "$(CURDIR):/workspace" \
		-w /workspace \
		"$(IMAGE_TAG)" \
		sh -c '$(COMPUTE_UMMA_2SM_BIN) -h >/dev/null 2>&1; test $$? -eq 2 && echo "-h: exit 2 (rejected, OK)"'
	@echo "compute-umma-2sm-check: OK"

compute-umma-2sm-self-test: compute-umma-2sm-sass
	@if [ -z "$${BLACKWELL_GPU_INDEX:-}" ]; then \
		echo "ERROR: BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index."; \
		echo "       Example: BLACKWELL_GPU_INDEX=3 make compute-umma-2sm-self-test"; \
		echo "       This project never selects a GPU automatically."; \
		exit 2; \
	fi
	scripts/run_container.sh $(COMPUTE_UMMA_2SM_BIN) --self-test

compute-umma-2sm-smoke: compute-umma-2sm-sass
	@if [ -z "$${BLACKWELL_GPU_INDEX:-}" ]; then \
		echo "ERROR: BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index."; \
		echo "       Example: BLACKWELL_GPU_INDEX=3 make compute-umma-2sm-smoke"; \
		echo "       This project never selects a GPU automatically."; \
		exit 2; \
	fi
	@echo "== compute-umma-2sm-smoke: self-test =="
	scripts/run_container.sh $(COMPUTE_UMMA_2SM_BIN) --self-test
	@echo "== compute-umma-2sm-smoke: short run_kind=smoke measurement (NOT a final result) =="
	scripts/run_container.sh $(COMPUTE_UMMA_2SM_BIN) \
		--run-kind smoke --n 128 --depth 16 \
		--iterations 20 --warmup-iterations 5 --repetitions 3
	@echo "=============================================================================="
	@echo "The run_kind=smoke output above is a functional smoke check only. It is NOT a"
	@echo "final experimental result, is not a TFLOP/s or saturation claim, and must not"
	@echo "be cited as a performance number."
	@echo "=============================================================================="

# --- P2.3: joint 1-SM/2-SM BF16 UMMA sweep infrastructure (exp02_umma_throughput) -
# compute-umma-sweep-plan and compute-umma-sweep-check never touch a GPU or
# Docker themselves: they only exercise scripts/run_exp02_umma_throughput.sh's
# and scripts/aggregate_exp02_umma_throughput.py's own GPU-free CLI paths and
# synthetic self-tests, plus the existing compute-umma-1sm-sass/
# compute-umma-2sm-sass build/SASS gates (which do run inside the pinned,
# network-less, unprivileged image, same secure pattern as every other
# GPU-free build/SASS target above; they still touch no GPU). compute-umma-
# sweep-smoke is the only P2.3 target that executes on GPU; it requires an
# explicit BLACKWELL_GPU_INDEX and reuses scripts/run_container.sh
# exclusively via the runner. compute-umma-sweep-smoke has no Make
# prerequisites of its own (audit repair): the runner already performs its
# own build/SASS gate internally (Step 3-4), so listing
# compute-umma-1sm-sass/compute-umma-2sm-sass as prerequisites here would let
# Make build and compile inside Docker before the recipe's own
# BLACKWELL_GPU_INDEX check ever ran. No P2.3 target invokes Nsight Compute or
# computes TFLOP/s, an empirical ceiling, 1-SM/2-SM speedup, scaling
# efficiency, or saturation; that remains explicit P2.4 work. P2.3 reuses
# the audited P2.1/P2.2 binaries and their existing CLIs completely
# unmodified and introduces no new CUDA kernel.

compute-umma-sweep-plan:
	$(EXP02_RUNNER) --print-plan

compute-umma-sweep-check: compute-umma-1sm-sass compute-umma-2sm-sass
	bash -n $(EXP02_RUNNER)
	@test -x $(EXP02_RUNNER)
	@test -x $(EXP02_AGGREGATOR)
	python3 -m py_compile $(EXP02_AGGREGATOR)
	@rm -rf scripts/__pycache__
	python3 $(EXP02_AGGREGATOR) --self-test
	@test "$$(python3 $(EXP02_AGGREGATOR) plan --format lines | wc -l | tr -d ' ')" -eq 24
	$(EXP02_RUNNER) --self-test
	@grep -Fq 'P2.3 | Sweep (≤24 configurations) | YES | YES | YES |' PLAN.md
	@grep -Fq 'P2.3 = YES / YES / YES' $(EXP02_PROTOCOL)
	@grep -Fq 'Phase 2 is closed' PLAN.md
	@grep -Fq 'Phase 2 is **closed**' $(EXP02_PROTOCOL)
	@grep -Fq 'Gate: Phase 2 gate passed.' PLAN.md
	@echo "compute-umma-sweep-check: OK"

compute-umma-sweep-smoke:
	@if [ -z "$${BLACKWELL_GPU_INDEX:-}" ]; then \
		echo "ERROR: BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index."; \
		echo "       Example: BLACKWELL_GPU_INDEX=3 make compute-umma-sweep-smoke"; \
		echo "       This project never selects a GPU automatically."; \
		exit 2; \
	fi
	@echo "== compute-umma-sweep-smoke: both binaries' self-tests, then all 24 smoke configurations =="
	$(EXP02_RUNNER) --run-kind smoke --iterations 20 --warmup-iterations 5 --repetitions 3
	@echo "=============================================================================="
	@echo "The run_kind=smoke output above is functional verification of the P2.3 sweep"
	@echo "infrastructure only. It is NOT a final experimental result, computes no"
	@echo "TFLOP/s, empirical ceiling, 1-SM/2-SM speedup, scaling efficiency, or"
	@echo "saturation, and must not be cited as a performance number."
	@echo "=============================================================================="

# --- P2.4: profiling and empirical BF16 UMMA per-SM ceiling candidate (exp02_umma_throughput_p24) -
# compute-umma-p24-plan and compute-umma-p24-check never touch a GPU, Docker,
# or the network: they only exercise scripts/run_exp02_umma_throughput_p24.sh's
# and scripts/analyze_exp02_umma_throughput_p24.py's own GPU-free CLI paths
# and synthetic/adversarial self-tests, plus the existing P2.1/P2.2/P2.3
# gates. compute-umma-p24-check (audit repair) has a REAL Make prerequisite
# on all three earlier GPU-free validation gates -- compute-umma-1sm-check,
# compute-umma-2sm-check, and compute-umma-sweep-check -- so a P2.1, P2.2, or
# P2.3 regression fails compute-umma-p24-check before any P2.4-specific check
# ever runs; each of those three gates is itself GPU-free (compilation and
# disassembly happen inside the pinned, network-less, unprivileged image,
# never against a selected GPU), so this adds no GPU selection, no
# benchmark, and no NCU profiling. Make only ever executes each of those
# three targets' recipe once per invocation (a target already brought up to
# date earlier in the same `make` run, e.g. compute-umma-1sm-sass via
# compute-umma-1sm-check, is not rebuilt again when compute-umma-sweep-check
# lists it too), so this reuses the existing targets rather than duplicating
# their commands. compute-umma-p24-pilot and compute-umma-p24-profile are
# the only P2.4 targets that execute on GPU; each requires an explicit
# BLACKWELL_GPU_INDEX, P2_4_CAMPAIGN_ID (a canonical UTC timestamp), and
# P2_4_PREFLIGHT_SUMMARY, and never selects a GPU automatically. Neither has
# a Make prerequisite of its own (mirrors compute-umma-sweep-smoke's own
# audited reasoning): scripts/run_exp02_umma_throughput_p24.sh checks every
# mandatory environment variable itself, before delegating to the unmodified
# P2.3 runner (which performs its own build/SASS gate) or touching NCU, so a
# Make prerequisite here would let Docker/compilation run before that check.
# compute-umma-p24-analyze is GPU-free and validates/analyzes an
# already-COMPLETE P2.4 campaign; it requires only P2_4_CAMPAIGN_ID.

compute-umma-p24-plan:
	$(EXP02_P24_RUNNER) --print-plan

compute-umma-p24-check: compute-umma-1sm-check compute-umma-2sm-check compute-umma-sweep-check
	@echo "== P2.4 gate: P2.1/P2.2/P2.3 GPU-free gates passed (compute-umma-1sm-check, compute-umma-2sm-check, compute-umma-sweep-check) =="
	bash -n $(EXP02_P24_RUNNER)
	@test -x $(EXP02_P24_RUNNER)
	@test -x $(EXP02_P24_ANALYZER)
	@test -x $(EXP02_P24_SAFE_CAPTURE)
	@test -x $(EXP02_P24_NCU_BRIDGE)
	python3 -m py_compile $(EXP02_P24_ANALYZER) $(EXP02_P24_SAFE_CAPTURE) $(EXP02_P24_NCU_BRIDGE)
	@rm -rf scripts/__pycache__
	python3 $(EXP02_P24_ANALYZER) --self-test
	python3 $(EXP02_P24_SAFE_CAPTURE) --self-test
	python3 $(EXP02_P24_NCU_BRIDGE) --self-test
	@test "$$(python3 $(EXP02_AGGREGATOR) plan --format lines | wc -l | tr -d ' ')" -eq 24
	@test "$$(python3 $(EXP02_P24_ANALYZER) plan --format lines | wc -l | tr -d ' ')" -eq 24
	$(EXP02_P24_RUNNER) --self-test
	@echo "compute-umma-p24-check: OK"

compute-umma-p24-pilot:
	@if [ -z "$${BLACKWELL_GPU_INDEX:-}" ]; then \
		echo "ERROR: BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index."; \
		echo "       Example: BLACKWELL_GPU_INDEX=3 make compute-umma-p24-pilot"; \
		echo "       This project never selects a GPU automatically."; \
		exit 2; \
	fi
	@if [ -z "$${P2_4_CAMPAIGN_ID:-}" ]; then \
		echo "ERROR: P2_4_CAMPAIGN_ID must be set explicitly to a canonical UTC timestamp"; \
		echo "       YYYYMMDDTHHMMSSZ. Example: P2_4_CAMPAIGN_ID=$$(date -u +%Y%m%dT%H%M%SZ)"; \
		exit 2; \
	fi
	@if [ -z "$${P2_4_PREFLIGHT_SUMMARY:-}" ]; then \
		echo "ERROR: P2_4_PREFLIGHT_SUMMARY must be set explicitly to a fresh preflight"; \
		echo "       summary.json path (see 'make preflight')."; \
		exit 2; \
	fi
	$(EXP02_P24_RUNNER) --pilot

compute-umma-p24-profile:
	@if [ -z "$${BLACKWELL_GPU_INDEX:-}" ]; then \
		echo "ERROR: BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index."; \
		echo "       Example: BLACKWELL_GPU_INDEX=3 make compute-umma-p24-profile"; \
		echo "       This project never selects a GPU automatically."; \
		exit 2; \
	fi
	@if [ -z "$${P2_4_CAMPAIGN_ID:-}" ]; then \
		echo "ERROR: P2_4_CAMPAIGN_ID must be set explicitly to the same canonical UTC"; \
		echo "       timestamp used for 'make compute-umma-p24-pilot'."; \
		exit 2; \
	fi
	@if [ -z "$${P2_4_PREFLIGHT_SUMMARY:-}" ]; then \
		echo "ERROR: P2_4_PREFLIGHT_SUMMARY must be set explicitly to a fresh preflight"; \
		echo "       summary.json path (see 'make preflight')."; \
		exit 2; \
	fi
	$(EXP02_P24_RUNNER) --profile

compute-umma-p24-analyze:
	@if [ -z "$${P2_4_CAMPAIGN_ID:-}" ]; then \
		echo "ERROR: P2_4_CAMPAIGN_ID must be set explicitly to the campaign to analyze."; \
		exit 2; \
	fi
	python3 $(EXP02_P24_ANALYZER) analyze \
		--campaign-dir $(EXP02_P24_RAW_ROOT)/$${P2_4_CAMPAIGN_ID} \
		--analyzed-at-utc "$$(date -u +%Y%m%dT%H%M%SZ)"

# --- P3.1: pinned official CuTe DSL dense GEMM example ----------------------
# P3.1 executes NVIDIA's own example unmodified from the pinned /opt/cutlass
# checkout. This repository owns no GEMM source, adds no wrapper, no persistent
# variant, no 2-CTA instruction, no cuBLASLt baseline, no sweep, no Nsight
# Compute, and writes no result file; see src/gemm/P3_1_PROTOCOL.md.
#
# gemm-cutedsl-p31-check never touches a GPU, the network, or elevated
# privileges: it runs inside the pinned image (--network none, --cap-drop ALL,
# no-new-privileges, no --gpus, the invoking user, no repository mount) and
# fails closed unless /opt/cutlass exists, its HEAD is exactly the pinned
# CUTLASS_COMMIT, the checkout has no tracked or untracked modification, the
# example is a non-symlink regular file whose Git blob SHA and SHA-256 match
# the pinned values, CuTe DSL and PyTorch report the pinned versions,
# importlib.metadata reports the pinned cuda-python/cuda-bindings
# distributions, `python3 -m pip check` finds no broken requirement (never
# suppressed, filtered, or downgraded to a warning), and the example's own
# --help exits successfully -- with every frozen option present -- without a
# device. Every expected value is passed in from VERSIONS.env (global) or
# PHASE3_VERSIONS.env (Phase 3); none is duplicated as an unconnected constant
# here. /opt/cutlass is a root-owned checkout inside the image while the
# container runs as the invoking user, so each Git query carries an explicit,
# per-invocation -c safe.directory for that one path; nothing is ever written
# to the checkout.
#
# gemm-cutedsl-p31-smoke is the only P3.1 target that executes on GPU. Its
# first recipe line validates BLACKWELL_GPU_INDEX before Docker, any build, or
# any check can start, which is why it deliberately has no Make prerequisite
# (same audited reasoning as compute-umma-sweep-smoke). It then goes
# exclusively through scripts/run_container.sh, which alone owns GPU selection,
# UUID resolution, and the idle-device proof -- this target never calls Docker
# or --gpus itself. Inside that same GPU container it re-checks the upstream
# commit and source SHA-256 immediately before exec'ing the frozen command,
# never passes the upstream skip-reference-checking flag (reference validation
# is mandatory and is performed by the unchanged official example), and
# preserves the official program's exit code.

gemm-cutedsl-p31-check:
	docker run --rm \
		--network none \
		--security-opt no-new-privileges \
		--cap-drop ALL \
		--user "$$(id -u):$$(id -g)" \
		-e HOME=/tmp \
		-e CUTLASS_COMMIT="$(CUTLASS_COMMIT)" \
		-e CUTEDSL_VERSION="$(CUTEDSL_VERSION)" \
		-e PYTORCH_VERSION="$(PYTORCH_VERSION)" \
		-e PYTORCH_CUDA_VERSION="$(PYTORCH_CUDA_VERSION)" \
		-e CUDA_PYTHON_VERSION="$(CUDA_PYTHON_VERSION)" \
		-e CUDA_BINDINGS_VERSION="$(CUDA_BINDINGS_VERSION)" \
		-e P31_EXAMPLE="$(GEMM_P31_EXAMPLE)" \
		-e P31_EXAMPLE_GIT_BLOB="$(CUTEDSL_P31_EXAMPLE_GIT_BLOB)" \
		-e P31_EXAMPLE_SHA256="$(CUTEDSL_P31_EXAMPLE_SHA256)" \
		"$(IMAGE_TAG)" \
		bash -c 'set -euo pipefail; \
			fail() { echo "gemm-cutedsl-p31-check: FAIL: $$*" >&2; exit 1; }; \
			[ -d /opt/cutlass ] || fail "/opt/cutlass is missing"; \
			head_commit="$$(git -c safe.directory=/opt/cutlass -C /opt/cutlass rev-parse HEAD)" \
				|| fail "cannot read the /opt/cutlass HEAD commit"; \
			[ "$$head_commit" = "$$CUTLASS_COMMIT" ] \
				|| fail "/opt/cutlass HEAD $$head_commit != pinned $$CUTLASS_COMMIT"; \
			dirty="$$(git -c safe.directory=/opt/cutlass -C /opt/cutlass status --porcelain --untracked-files=all)" \
				|| fail "cannot read the /opt/cutlass working tree status"; \
			[ -z "$$dirty" ] || fail "/opt/cutlass has tracked or untracked modifications"; \
			[ ! -L "$$P31_EXAMPLE" ] || fail "$$P31_EXAMPLE is a symlink"; \
			[ -f "$$P31_EXAMPLE" ] || fail "$$P31_EXAMPLE is not a regular file"; \
			blob="$$(git -c safe.directory=/opt/cutlass -C /opt/cutlass hash-object -- "$$P31_EXAMPLE")" \
				|| fail "cannot compute the Git blob SHA of $$P31_EXAMPLE"; \
			[ "$$blob" = "$$P31_EXAMPLE_GIT_BLOB" ] \
				|| fail "Git blob $$blob != pinned $$P31_EXAMPLE_GIT_BLOB"; \
			sha="$$(sha256sum "$$P31_EXAMPLE" | cut -d" " -f1)" \
				|| fail "cannot compute the SHA-256 of $$P31_EXAMPLE"; \
			[ "$$sha" = "$$P31_EXAMPLE_SHA256" ] \
				|| fail "SHA-256 $$sha != pinned $$P31_EXAMPLE_SHA256"; \
			echo "upstream provenance OK: commit $$head_commit"; \
			echo "                       blob   $$blob"; \
			echo "                       sha256 $$sha"; \
			python3 -c "import os, cutlass, torch; \
				ce = os.environ[\"CUTEDSL_VERSION\"]; \
				assert cutlass.__version__ == ce, f\"CuTeDSL {cutlass.__version__} != pinned {ce}\"; \
				pe = os.environ[\"PYTORCH_VERSION\"]; \
				assert torch.__version__ == pe, f\"torch {torch.__version__} != pinned {pe}\"; \
				pc = os.environ[\"PYTORCH_CUDA_VERSION\"]; \
				assert torch.version.cuda == pc, f\"torch CUDA {torch.version.cuda} != pinned {pc}\"; \
				print(\"versions OK: cutedsl\", ce, \"torch\", pe, \"torch-cuda\", pc)"; \
			python3 -c "import os; from importlib.metadata import version; \
				expected = {\"cuda-python\": os.environ[\"CUDA_PYTHON_VERSION\"], \
					\"cuda-bindings\": os.environ[\"CUDA_BINDINGS_VERSION\"]}; \
				installed = {name: version(name) for name in expected}; \
				assert installed == expected, f\"installed distributions {installed} != pinned {expected}\"; \
				print(\"cuda distributions OK:\", installed)"; \
			echo "== pip check: the dependency graph must be consistent =="; \
			python3 -m pip check; \
			help_text="$$(python3 "$$P31_EXAMPLE" --help)" \
				|| fail "the official example --help did not exit successfully"; \
			for opt in --mnkl --ab_dtype --c_dtype --acc_dtype --a_major --b_major \
				--c_major --mma_tiler_mn --cluster_shape_mn --use_tma_store \
				--warmup_iterations --iterations; do \
				case "$$help_text" in \
					*"$$opt"*) ;; \
					*) fail "the official example does not offer $$opt";; \
				esac; \
			done; \
			echo "example --help OK (GPU-free); every frozen option is present"'
	@echo "gemm-cutedsl-p31-check: OK"

gemm-cutedsl-p31-smoke:
	@if [ -z "$${BLACKWELL_GPU_INDEX:-}" ]; then \
		echo "ERROR: BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index."; \
		echo "       Example: BLACKWELL_GPU_INDEX=3 make gemm-cutedsl-p31-smoke"; \
		echo "       This project never selects a GPU automatically."; \
		exit 2; \
	fi
	status=0; \
	scripts/run_container.sh bash -c 'set -euo pipefail; \
		head_commit="$$(git -c safe.directory=/opt/cutlass -C /opt/cutlass rev-parse HEAD)"; \
		[ "$$head_commit" = "$(CUTLASS_COMMIT)" ] \
			|| { echo "gemm-cutedsl-p31-smoke: FAIL: /opt/cutlass HEAD $$head_commit != pinned $(CUTLASS_COMMIT)" >&2; exit 1; }; \
		sha="$$(sha256sum "$(GEMM_P31_EXAMPLE)" | cut -d" " -f1)"; \
		[ "$$sha" = "$(CUTEDSL_P31_EXAMPLE_SHA256)" ] \
			|| { echo "gemm-cutedsl-p31-smoke: FAIL: SHA-256 $$sha != pinned $(CUTEDSL_P31_EXAMPLE_SHA256)" >&2; exit 1; }; \
		echo "gemm-cutedsl-p31-smoke: upstream re-checked in this GPU container: commit $$head_commit sha256 $$sha"; \
		exec python3 "$(GEMM_P31_EXAMPLE)" \
			--mnkl 256,256,512,1 \
			--ab_dtype BFloat16 --c_dtype Float32 --acc_dtype Float32 \
			--a_major k --b_major k --c_major n \
			--mma_tiler_mn 128,128 --cluster_shape_mn 1,1 \
			--use_tma_store \
			--warmup_iterations 0 --iterations 1' || status=$$?; \
	echo "=============================================================================="; \
	echo "P3.1 FUNCTIONAL SMOKE CHECK ONLY -- NOT A PERFORMANCE RESULT."; \
	echo "The output above comes from NVIDIA's unmodified official CuTe DSL example,"; \
	echo "run once at (M,N,K,L)=(256,256,512,1) with mandatory reference validation."; \
	echo "Any timing the example computed internally is discarded: P3.1 emits no"; \
	echo "TFLOP/s, no comparison, no cuBLASLt baseline, and no publishable result."; \
	echo "=============================================================================="; \
	exit $$status

# --- P3.2: one-shape CuTe DSL GEMM wrapper ----------------------------------
# P3.2 adds a thin, repository-owned orchestration wrapper around the very same
# pinned upstream example, executing one frozen BF16 shape and separating
# compile_time_ms, first_launch_ms, and kernel_time_ms -- which the upstream
# run() function fuses into a single number and therefore cannot provide. It
# still vendors no NVIDIA GEMM source, adds no key to either version contract,
# introduces no cuBLASLt baseline, no persistent scheduler, no 2-CTA MMA group,
# no other shape, no sweep, no autotuning, no Nsight Compute, no campaign
# directory, and no result file; see src/gemm/P3_2_PROTOCOL.md.
#
# gemm-cutedsl-p32-check never touches a GPU, the network, or elevated
# privileges. It runs the existing P3.1 gate first (also GPU-free and
# network-free, and left completely intact), then runs inside the pinned image
# with --network none, --cap-drop ALL, no-new-privileges, the invoking UID/GID,
# no --gpus, and the repository mounted READ-ONLY -- a checker must not be able
# to modify what it checks, which is also why PYTHONPYCACHEPREFIX sends every
# byte-compilation artefact to the container's own /tmp. Inside, it re-verifies
# the upstream commit, checkout cleanliness, regular-file identity, Git blob
# SHA, SHA-256, the CuTe DSL/PyTorch/cuda-python/cuda-bindings pins and a
# consistent dependency graph, then compiles both P3.2 files and runs the
# wrapper's GPU-free --help and --self-test plus the checker and its own
# self-test. Every expected value is passed in from VERSIONS.env (global) or
# PHASE3_VERSIONS.env (Phase 3); none is duplicated as an unconnected constant.
#
# gemm-cutedsl-p32-smoke is the only P3.2 target that executes on GPU. Its
# first recipe line validates BLACKWELL_GPU_INDEX before Docker, any build, or
# any check can start, which is why it deliberately has no Make prerequisite
# (same audited reasoning as gemm-cutedsl-p31-smoke). It then goes exclusively
# through scripts/run_container.sh, which alone owns GPU selection, UUID
# resolution, and the idle-device proof -- this target never calls Docker or
# --gpus itself. Inside that same GPU container it re-checks the upstream commit
# and source SHA-256 immediately before exec'ing the wrapper, runs exactly the
# frozen one-shape configuration, preserves the wrapper's exit code, and prints
# an explicit stderr notice that the emitted timings are non-publishable
# functional evidence. Correctness is mandatory and cannot be disabled: the
# wrapper has no option for it and emits no row unless the full check passed.

gemm-cutedsl-p32-check: gemm-cutedsl-p31-check
	docker run --rm \
		--network none \
		--security-opt no-new-privileges \
		--cap-drop ALL \
		--user "$$(id -u):$$(id -g)" \
		-e HOME=/tmp \
		-e PYTHONPYCACHEPREFIX=/tmp/p32-pycache \
		-e CUTLASS_COMMIT="$(CUTLASS_COMMIT)" \
		-e CUTEDSL_VERSION="$(CUTEDSL_VERSION)" \
		-e PYTORCH_VERSION="$(PYTORCH_VERSION)" \
		-e PYTORCH_CUDA_VERSION="$(PYTORCH_CUDA_VERSION)" \
		-e CUDA_PYTHON_VERSION="$(CUDA_PYTHON_VERSION)" \
		-e CUDA_BINDINGS_VERSION="$(CUDA_BINDINGS_VERSION)" \
		-e P31_EXAMPLE="$(GEMM_P31_EXAMPLE)" \
		-e P31_EXAMPLE_GIT_BLOB="$(CUTEDSL_P31_EXAMPLE_GIT_BLOB)" \
		-e P31_EXAMPLE_SHA256="$(CUTEDSL_P31_EXAMPLE_SHA256)" \
		-e P32_WRAPPER="$(GEMM_P32_WRAPPER)" \
		-e P32_CHECKER="$(GEMM_P32_CHECKER)" \
		-v "$(CURDIR):/workspace:ro" \
		-w /workspace \
		"$(IMAGE_TAG)" \
		bash -c 'set -euo pipefail; \
			fail() { echo "gemm-cutedsl-p32-check: FAIL: $$*" >&2; exit 1; }; \
			[ -d /opt/cutlass ] || fail "/opt/cutlass is missing"; \
			head_commit="$$(git -c safe.directory=/opt/cutlass -C /opt/cutlass rev-parse HEAD)" \
				|| fail "cannot read the /opt/cutlass HEAD commit"; \
			[ "$$head_commit" = "$$CUTLASS_COMMIT" ] \
				|| fail "/opt/cutlass HEAD $$head_commit != pinned $$CUTLASS_COMMIT"; \
			dirty="$$(git -c safe.directory=/opt/cutlass -C /opt/cutlass status --porcelain --untracked-files=all)" \
				|| fail "cannot read the /opt/cutlass working tree status"; \
			[ -z "$$dirty" ] || fail "/opt/cutlass has tracked or untracked modifications"; \
			[ ! -L "$$P31_EXAMPLE" ] || fail "$$P31_EXAMPLE is a symlink"; \
			[ -f "$$P31_EXAMPLE" ] || fail "$$P31_EXAMPLE is not a regular file"; \
			blob="$$(git -c safe.directory=/opt/cutlass -C /opt/cutlass hash-object -- "$$P31_EXAMPLE")" \
				|| fail "cannot compute the Git blob SHA of $$P31_EXAMPLE"; \
			[ "$$blob" = "$$P31_EXAMPLE_GIT_BLOB" ] \
				|| fail "Git blob $$blob != pinned $$P31_EXAMPLE_GIT_BLOB"; \
			sha="$$(sha256sum "$$P31_EXAMPLE" | cut -d" " -f1)" \
				|| fail "cannot compute the SHA-256 of $$P31_EXAMPLE"; \
			[ "$$sha" = "$$P31_EXAMPLE_SHA256" ] \
				|| fail "SHA-256 $$sha != pinned $$P31_EXAMPLE_SHA256"; \
			echo "upstream provenance OK: commit $$head_commit"; \
			echo "                       blob   $$blob"; \
			echo "                       sha256 $$sha"; \
			python3 -c "import os, cutlass, torch; \
				ce = os.environ[\"CUTEDSL_VERSION\"]; \
				assert cutlass.__version__ == ce, f\"CuTeDSL {cutlass.__version__} != pinned {ce}\"; \
				pe = os.environ[\"PYTORCH_VERSION\"]; \
				assert torch.__version__ == pe, f\"torch {torch.__version__} != pinned {pe}\"; \
				pc = os.environ[\"PYTORCH_CUDA_VERSION\"]; \
				assert torch.version.cuda == pc, f\"torch CUDA {torch.version.cuda} != pinned {pc}\"; \
				print(\"versions OK: cutedsl\", ce, \"torch\", pe, \"torch-cuda\", pc)"; \
			python3 -c "import os; from importlib.metadata import version; \
				expected = {\"cuda-python\": os.environ[\"CUDA_PYTHON_VERSION\"], \
					\"cuda-bindings\": os.environ[\"CUDA_BINDINGS_VERSION\"]}; \
				installed = {name: version(name) for name in expected}; \
				assert installed == expected, f\"installed distributions {installed} != pinned {expected}\"; \
				print(\"cuda distributions OK:\", installed)"; \
			echo "== pip check: the dependency graph must be consistent =="; \
			python3 -m pip check; \
			echo "== P3.2 python syntax =="; \
			python3 -m py_compile "$$P32_WRAPPER" "$$P32_CHECKER"; \
			echo "== P3.2 wrapper --help and --self-test are GPU-free =="; \
			python3 "$$P32_WRAPPER" --help > /dev/null \
				|| fail "the wrapper --help did not exit successfully"; \
			python3 "$$P32_WRAPPER" --self-test \
				|| fail "the wrapper GPU-free self-test failed"; \
			echo "== P3.2 checker self-test and full frozen-contract check =="; \
			python3 "$$P32_CHECKER" --self-test \
				|| fail "the checker self-test failed"; \
			python3 "$$P32_CHECKER" /workspace \
				|| fail "the P3.2 frozen-contract check failed"; \
			echo "P3.2 GPU-free contract OK (no GPU was used or required)"'
	@echo "gemm-cutedsl-p32-check: OK"

gemm-cutedsl-p32-smoke:
	@if [ -z "$${BLACKWELL_GPU_INDEX:-}" ]; then \
		echo "ERROR: BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index." >&2; \
		echo "       Example: BLACKWELL_GPU_INDEX=3 make gemm-cutedsl-p32-smoke" >&2; \
		echo "       This project never selects a GPU automatically." >&2; \
		exit 2; \
	fi
	@status=0; \
	RUN_CONTAINER_STDOUT_IS_DATA=1 scripts/run_container.sh bash -c 'set -euo pipefail; \
		head_commit="$$(git -c safe.directory=/opt/cutlass -C /opt/cutlass rev-parse HEAD)"; \
		[ "$$head_commit" = "$(CUTLASS_COMMIT)" ] \
			|| { echo "gemm-cutedsl-p32-smoke: FAIL: /opt/cutlass HEAD $$head_commit != pinned $(CUTLASS_COMMIT)" >&2; exit 1; }; \
		sha="$$(sha256sum "$(GEMM_P31_EXAMPLE)" | cut -d" " -f1)"; \
		[ "$$sha" = "$(CUTEDSL_P31_EXAMPLE_SHA256)" ] \
			|| { echo "gemm-cutedsl-p32-smoke: FAIL: SHA-256 $$sha != pinned $(CUTEDSL_P31_EXAMPLE_SHA256)" >&2; exit 1; }; \
		echo "gemm-cutedsl-p32-smoke: upstream re-checked in this GPU container: commit $$head_commit sha256 $$sha" >&2; \
		exec python3 $(GEMM_P32_WRAPPER) \
			--warmup-iterations 2 \
			--iterations 10' || status=$$?; \
	echo "==============================================================================" >&2; \
	echo "P3.2 FUNCTIONAL VERIFICATION ONLY -- NOT AN EXPERIMENTAL RESULT." >&2; \
	echo "Any CSV row on stdout is P3.2 infrastructure evidence: one frozen shape," >&2; \
	echo "(M,N,K,L)=(4096,4096,4096,1), 2 warm-ups and 10 measured launches, with hot" >&2; \
	echo "reused operands. compile_time_ms, first_launch_ms, and kernel_time_ms are" >&2; \
	echo "NON-PUBLISHABLE diagnostic fields; every row carries publishable=false." >&2; \
	echo "P3.2 computes no TFLOP/s, no speedup, no efficiency, and no comparison. The" >&2; \
	echo "P3.3 cuBLASLt baseline now exists but is a separate unit: no P3.2-versus-P3.3" >&2; \
	echo "comparison exists anywhere, and P3.5 owns that comparison." >&2; \
	if [ "$$status" -eq 0 ]; then \
		echo "P3.2 smoke completed: correctness passed before warm-up and steady-state timing." >&2; \
	else \
		echo "P3.2 smoke FAILED with exit status $$status: no CSV header and no CSV row" >&2; \
		echo "were emitted, and no result may be read from this run." >&2; \
	fi; \
	echo "==============================================================================" >&2; \
	exit $$status

# --- P3.3: equivalent cuBLASLt baseline --------------------------------------
# P3.3 answers exactly one question that P3.2 cannot: what does the vendor
# library do with the very same problem, on the very same bytes? It runs the
# same frozen BF16 geometry, built by the same pinned cutlass.torch.matrix
# factory with the same seed and the same call order, through a direct
# cublasLtMatmul call issued by a small repository-owned C-ABI bridge. The
# bridge owns no GEMM kernel, copies no NVIDIA source, prints nothing, and lets
# no C++ exception cross the ABI; cuBLASLt itself already ships inside the
# pinned CUDA 13.1 image, so no package is added, no image changes, and no key
# is added to either version contract -- the library's own runtime version is
# read with cublasLtGetVersion().
#
# The algorithm policy is fixed and never autotuned: a 64 MiB workspace limit,
# 32 requested heuristic results, CUBLASLT_SEARCH_BEST_FIT, the first result
# whose state is CUBLAS_STATUS_SUCCESS, re-validated with
# cublasLtMatmulAlgoCheck, rejected if it needs more than the fixed limit, and
# given exactly the workspace it asks for. No candidate is ever executed for
# comparison, and there is no retry with another layout, type, compute mode,
# workspace limit, or API: an unsupported configuration fails with a
# diagnostic and emits no CSV.
#
# P3.3 introduces no persistent scheduler, no 2-CTA MMA group, no additional
# shape, no sweep, no autotuning, no Nsight Compute, no SASS analysis of
# proprietary kernels, no campaign directory, and no result file, and it makes
# no CuTe-versus-cuBLASLt comparison of any kind -- that is P3.5's job. See
# src/gemm/P3_3_PROTOCOL.md.
#
# gemm-cublaslt-p33-check never touches a GPU, the network, or elevated
# privileges. It runs the existing P3.2 gate first (which itself runs the P3.1
# gate, both GPU-free and network-free, and both left completely intact), then
# runs inside the pinned image with --network none, --cap-drop ALL,
# no-new-privileges, the invoking UID/GID, no --gpus, and the repository mounted
# READ-ONLY -- a checker must not be able to modify what it checks, which is why
# PYTHONPYCACHEPREFIX and the bridge build output both go to the container's own
# /tmp. Inside, it compiles the bridge with the pinned toolchain, inspects the
# resulting shared object with nm/readelf to prove the measured path references
# cublasLtMatmul and references no fallback GEMM entry point, and runs both
# GPU-free self-tests plus the full contract check.
#
# gemm-cublaslt-p33-smoke is the only P3.3 target that executes on GPU. Its
# first recipe line validates BLACKWELL_GPU_INDEX before Docker, any
# compilation, or any other work can start, which is why it deliberately has no
# Make prerequisite (same audited reasoning as gemm-cutedsl-p31-smoke and
# gemm-cutedsl-p32-smoke). It then goes exclusively through
# scripts/run_container.sh, which alone owns GPU selection, UUID resolution, and
# the idle-device proof -- this target never calls Docker or --gpus itself.
# Inside that same GPU container it compiles the bridge into private /tmp,
# re-checks the upstream commit and source SHA-256, runs exactly the frozen
# configuration, preserves the wrapper's exit code, and prints an explicit
# stderr notice. Correctness is mandatory and cannot be disabled: the wrapper
# has no option for it and emits no row unless the full check passed.

gemm-cublaslt-p33-check: gemm-cutedsl-p32-check
	docker run --rm \
		--network none \
		--security-opt no-new-privileges \
		--cap-drop ALL \
		--user "$$(id -u):$$(id -g)" \
		-e HOME=/tmp \
		-e PYTHONPYCACHEPREFIX=/tmp/p33-pycache \
		-e CUTLASS_COMMIT="$(CUTLASS_COMMIT)" \
		-e CUDA_SHORT_VERSION="$(CUDA_SHORT_VERSION)" \
		-e P31_EXAMPLE="$(GEMM_P31_EXAMPLE)" \
		-e P31_EXAMPLE_SHA256="$(CUTEDSL_P31_EXAMPLE_SHA256)" \
		-e P33_WRAPPER="$(GEMM_P33_WRAPPER)" \
		-e P33_BRIDGE="$(GEMM_P33_BRIDGE)" \
		-e P33_CHECKER="$(GEMM_P33_CHECKER)" \
		-e P33_BRIDGE_DIR="$(GEMM_P33_BRIDGE_DIR)" \
		-e P33_BRIDGE_LIB="$(GEMM_P33_BRIDGE_LIB)" \
		-e P33_ARCH_FLAGS="$(GEMM_P33_ARCH_FLAGS)" \
		-v "$(CURDIR):/workspace:ro" \
		-w /workspace \
		"$(IMAGE_TAG)" \
		bash -c 'set -euo pipefail; \
			fail() { echo "gemm-cublaslt-p33-check: FAIL: $$*" >&2; exit 1; }; \
			echo "== the pinned CUDA toolkit that must supply cuBLASLt =="; \
			nvcc_version="$$(nvcc --version | sed -n "s/.*release \([0-9.]*\).*/\1/p")"; \
			[ "$$nvcc_version" = "$$CUDA_SHORT_VERSION" ] \
				|| fail "nvcc reports CUDA $$nvcc_version, pinned is $$CUDA_SHORT_VERSION"; \
			echo "nvcc CUDA $$nvcc_version (cuBLASLt ships with it; no package is added)"; \
			echo "== upstream provenance is still the pinned P3.1 file =="; \
			head_commit="$$(git -c safe.directory=/opt/cutlass -C /opt/cutlass rev-parse HEAD)" \
				|| fail "cannot read the /opt/cutlass HEAD commit"; \
			[ "$$head_commit" = "$$CUTLASS_COMMIT" ] \
				|| fail "/opt/cutlass HEAD $$head_commit != pinned $$CUTLASS_COMMIT"; \
			sha="$$(sha256sum "$$P31_EXAMPLE" | cut -d" " -f1)" \
				|| fail "cannot compute the SHA-256 of $$P31_EXAMPLE"; \
			[ "$$sha" = "$$P31_EXAMPLE_SHA256" ] \
				|| fail "SHA-256 $$sha != pinned $$P31_EXAMPLE_SHA256"; \
			echo "upstream provenance OK: commit $$head_commit sha256 $$sha"; \
			echo "== compile the C-ABI cuBLASLt bridge into container-private /tmp =="; \
			mkdir -p "$$P33_BRIDGE_DIR"; \
			nvcc -std=c++17 -O3 -lineinfo \
				-Xcompiler -fPIC -shared \
				$$P33_ARCH_FLAGS \
				-o "$$P33_BRIDGE_LIB" "$$P33_BRIDGE" \
				-lcublasLt -lcudart \
				|| fail "the cuBLASLt bridge did not compile"; \
			echo "bridge compiled: $$P33_BRIDGE_LIB"; \
			echo "== the shared object must call cublasLtMatmul and no fallback GEMM API =="; \
			nm -D --defined-only "$$P33_BRIDGE_LIB" > /tmp/p33-defined.txt \
				|| fail "cannot read the defined symbols of the bridge"; \
			nm -D -u "$$P33_BRIDGE_LIB" > /tmp/p33-undefined.txt \
				|| fail "cannot read the undefined symbols of the bridge"; \
			readelf -d "$$P33_BRIDGE_LIB" > /tmp/p33-dynamic.txt \
				|| fail "cannot read the dynamic section of the bridge"; \
			grep -qw "cublasLtMatmul" /tmp/p33-undefined.txt \
				|| fail "the measured path does not reference cublasLtMatmul"; \
			grep -qw "cublasLtMatmulAlgoCheck" /tmp/p33-undefined.txt \
				|| fail "the bridge never validates the selected algorithm"; \
			grep -qw "cublasLtMatmulAlgoGetHeuristic" /tmp/p33-undefined.txt \
				|| fail "the bridge never queries the vendor heuristic"; \
			for symbol in p33_plan_create p33_plan_execute p33_plan_destroy \
					p33_cublaslt_version p33_plan_info_size p33_bridge_abi_version; do \
				grep -qw "$$symbol" /tmp/p33-defined.txt \
					|| fail "the bridge does not export $$symbol"; \
			done; \
			for forbidden in cublasGemmEx cublasGemmStridedBatchedEx cublasGemmBatchedEx \
					cublasSgemm cublasHgemm cublasLtMatmulAlgoGetIds cublasLtMatmulAlgoInit; do \
				if grep -qw "$$forbidden" /tmp/p33-defined.txt /tmp/p33-undefined.txt; then \
					fail "the bridge references the forbidden fallback API $$forbidden"; \
				fi; \
			done; \
			grep -q "libcublasLt.so" /tmp/p33-dynamic.txt \
				|| fail "the bridge is not linked against libcublasLt"; \
			grep -q "libcudart.so" /tmp/p33-dynamic.txt \
				|| fail "the bridge is not linked against libcudart"; \
			echo "ELF inspection OK: cublasLtMatmul present, no fallback GEMM API present"; \
			echo "== P3.3 python syntax =="; \
			python3 -m py_compile "$$P33_WRAPPER" "$$P33_CHECKER"; \
			echo "== P3.3 wrapper --help and --self-test are GPU-free =="; \
			python3 "$$P33_WRAPPER" --help > /dev/null \
				|| fail "the wrapper --help did not exit successfully"; \
			python3 "$$P33_WRAPPER" --self-test \
				|| fail "the wrapper GPU-free self-test failed"; \
			echo "== P3.3 checker self-test and full frozen-contract check =="; \
			python3 "$$P33_CHECKER" --self-test \
				|| fail "the checker self-test failed"; \
			python3 "$$P33_CHECKER" /workspace \
				|| fail "the P3.3 frozen-contract check failed"; \
			echo "P3.3 GPU-free contract OK (no GPU was used or required)"'
	@echo "gemm-cublaslt-p33-check: OK"

gemm-cublaslt-p33-smoke:
	@if [ -z "$${BLACKWELL_GPU_INDEX:-}" ]; then \
		echo "ERROR: BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index." >&2; \
		echo "       Example: BLACKWELL_GPU_INDEX=3 make gemm-cublaslt-p33-smoke" >&2; \
		echo "       This project never selects a GPU automatically." >&2; \
		exit 2; \
	fi
	@status=0; \
	RUN_CONTAINER_STDOUT_IS_DATA=1 scripts/run_container.sh bash -c 'set -euo pipefail; \
		head_commit="$$(git -c safe.directory=/opt/cutlass -C /opt/cutlass rev-parse HEAD)"; \
		[ "$$head_commit" = "$(CUTLASS_COMMIT)" ] \
			|| { echo "gemm-cublaslt-p33-smoke: FAIL: /opt/cutlass HEAD $$head_commit != pinned $(CUTLASS_COMMIT)" >&2; exit 1; }; \
		sha="$$(sha256sum "$(GEMM_P31_EXAMPLE)" | cut -d" " -f1)"; \
		[ "$$sha" = "$(CUTEDSL_P31_EXAMPLE_SHA256)" ] \
			|| { echo "gemm-cublaslt-p33-smoke: FAIL: SHA-256 $$sha != pinned $(CUTEDSL_P31_EXAMPLE_SHA256)" >&2; exit 1; }; \
		echo "gemm-cublaslt-p33-smoke: upstream re-checked in this GPU container: commit $$head_commit sha256 $$sha" >&2; \
		mkdir -p $(GEMM_P33_BRIDGE_DIR); \
		nvcc -std=c++17 -O3 -lineinfo -Xcompiler -fPIC -shared \
			$(GEMM_P33_ARCH_FLAGS) \
			-o $(GEMM_P33_BRIDGE_LIB) $(GEMM_P33_BRIDGE) \
			-lcublasLt -lcudart >&2; \
		echo "gemm-cublaslt-p33-smoke: bridge compiled into container-private $(GEMM_P33_BRIDGE_LIB)" >&2; \
		exec python3 $(GEMM_P33_WRAPPER) \
			--warmup-iterations 2 \
			--iterations 10' || status=$$?; \
	echo "==============================================================================" >&2; \
	echo "P3.3 FUNCTIONAL VERIFICATION ONLY -- NOT AN EXPERIMENTAL RESULT AND" >&2; \
	echo "NOT A PERFORMANCE COMPARISON." >&2; \
	echo "Any CSV row on stdout is P3.3 infrastructure evidence: one frozen shape," >&2; \
	echo "(M,N,K,L)=(4096,4096,4096,1), 2 warm-ups and 10 measured launches, with hot" >&2; \
	echo "reused operands. setup_time_ms, first_launch_ms, and kernel_time_ms are" >&2; \
	echo "NON-PUBLISHABLE diagnostic fields; every row carries publishable=false." >&2; \
	echo "P3.3 computes no TFLOP/s, no speedup, no efficiency, and no comparison against" >&2; \
	echo "the P3.2 CuTe DSL wrapper. That comparison is P3.5 and does not exist." >&2; \
	if [ "$$status" -eq 0 ]; then \
		echo "P3.3 smoke completed: correctness passed before warm-up and steady-state timing." >&2; \
	else \
		echo "P3.3 smoke FAILED with exit status $$status: no CSV header and no CSV row" >&2; \
		echo "were emitted, and no result may be read from this run." >&2; \
	fi; \
	echo "==============================================================================" >&2; \
	exit $$status

# --- P3.4: three CuTe DSL execution variants ---------------------------------
# P3.2 established one CuTe DSL execution variant at the first final shape and
# P3.3 established the cuBLASLt baseline for the same geometry. P3.4 adds the
# two remaining execution variants the plan froze, so all three exist under one
# identical operand set, one identical correctness oracle, and one identical
# timing discipline:
#
#   nonpersistent_1cta   DenseGemmKernel            tiler (128,128) cluster (1,1)
#   persistent_1cta      PersistentDenseGemmKernel  tiler (128,128) cluster (1,1)
#   persistent_2cta      PersistentDenseGemmKernel  tiler (256,128) cluster (2,1)
#
# The 2-CTA row deliberately uses an M tile of 256 so each of the two
# participating CTAs keeps a local M extent of 128 -- the same two-SM geometry
# P2.2 measured, and the shape NVIDIA's own persistent example documents for
# use_2cta_instrs=True. No other tiler or cluster is ever substituted.
#
# This repository still owns no GEMM kernel. The non-persistent variant keeps
# using P3.1's pinned example and the two persistent variants use the official
# static-persistent example from the SAME pinned CUTLASS commit, both loaded
# read-only and in place from /opt/cutlass after their commit, blob, and
# SHA-256 are verified. Neither upstream run() is ever called and neither
# upstream benchmarking helper is used: P3.4 owns every timer. The only new
# pins are the three CUTEDSL_P34_* keys in PHASE3_VERSIONS.env; VERSIONS.env,
# the Dockerfile, and every closed P3.1/P3.2/P3.3 interface are untouched.
#
# P3.4 introduces no additional shape, no autotuning, no fourth candidate, no
# ranking, no TFLOP/s, no cuBLASLt execution or comparison, no campaign
# directory, and no result file. Comparing the three variants against each
# other or against P3.3 is P3.5's job. See src/gemm/P3_4_PROTOCOL.md.
#
# gemm-cutedsl-p34-check never touches a GPU, the network, or elevated
# privileges. It runs the existing P3.3 gate first (which itself runs the P3.2
# and P3.1 gates, all GPU-free and network-free, and all left completely
# intact), then runs inside the pinned image with --network none, --cap-drop
# ALL, no-new-privileges, the invoking UID/GID, no --gpus, and the repository
# mounted READ-ONLY, with PYTHONPYCACHEPREFIX and every temporary file under the
# container's own /tmp.
#
# gemm-cutedsl-p34-smoke is the only P3.4 target that executes on GPU. Its
# first recipe line validates BLACKWELL_GPU_INDEX before Docker, any build, or
# any check can start, which is why it deliberately has no Make prerequisite
# (same audited reasoning as the P3.1/P3.2/P3.3 smoke targets). It then goes
# exclusively through scripts/run_container.sh, which alone owns GPU selection,
# UUID resolution, and the idle-device proof.

gemm-cutedsl-p34-check: gemm-cublaslt-p33-check
	docker run --rm \
		--network none \
		--security-opt no-new-privileges \
		--cap-drop ALL \
		--user "$$(id -u):$$(id -g)" \
		-e HOME=/tmp \
		-e TMPDIR=/tmp \
		-e PYTHONPYCACHEPREFIX=/tmp/p34-pycache \
		-e CUTLASS_COMMIT="$(CUTLASS_COMMIT)" \
		-e CUTEDSL_VERSION="$(CUTEDSL_VERSION)" \
		-e PYTORCH_VERSION="$(PYTORCH_VERSION)" \
		-e PYTORCH_CUDA_VERSION="$(PYTORCH_CUDA_VERSION)" \
		-e CUDA_PYTHON_VERSION="$(CUDA_PYTHON_VERSION)" \
		-e CUDA_BINDINGS_VERSION="$(CUDA_BINDINGS_VERSION)" \
		-e P31_EXAMPLE="$(GEMM_P31_EXAMPLE)" \
		-e P31_EXAMPLE_GIT_BLOB="$(CUTEDSL_P31_EXAMPLE_GIT_BLOB)" \
		-e P31_EXAMPLE_SHA256="$(CUTEDSL_P31_EXAMPLE_SHA256)" \
		-e P34_EXAMPLE="$(GEMM_P34_PERSISTENT_EXAMPLE)" \
		-e P34_EXAMPLE_GIT_BLOB="$(CUTEDSL_P34_PERSISTENT_EXAMPLE_GIT_BLOB)" \
		-e P34_EXAMPLE_SHA256="$(CUTEDSL_P34_PERSISTENT_EXAMPLE_SHA256)" \
		-e P34_WRAPPER="$(GEMM_P34_WRAPPER)" \
		-e P34_CHECKER="$(GEMM_P34_CHECKER)" \
		-v "$(CURDIR):/workspace:ro" \
		-w /workspace \
		"$(IMAGE_TAG)" \
		bash -c 'set -euo pipefail; \
			fail() { echo "gemm-cutedsl-p34-check: FAIL: $$*" >&2; exit 1; }; \
			echo "== the pinned CUTLASS checkout =="; \
			[ -d /opt/cutlass ] || fail "/opt/cutlass is missing"; \
			head_commit="$$(git -c safe.directory=/opt/cutlass -C /opt/cutlass rev-parse HEAD)" \
				|| fail "cannot read the /opt/cutlass HEAD commit"; \
			[ "$$head_commit" = "$$CUTLASS_COMMIT" ] \
				|| fail "/opt/cutlass HEAD $$head_commit != pinned $$CUTLASS_COMMIT"; \
			dirty="$$(git -c safe.directory=/opt/cutlass -C /opt/cutlass status --porcelain --untracked-files=all)" \
				|| fail "cannot read the /opt/cutlass working tree status"; \
			[ -z "$$dirty" ] || fail "/opt/cutlass has tracked or untracked modifications"; \
			echo "checkout OK: commit $$head_commit"; \
			echo "== BOTH pinned official sources, verified independently =="; \
			for pair in "$$P31_EXAMPLE|$$P31_EXAMPLE_GIT_BLOB|$$P31_EXAMPLE_SHA256" \
					"$$P34_EXAMPLE|$$P34_EXAMPLE_GIT_BLOB|$$P34_EXAMPLE_SHA256"; do \
				file="$${pair%%|*}"; rest="$${pair#*|}"; \
				want_blob="$${rest%%|*}"; want_sha="$${rest#*|}"; \
				[ ! -L "$$file" ] || fail "$$file is a symlink"; \
				[ -f "$$file" ] || fail "$$file is not a regular file"; \
				blob="$$(git -c safe.directory=/opt/cutlass -C /opt/cutlass hash-object -- "$$file")" \
					|| fail "cannot compute the Git blob SHA of $$file"; \
				[ "$$blob" = "$$want_blob" ] \
					|| fail "$$file Git blob $$blob != pinned $$want_blob"; \
				sha="$$(sha256sum "$$file" | cut -d" " -f1)" \
					|| fail "cannot compute the SHA-256 of $$file"; \
				[ "$$sha" = "$$want_sha" ] \
					|| fail "$$file SHA-256 $$sha != pinned $$want_sha"; \
				echo "source OK: $$file"; \
				echo "           blob   $$blob"; \
				echo "           sha256 $$sha"; \
			done; \
			[ "$$P31_EXAMPLE" != "$$P34_EXAMPLE" ] \
				|| fail "the two pinned sources are the same file"; \
			echo "== the persistent source really carries the official static persistent path =="; \
			grep -q "StaticPersistentTileScheduler" "$$P34_EXAMPLE" \
				|| fail "the pinned persistent example has no StaticPersistentTileScheduler"; \
			grep -q "CtaGroup.TWO" "$$P34_EXAMPLE" \
				|| fail "the pinned persistent example has no CtaGroup.TWO selection path"; \
			grep -q "class PersistentDenseGemmKernel" "$$P34_EXAMPLE" \
				|| fail "the pinned persistent example does not define PersistentDenseGemmKernel"; \
			grep -q "class DenseGemmKernel" "$$P31_EXAMPLE" \
				|| fail "the pinned non-persistent example does not define DenseGemmKernel"; \
			echo "persistent scheduler and 2-CTA selection paths present"; \
			python3 -c "import os, cutlass, torch; \
				ce = os.environ[\"CUTEDSL_VERSION\"]; \
				assert cutlass.__version__ == ce, f\"CuTeDSL {cutlass.__version__} != pinned {ce}\"; \
				pe = os.environ[\"PYTORCH_VERSION\"]; \
				assert torch.__version__ == pe, f\"torch {torch.__version__} != pinned {pe}\"; \
				pc = os.environ[\"PYTORCH_CUDA_VERSION\"]; \
				assert torch.version.cuda == pc, f\"torch CUDA {torch.version.cuda} != pinned {pc}\"; \
				print(\"versions OK: cutedsl\", ce, \"torch\", pe, \"torch-cuda\", pc)"; \
			python3 -c "import os; from importlib.metadata import version; \
				expected = {\"cuda-python\": os.environ[\"CUDA_PYTHON_VERSION\"], \
					\"cuda-bindings\": os.environ[\"CUDA_BINDINGS_VERSION\"]}; \
				installed = {name: version(name) for name in expected}; \
				assert installed == expected, f\"installed distributions {installed} != pinned {expected}\"; \
				print(\"cuda distributions OK:\", installed)"; \
			echo "== pip check: the dependency graph must be consistent =="; \
			python3 -m pip check; \
			echo "== P3.4 python syntax =="; \
			python3 -m py_compile "$$P34_WRAPPER" "$$P34_CHECKER"; \
			echo "== P3.4 wrapper --help and --self-test are GPU-free =="; \
			python3 "$$P34_WRAPPER" --help > /dev/null \
				|| fail "the wrapper --help did not exit successfully"; \
			python3 "$$P34_WRAPPER" --self-test \
				|| fail "the wrapper GPU-free self-test failed"; \
			echo "== P3.4 checker self-test and full frozen-contract check =="; \
			python3 "$$P34_CHECKER" --self-test \
				|| fail "the checker self-test failed"; \
			python3 "$$P34_CHECKER" /workspace \
				|| fail "the P3.4 frozen-contract check failed"; \
			echo "P3.4 GPU-free contract OK (no GPU was used or required)"'
	@echo "gemm-cutedsl-p34-check: OK"

gemm-cutedsl-p34-smoke:
	@if [ -z "$${BLACKWELL_GPU_INDEX:-}" ]; then \
		echo "ERROR: BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index." >&2; \
		echo "       Example: BLACKWELL_GPU_INDEX=3 make gemm-cutedsl-p34-smoke" >&2; \
		echo "       This project never selects a GPU automatically." >&2; \
		exit 2; \
	fi
	@case "$${BLACKWELL_GPU_INDEX}" in \
		'' | *[!0-9]*) \
			echo "ERROR: BLACKWELL_GPU_INDEX must be a non-negative integer, got '$${BLACKWELL_GPU_INDEX}'." >&2; \
			exit 2; ;; \
	esac
	@status=0; \
	RUN_CONTAINER_STDOUT_IS_DATA=1 scripts/run_container.sh bash -c 'set -euo pipefail; \
		head_commit="$$(git -c safe.directory=/opt/cutlass -C /opt/cutlass rev-parse HEAD)"; \
		[ "$$head_commit" = "$(CUTLASS_COMMIT)" ] \
			|| { echo "gemm-cutedsl-p34-smoke: FAIL: /opt/cutlass HEAD $$head_commit != pinned $(CUTLASS_COMMIT)" >&2; exit 1; }; \
		sha_nonpersistent="$$(sha256sum "$(GEMM_P31_EXAMPLE)" | cut -d" " -f1)"; \
		[ "$$sha_nonpersistent" = "$(CUTEDSL_P31_EXAMPLE_SHA256)" ] \
			|| { echo "gemm-cutedsl-p34-smoke: FAIL: non-persistent SHA-256 $$sha_nonpersistent != pinned $(CUTEDSL_P31_EXAMPLE_SHA256)" >&2; exit 1; }; \
		sha_persistent="$$(sha256sum "$(GEMM_P34_PERSISTENT_EXAMPLE)" | cut -d" " -f1)"; \
		[ "$$sha_persistent" = "$(CUTEDSL_P34_PERSISTENT_EXAMPLE_SHA256)" ] \
			|| { echo "gemm-cutedsl-p34-smoke: FAIL: persistent SHA-256 $$sha_persistent != pinned $(CUTEDSL_P34_PERSISTENT_EXAMPLE_SHA256)" >&2; exit 1; }; \
		echo "gemm-cutedsl-p34-smoke: both upstream sources re-checked in this GPU container:" >&2; \
		echo "gemm-cutedsl-p34-smoke:   commit          $$head_commit" >&2; \
		echo "gemm-cutedsl-p34-smoke:   non-persistent  $$sha_nonpersistent" >&2; \
		echo "gemm-cutedsl-p34-smoke:   persistent      $$sha_persistent" >&2; \
		exec python3 $(GEMM_P34_WRAPPER) \
			--warmup-iterations 2 \
			--iterations 10' || status=$$?; \
	echo "==============================================================================" >&2; \
	echo "P3.4 FUNCTIONAL VERIFICATION ONLY -- NOT AN EXPERIMENTAL RESULT." >&2; \
	echo "Any CSV on stdout is P3.4 infrastructure evidence: three frozen execution" >&2; \
	echo "variants at ONE frozen shape, (M,N,K,L)=(4096,4096,4096,1), 2 warm-ups and 10" >&2; \
	echo "measured launches each, with hot reused operands. ALL TIMINGS ARE" >&2; \
	echo "NON-PUBLISHABLE diagnostic fields; every row carries publishable=false." >&2; \
	echo "P3.4 computes no TFLOP/s, no speedup, no efficiency, and no ranking, and NO" >&2; \
	echo "variant-versus-variant or CuTe-versus-cuBLASLt COMPARISON has been performed." >&2; \
	echo "That comparison is P3.5 and does not exist." >&2; \
	if [ "$$status" -eq 0 ]; then \
		echo "P3.4 smoke completed: every variant passed correctness before its warm-up and steady-state timing." >&2; \
	else \
		echo "P3.4 smoke FAILED with exit status $$status: no CSV header and no CSV row" >&2; \
		echo "were emitted for ANY variant, and no result may be read from this run." >&2; \
	fi; \
	echo "==============================================================================" >&2; \
	exit $$status

# --- P3.5: five shapes and comparison ----------------------------------------
# P3.2 established one CuTe DSL execution variant at the first final shape, P3.3
# established the cuBLASLt baseline for the same geometry, and P3.4 added the two
# remaining execution variants -- all still at that single shape. P3.5 extends
# the same already verified infrastructure to ALL FIVE final Experiment 3 shapes
# and performs the first explicit, purely descriptive comparison among the four
# candidates:
#
#   shapes      (4096,4096,4096,1) (8192,8192,8192,1) (16384,512,4096,1)
#               (32768,512,4096,1) (512,16384,4096,1)
#   candidates  nonpersistent_1cta  persistent_1cta  persistent_2cta
#               cublaslt/heuristic_first_supported   (the comparison baseline)
#
# Output is shape-major: five shapes in the frozen order, four candidates in the
# frozen order inside each, for exactly 20 rows and 21 lines. No arbitrary shape
# is reachable from the CLI, the environment, a configuration file, or an input
# CSV: the Python wrapper and the C bridge freeze the same five independently,
# and the wrapper reads the bridge's own allowlist back and requires the two to
# be identical before any measurement runs.
#
# This repository still owns no GEMM kernel and adds no pin: the three CuTe DSL
# candidates use the SAME two pinned official NVIDIA examples P3.4 uses, and the
# cuBLASLt candidate uses the library that already ships in the pinned CUDA 13.1
# image. VERSIONS.env, PHASE3_VERSIONS.env, the Dockerfile, scripts/
# run_container.sh, and every closed P3.1/P3.2/P3.3/P3.4 interface are untouched.
#
# The comparison is arithmetic, not a conclusion: exact 2*M*N*K FLOP counts,
# TFLOP/s from kernel_time_ms alone, a ratio and a SIGNED gap against the
# cuBLASLt baseline (negative means faster, and it is never clamped), a rank with
# a frozen-order tie break, and the best CuTe DSL variant per shape. No
# confidence interval, p-value, outlier removal, roofline, empirical-ceiling
# utilization, bandwidth, arithmetic intensity, or causal interpretation is
# computed anywhere. Every row is publishable=false. The pilot, the final
# campaigns, the statistics, the experiment integration, and every
# interpretation are Phase 4 work. See src/gemm/P3_5_PROTOCOL.md.
#
# gemm-comparison-p35-check never touches a GPU, the network, or elevated
# privileges. It runs the existing P3.4 gate first (which itself runs the P3.3,
# P3.2, and P3.1 gates, all GPU-free and network-free, and all left completely
# intact), then runs inside the pinned image with --network none, --cap-drop
# ALL, no-new-privileges, the invoking UID/GID, no --gpus, and the repository
# mounted READ-ONLY, with PYTHONPYCACHEPREFIX and every build artifact under the
# container's own /tmp.
#
# gemm-comparison-p35-smoke is the only P3.5 target that executes on GPU. Its
# first recipe action validates BLACKWELL_GPU_INDEX before Docker, any build, or
# any check can start, which is why it deliberately has no Make prerequisite
# (same audited reasoning as the P3.1/P3.2/P3.3/P3.4 smoke targets). It then goes
# exclusively through scripts/run_container.sh, which alone owns GPU selection,
# UUID resolution, and the idle-device proof.

gemm-comparison-p35-check: gemm-cutedsl-p34-check
	docker run --rm \
		--network none \
		--security-opt no-new-privileges \
		--cap-drop ALL \
		--user "$$(id -u):$$(id -g)" \
		-e HOME=/tmp \
		-e TMPDIR=/tmp \
		-e PYTHONPYCACHEPREFIX=/tmp/p35-pycache \
		-e CUTLASS_COMMIT="$(CUTLASS_COMMIT)" \
		-e CUDA_SHORT_VERSION="$(CUDA_SHORT_VERSION)" \
		-e CUTEDSL_VERSION="$(CUTEDSL_VERSION)" \
		-e PYTORCH_VERSION="$(PYTORCH_VERSION)" \
		-e PYTORCH_CUDA_VERSION="$(PYTORCH_CUDA_VERSION)" \
		-e CUDA_PYTHON_VERSION="$(CUDA_PYTHON_VERSION)" \
		-e CUDA_BINDINGS_VERSION="$(CUDA_BINDINGS_VERSION)" \
		-e P31_EXAMPLE="$(GEMM_P31_EXAMPLE)" \
		-e P31_EXAMPLE_GIT_BLOB="$(CUTEDSL_P31_EXAMPLE_GIT_BLOB)" \
		-e P31_EXAMPLE_SHA256="$(CUTEDSL_P31_EXAMPLE_SHA256)" \
		-e P34_EXAMPLE="$(GEMM_P34_PERSISTENT_EXAMPLE)" \
		-e P34_EXAMPLE_GIT_BLOB="$(CUTEDSL_P34_PERSISTENT_EXAMPLE_GIT_BLOB)" \
		-e P34_EXAMPLE_SHA256="$(CUTEDSL_P34_PERSISTENT_EXAMPLE_SHA256)" \
		-e P35_WRAPPER="$(GEMM_P35_WRAPPER)" \
		-e P35_BRIDGE="$(GEMM_P35_BRIDGE)" \
		-e P35_CHECKER="$(GEMM_P35_CHECKER)" \
		-e P35_BRIDGE_DIR="$(GEMM_P35_BRIDGE_DIR)" \
		-e P35_BRIDGE_LIB="$(GEMM_P35_BRIDGE_LIB)" \
		-e P35_ARCH_FLAGS="$(GEMM_P35_ARCH_FLAGS)" \
		-v "$(CURDIR):/workspace:ro" \
		-w /workspace \
		"$(IMAGE_TAG)" \
		bash -c 'set -euo pipefail; \
			fail() { echo "gemm-comparison-p35-check: FAIL: $$*" >&2; exit 1; }; \
			echo "== the pinned CUDA toolkit that must supply cuBLASLt =="; \
			nvcc_version="$$(nvcc --version | sed -n "s/.*release \([0-9.]*\).*/\1/p")"; \
			[ "$$nvcc_version" = "$$CUDA_SHORT_VERSION" ] \
				|| fail "nvcc reports CUDA $$nvcc_version, pinned is $$CUDA_SHORT_VERSION"; \
			echo "nvcc CUDA $$nvcc_version (cuBLASLt ships with it; no package is added)"; \
			echo "== the pinned CUTLASS checkout =="; \
			[ -d /opt/cutlass ] || fail "/opt/cutlass is missing"; \
			head_commit="$$(git -c safe.directory=/opt/cutlass -C /opt/cutlass rev-parse HEAD)" \
				|| fail "cannot read the /opt/cutlass HEAD commit"; \
			[ "$$head_commit" = "$$CUTLASS_COMMIT" ] \
				|| fail "/opt/cutlass HEAD $$head_commit != pinned $$CUTLASS_COMMIT"; \
			dirty="$$(git -c safe.directory=/opt/cutlass -C /opt/cutlass status --porcelain --untracked-files=all)" \
				|| fail "cannot read the /opt/cutlass working tree status"; \
			[ -z "$$dirty" ] || fail "/opt/cutlass has tracked or untracked modifications"; \
			echo "checkout OK: commit $$head_commit"; \
			echo "== BOTH pinned official sources, verified independently =="; \
			for pair in "$$P31_EXAMPLE|$$P31_EXAMPLE_GIT_BLOB|$$P31_EXAMPLE_SHA256" \
					"$$P34_EXAMPLE|$$P34_EXAMPLE_GIT_BLOB|$$P34_EXAMPLE_SHA256"; do \
				file="$${pair%%|*}"; rest="$${pair#*|}"; \
				want_blob="$${rest%%|*}"; want_sha="$${rest#*|}"; \
				[ ! -L "$$file" ] || fail "$$file is a symlink"; \
				[ -f "$$file" ] || fail "$$file is not a regular file"; \
				blob="$$(git -c safe.directory=/opt/cutlass -C /opt/cutlass hash-object -- "$$file")" \
					|| fail "cannot compute the Git blob SHA of $$file"; \
				[ "$$blob" = "$$want_blob" ] \
					|| fail "$$file Git blob $$blob != pinned $$want_blob"; \
				sha="$$(sha256sum "$$file" | cut -d" " -f1)" \
					|| fail "cannot compute the SHA-256 of $$file"; \
				[ "$$sha" = "$$want_sha" ] \
					|| fail "$$file SHA-256 $$sha != pinned $$want_sha"; \
				echo "source OK: $$file"; \
				echo "           blob   $$blob"; \
				echo "           sha256 $$sha"; \
			done; \
			[ "$$P31_EXAMPLE" != "$$P34_EXAMPLE" ] \
				|| fail "the two pinned sources are the same file"; \
			python3 -c "import os, cutlass, torch; \
				ce = os.environ[\"CUTEDSL_VERSION\"]; \
				assert cutlass.__version__ == ce, f\"CuTeDSL {cutlass.__version__} != pinned {ce}\"; \
				pe = os.environ[\"PYTORCH_VERSION\"]; \
				assert torch.__version__ == pe, f\"torch {torch.__version__} != pinned {pe}\"; \
				pc = os.environ[\"PYTORCH_CUDA_VERSION\"]; \
				assert torch.version.cuda == pc, f\"torch CUDA {torch.version.cuda} != pinned {pc}\"; \
				print(\"versions OK: cutedsl\", ce, \"torch\", pe, \"torch-cuda\", pc)"; \
			python3 -c "import os; from importlib.metadata import version; \
				expected = {\"cuda-python\": os.environ[\"CUDA_PYTHON_VERSION\"], \
					\"cuda-bindings\": os.environ[\"CUDA_BINDINGS_VERSION\"]}; \
				installed = {name: version(name) for name in expected}; \
				assert installed == expected, f\"installed distributions {installed} != pinned {expected}\"; \
				print(\"cuda distributions OK:\", installed)"; \
			echo "== pip check: the dependency graph must be consistent =="; \
			python3 -m pip check; \
			echo "== compile the P3.5 C-ABI cuBLASLt bridge into container-private /tmp =="; \
			mkdir -p "$$P35_BRIDGE_DIR"; \
			nvcc -std=c++17 -O3 -lineinfo \
				-Xcompiler -fPIC -shared \
				$$P35_ARCH_FLAGS \
				-o "$$P35_BRIDGE_LIB" "$$P35_BRIDGE" \
				-lcublasLt -lcudart \
				|| fail "the P3.5 cuBLASLt bridge did not compile"; \
			echo "bridge compiled: $$P35_BRIDGE_LIB"; \
			echo "== the shared object must call cublasLtMatmul and no fallback GEMM API =="; \
			nm -D --defined-only "$$P35_BRIDGE_LIB" > /tmp/p35-defined.txt \
				|| fail "cannot read the defined symbols of the bridge"; \
			nm -D -u "$$P35_BRIDGE_LIB" > /tmp/p35-undefined.txt \
				|| fail "cannot read the undefined symbols of the bridge"; \
			readelf -d "$$P35_BRIDGE_LIB" > /tmp/p35-dynamic.txt \
				|| fail "cannot read the dynamic section of the bridge"; \
			grep -qw "cublasLtMatmul" /tmp/p35-undefined.txt \
				|| fail "the measured path does not reference cublasLtMatmul"; \
			grep -qw "cublasLtMatmulAlgoCheck" /tmp/p35-undefined.txt \
				|| fail "the bridge never validates the selected algorithm"; \
			grep -qw "cublasLtMatmulAlgoGetHeuristic" /tmp/p35-undefined.txt \
				|| fail "the bridge never queries the vendor heuristic"; \
			for symbol in p35_bridge_abi_version p35_plan_info_size p35_last_error \
					p35_cublaslt_version p35_shape_count p35_shape_at \
					p35_plan_create p35_plan_execute p35_stream_synchronize \
					p35_plan_destroy; do \
				grep -qw "$$symbol" /tmp/p35-defined.txt \
					|| fail "the bridge does not export $$symbol"; \
			done; \
			for forbidden in cublasGemmEx cublasGemmStridedBatchedEx cublasGemmBatchedEx \
					cublasSgemm cublasHgemm cublasLtMatmulAlgoGetIds cublasLtMatmulAlgoInit; do \
				if grep -qw "$$forbidden" /tmp/p35-defined.txt /tmp/p35-undefined.txt; then \
					fail "the bridge references the forbidden fallback API $$forbidden"; \
				fi; \
			done; \
			grep -q "libcublasLt.so" /tmp/p35-dynamic.txt \
				|| fail "the bridge is not linked against libcublasLt"; \
			grep -q "libcudart.so" /tmp/p35-dynamic.txt \
				|| fail "the bridge is not linked against libcudart"; \
			echo "ELF inspection OK: cublasLtMatmul present, no fallback GEMM API present"; \
			echo "== P3.5 python syntax =="; \
			python3 -m py_compile "$$P35_WRAPPER" "$$P35_CHECKER"; \
			echo "== P3.5 wrapper --help and --self-test are GPU-free =="; \
			python3 "$$P35_WRAPPER" --help > /dev/null \
				|| fail "the wrapper --help did not exit successfully"; \
			python3 "$$P35_WRAPPER" --self-test \
				|| fail "the wrapper GPU-free self-test failed"; \
			echo "== P3.5 checker self-test and full frozen-contract check =="; \
			python3 "$$P35_CHECKER" --self-test \
				|| fail "the checker self-test failed"; \
			python3 "$$P35_CHECKER" /workspace \
				|| fail "the P3.5 frozen-contract check failed"; \
			echo "P3.5 GPU-free contract OK (no GPU was used or required)"'
	@echo "gemm-comparison-p35-check: OK"

gemm-comparison-p35-smoke:
	@if [ -z "$${BLACKWELL_GPU_INDEX:-}" ]; then \
		echo "ERROR: BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index." >&2; \
		echo "       Example: BLACKWELL_GPU_INDEX=3 make gemm-comparison-p35-smoke" >&2; \
		echo "       This project never selects a GPU automatically." >&2; \
		exit 2; \
	fi
	@case "$${BLACKWELL_GPU_INDEX}" in \
		'' | *[!0-9]*) \
			echo "ERROR: BLACKWELL_GPU_INDEX must be a non-negative integer, got '$${BLACKWELL_GPU_INDEX}'." >&2; \
			exit 2; ;; \
	esac
	@status=0; \
	RUN_CONTAINER_STDOUT_IS_DATA=1 scripts/run_container.sh bash -c 'set -euo pipefail; \
		head_commit="$$(git -c safe.directory=/opt/cutlass -C /opt/cutlass rev-parse HEAD)"; \
		[ "$$head_commit" = "$(CUTLASS_COMMIT)" ] \
			|| { echo "gemm-comparison-p35-smoke: FAIL: /opt/cutlass HEAD $$head_commit != pinned $(CUTLASS_COMMIT)" >&2; exit 1; }; \
		sha_nonpersistent="$$(sha256sum "$(GEMM_P31_EXAMPLE)" | cut -d" " -f1)"; \
		[ "$$sha_nonpersistent" = "$(CUTEDSL_P31_EXAMPLE_SHA256)" ] \
			|| { echo "gemm-comparison-p35-smoke: FAIL: non-persistent SHA-256 $$sha_nonpersistent != pinned $(CUTEDSL_P31_EXAMPLE_SHA256)" >&2; exit 1; }; \
		sha_persistent="$$(sha256sum "$(GEMM_P34_PERSISTENT_EXAMPLE)" | cut -d" " -f1)"; \
		[ "$$sha_persistent" = "$(CUTEDSL_P34_PERSISTENT_EXAMPLE_SHA256)" ] \
			|| { echo "gemm-comparison-p35-smoke: FAIL: persistent SHA-256 $$sha_persistent != pinned $(CUTEDSL_P34_PERSISTENT_EXAMPLE_SHA256)" >&2; exit 1; }; \
		echo "gemm-comparison-p35-smoke: both upstream sources re-checked in this GPU container:" >&2; \
		echo "gemm-comparison-p35-smoke:   commit          $$head_commit" >&2; \
		echo "gemm-comparison-p35-smoke:   non-persistent  $$sha_nonpersistent" >&2; \
		echo "gemm-comparison-p35-smoke:   persistent      $$sha_persistent" >&2; \
		mkdir -p $(GEMM_P35_BRIDGE_DIR); \
		nvcc -std=c++17 -O3 -lineinfo -Xcompiler -fPIC -shared \
			$(GEMM_P35_ARCH_FLAGS) \
			-o $(GEMM_P35_BRIDGE_LIB) $(GEMM_P35_BRIDGE) \
			-lcublasLt -lcudart >&2; \
		echo "gemm-comparison-p35-smoke: bridge compiled into container-private $(GEMM_P35_BRIDGE_LIB)" >&2; \
		exec python3 $(GEMM_P35_WRAPPER) \
			--warmup-iterations 2 \
			--iterations 10' || status=$$?; \
	echo "==============================================================================" >&2; \
	echo "P3.5 FUNCTIONAL COMPARISON EVIDENCE ONLY -- NOT AN EXPERIMENTAL RESULT." >&2; \
	echo "ALL FIVE SHAPES and ALL FOUR CANDIDATES were required: (4096,4096,4096,1)," >&2; \
	echo "(8192,8192,8192,1), (16384,512,4096,1), (32768,512,4096,1), (512,16384,4096,1)" >&2; \
	echo "x nonpersistent_1cta, persistent_1cta, persistent_2cta, and cuBLASLt" >&2; \
	echo "heuristic_first_supported, with 2 warm-ups and 10 measured launches each and" >&2; \
	echo "hot reused operands. ALL 20 ROWS ARE NON-PUBLISHABLE: every row carries" >&2; \
	echo "publishable=false, and the comparison fields (TFLOP/s, the baseline ratio, the" >&2; \
	echo "signed gap, the rank, and the best CuTe DSL variant) are ARITHMETIC, NOT A" >&2; \
	echo "CONCLUSION. NO FINAL CAMPAIGN, NO PILOT, NO STATISTICAL treatment or" >&2; \
	echo "significance claim, NO NSIGHT COMPUTE analysis, and NO PHASE 4 interpretation" >&2; \
	echo "has been performed. Beating cuBLASLt is not a success criterion." >&2; \
	if [ "$$status" -eq 0 ]; then \
		echo "P3.5 smoke completed: every candidate of every shape passed correctness before its warm-up and steady-state timing." >&2; \
	else \
		echo "P3.5 smoke FAILED with exit status $$status: no CSV header and no CSV row" >&2; \
		echo "were emitted for ANY shape or candidate, including rows already completed," >&2; \
		echo "and no result or comparison may be read from this run." >&2; \
	fi; \
	echo "==============================================================================" >&2; \
	exit $$status
