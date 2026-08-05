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
# compute-umma-p24-analyze, gemm-cutedsl-p31-check, gemm-cutedsl-p31-smoke.
# No target selects a GPU automatically, elevates privileges, or exceeds two
# build jobs.

include VERSIONS.env

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
# only P3.1 file it adds is the protocol below. Every provenance value comes
# from VERSIONS.env (CUTLASS_COMMIT, CUTEDSL_P31_EXAMPLE_PATH,
# CUTEDSL_P31_EXAMPLE_GIT_BLOB, CUTEDSL_P31_EXAMPLE_SHA256).
GEMM_P31_PROTOCOL := src/gemm/P3_1_PROTOCOL.md
GEMM_P31_EXAMPLE := /opt/cutlass/$(CUTEDSL_P31_EXAMPLE_PATH)

REQUIRED_FILES := \
	AGENTS.md README.md PLAN.md LICENSE .gitignore VERSIONS.env \
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
	$(GEMM_P31_PROTOCOL)

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
	gemm-cutedsl-p31-check gemm-cutedsl-p31-smoke

help:
	@echo "gb300-gemm-anatomy — Phase 0 + P1.1 (LDGSTS) + P1.2 (TMA) + P1.3 (sweep) targets"
	@echo ""
	@echo "  make help                     Show this help."
	@echo "  make check-static             Static validation: no Docker, no GPU, no network."
	@echo "  make build-image              Build the pinned image ($(IMAGE_TAG)). No GPU."
	@echo "  make check-env                Check tools/versions inside a GPU-less container."
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
	@echo "     implemented; pending independent audit and GB300 verification. Executes"
	@echo "     NVIDIA's own unmodified dense_gemm.py from the pinned /opt/cutlass"
	@echo "     checkout: BF16 x BF16 -> FP32, (M,N,K,L)=(256,256,512,1), non-persistent,"
	@echo "     1-CTA MMA group, mma tiler (128,128), cluster (1,1), TMA store. This"
	@echo "     repository owns no GEMM source and P3.1 produces NO performance result.) --"
	@echo "  GPU-free P3.1 provenance/environment gate (no GPU, no network):"
	@echo "  make gemm-cutedsl-p31-check    Verify /opt/cutlass HEAD, checkout cleanliness,"
	@echo "                                 the example's regular-file identity, Git blob"
	@echo "                                 SHA and SHA-256, the CuTe DSL/PyTorch pins, and"
	@echo "                                 that the example's own --help runs GPU-free."
	@echo "  GPU-executing (requires BLACKWELL_GPU_INDEX; never selects a GPU automatically):"
	@echo "  make gemm-cutedsl-p31-smoke    Re-check the upstream commit and SHA-256 inside"
	@echo "                                 the GPU container, then run the frozen official"
	@echo "                                 command with mandatory reference validation."
	@echo "                                 Functional smoke check only, NOT a performance"
	@echo "                                 result; any internally computed timing is"
	@echo "                                 discarded."
	@echo ""
	@echo "Pinned contract (VERSIONS.env): CUDA $(CUDA_VERSION), CUTLASS $(CUTLASS_VERSION),"
	@echo "arch $(CUDA_ARCH), max build jobs $(MAX_BUILD_JOBS), auxiliary PyTorch"
	@echo "$(PYTORCH_VERSION) (CUDA $(PYTORCH_CUDA_VERSION))."

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
	@echo "== P3.1 version contract format (exact, non-floating pins) =="
	@grep -Eq '^PYTORCH_VERSION=2\.10\.0\+cu130$$' VERSIONS.env
	@grep -Eq '^PYTORCH_INDEX_URL=https://download\.pytorch\.org/whl/cu130$$' VERSIONS.env
	@grep -Eq '^PYTORCH_CUDA_VERSION=13\.0$$' VERSIONS.env
	@grep -Eq '^CUTEDSL_P31_EXAMPLE_PATH=examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm\.py$$' VERSIONS.env
	@grep -Eq '^CUTEDSL_P31_EXAMPLE_GIT_BLOB=[0-9a-f]{40}$$' VERSIONS.env
	@grep -Eq '^CUTEDSL_P31_EXAMPLE_SHA256=[0-9a-f]{64}$$' VERSIONS.env
	@echo "== Dockerfile consistent with VERSIONS.env =="
	@grep -Fq "$(CUDA_IMAGE)@$(CUDA_IMAGE_DIGEST)" Dockerfile
	@grep -Fq "CUTLASS_COMMIT=$(CUTLASS_COMMIT)" Dockerfile
	@grep -Fq "PYTORCH_VERSION=$(PYTORCH_VERSION)" Dockerfile
	@grep -Fq "PYTORCH_INDEX_URL=$(PYTORCH_INDEX_URL)" Dockerfile
	@grep -Fq "PYTORCH_CUDA_VERSION=$(PYTORCH_CUDA_VERSION)" Dockerfile
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
	@pat='--use_2cta'; pat="$$pat""_instrs|--skip_ref""_check|--use_cold""_l2|dense_gemm""_persistent"; \
	! grep -nE -- "$$pat" Makefile
	@echo "== P3.1 smoke validates BLACKWELL_GPU_INDEX before any Docker prerequisite =="
	@grep -Eq '^gemm-cutedsl-p31-smoke:$$' Makefile
	@grep -Fq 'scripts/run_container.sh' Makefile
	@echo "== truthful P3.1 status assertions =="
	@grep -Fq 'P3.1 | Pinned official CuTe DSL example | YES | NO | NO |' PLAN.md
	@grep -Fq 'P3.2 | One-shape wrapper | NO | NO | NO |' PLAN.md
	@grep -Fq 'P3.3 | cuBLASLt baseline | NO | NO | NO |' PLAN.md
	@grep -Fq 'P3.4 | Three execution variants | NO | NO | NO |' PLAN.md
	@grep -Fq 'P3.5 | Five shapes and comparison | NO | NO | NO |' PLAN.md
	@grep -Fq 'P3.1 = YES / NO / NO' $(GEMM_P31_PROTOCOL)
	@grep -Fq 'P3.1 produces no experimental result' $(GEMM_P31_PROTOCOL)
	@grep -Fq 'non-persistent' $(GEMM_P31_PROTOCOL)
	@grep -Fq 'P3.1 (pinned official CuTe DSL example)' README.md
	@! grep -nF 'P3.1 | Pinned official CuTe DSL example | YES | YES' PLAN.md
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
# the pinned values, CuTe DSL and PyTorch report the pinned versions, and the
# example's own --help exits successfully -- with every frozen option present
# -- without a device. Every expected value is passed in from VERSIONS.env;
# none is duplicated as an unconnected constant here. /opt/cutlass is a
# root-owned checkout inside the image while the container runs as the invoking
# user, so each Git query carries an explicit, per-invocation
# -c safe.directory for that one path; nothing is ever written to the checkout.
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
