# gb300-gemm-anatomy Makefile.
# Exposed targets: help, check-static, build-image, check-env, preflight,
# memory-ldgsts-build, memory-ldgsts-sass, memory-ldgsts-self-test,
# memory-ldgsts-smoke, memory-tma-build, memory-tma-sass,
# memory-tma-self-test, memory-tma-smoke.
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

REQUIRED_FILES := \
	AGENTS.md README.md PLAN.md LICENSE .gitignore VERSIONS.env \
	Dockerfile Makefile \
	scripts/run_container.sh scripts/preflight.sh scripts/check_ldgsts_sass.py \
	scripts/check_tma_sass.py \
	smoke/cuda_smoke.cu smoke/cutedsl_smoke.py \
	src/memory/ldgsts.cu src/memory/tma.cu src/memory/README.md \
	results/README.md

.DEFAULT_GOAL := help
.PHONY: help check-static build-image check-env preflight \
	memory-ldgsts-build memory-ldgsts-sass memory-ldgsts-self-test memory-ldgsts-smoke \
	memory-tma-build memory-tma-sass memory-tma-self-test memory-tma-smoke

help:
	@echo "gb300-gemm-anatomy — Phase 0 + P1.1 (LDGSTS) + P1.2 (TMA) targets"
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
	@echo "Pinned contract (VERSIONS.env): CUDA $(CUDA_VERSION), CUTLASS $(CUTLASS_VERSION),"
	@echo "arch $(CUDA_ARCH), max build jobs $(MAX_BUILD_JOBS)."

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
	@echo "== Dockerfile consistent with VERSIONS.env =="
	@grep -Fq "$(CUDA_IMAGE)@$(CUDA_IMAGE_DIGEST)" Dockerfile
	@grep -Fq "CUTLASS_COMMIT=$(CUTLASS_COMMIT)" Dockerfile
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
	@grep -Fq 'P1.2 | Equivalent TMA path | YES | NO | NO |' PLAN.md
	@! grep -rnF 'has not been started' README.md src/memory/README.md
	@! grep -rnF 'no TMA code exists yet' README.md src/memory/README.md
	@! grep -nF 'P1.2 and experiments' README.md
	@echo "check-static: OK"

build-image:
	docker build \
		--platform "$(CUDA_IMAGE_PLATFORM)" \
		--build-arg BASE_IMAGE="$(CUDA_IMAGE)@$(CUDA_IMAGE_DIGEST)" \
		--build-arg CUDA_VERSION="$(CUDA_VERSION)" \
		--build-arg CUTLASS_VERSION="$(CUTLASS_VERSION)" \
		--build-arg CUTLASS_COMMIT="$(CUTLASS_COMMIT)" \
		--build-arg MAX_BUILD_JOBS="$(MAX_BUILD_JOBS)" \
		--tag "$(IMAGE_TAG)" \
		.

check-env:
	docker run --rm \
		--network none \
		--security-opt no-new-privileges \
		--cap-drop ALL \
		-e CUDA_SHORT_VERSION="$(CUDA_SHORT_VERSION)" \
		-e CUTEDSL_VERSION="$(CUTEDSL_VERSION)" \
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
