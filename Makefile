# gb300-gemm-anatomy Phase 0 Makefile.
# Exposed targets: help, check-static, build-image, check-env, preflight.
# No target selects a GPU automatically, elevates privileges, or exceeds two
# build jobs.

include VERSIONS.env

IMAGE_TAG ?= gb300-gemm-anatomy:phase0

# Derived pins: "13.1" from CUDA_VERSION=13.1.0, "4.6.1" from CUTLASS_VERSION=v4.6.1.
CUDA_SHORT_VERSION := $(basename $(CUDA_VERSION))
CUTEDSL_VERSION := $(patsubst v%,%,$(CUTLASS_VERSION))

REQUIRED_FILES := \
	AGENTS.md README.md PLAN.md LICENSE .gitignore VERSIONS.env \
	Dockerfile Makefile \
	scripts/run_container.sh scripts/preflight.sh \
	smoke/cuda_smoke.cu smoke/cutedsl_smoke.py \
	results/README.md

.DEFAULT_GOAL := help
.PHONY: help check-static build-image check-env preflight

help:
	@echo "gb300-gemm-anatomy — Phase 0 targets"
	@echo ""
	@echo "  make help          Show this help."
	@echo "  make check-static  Static validation: no Docker, no GPU, no network."
	@echo "  make build-image   Build the pinned image ($(IMAGE_TAG)). No GPU."
	@echo "  make check-env     Check tools/versions inside a GPU-less container."
	@echo "  make preflight     Run the single-GPU preflight. Requires an explicit"
	@echo "                     BLACKWELL_GPU_INDEX=<physical-index>; never"
	@echo "                     selects a GPU automatically."
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
	@echo "== forbidden patterns absent from scripts, Dockerfile, smoke =="
	@pat='--gpus[ =]+all|NVIDIA_VISIBLE_DEVICES=all|--privileged|--pid[ =]+host|docker\.sock|--cap-add|SYS_ADMIN|set -x'; \
	pat="$$pat|\bs""udo\b|\$$\(np""roc\)|nvidia-smi[^|]*(-pm|--persistence-mode|-lgc|--lock-gpu-clocks|-pl|--power-limit)"; \
	! grep -nE -- "$$pat" scripts/run_container.sh scripts/preflight.sh Dockerfile \
		smoke/cuda_smoke.cu smoke/cutedsl_smoke.py
	@! grep -nE "s""udo|np""roc" Makefile
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
