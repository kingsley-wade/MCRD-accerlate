#!/usr/bin/env bash
# Non-interactive ACP entry point for the canonical PDE smoke and p53 MCRD study.
set -Eeuo pipefail

PROJECT_ROOT="/data/250010002/DNA-p53-MCRD-accelerated"
PYTHON="${PROJECT_ROOT}/.venv312/bin/python"

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "ERROR: expected project directory does not exist: ${PROJECT_ROOT}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: expected Python/JAX environment does not exist: ${PYTHON}" >&2
  exit 2
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${PROJECT_ROOT}/reports/acp-h100-${RUN_ID}"
mkdir -p "${RESULT_DIR}"
LOG_FILE="${RESULT_DIR}/job.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

on_exit() {
  status=$?
  if [[ ${status} -eq 0 ]]; then
    echo "JOB_STATUS=success"
  else
    echo "JOB_STATUS=failed exit_code=${status}"
  fi
  echo "RESULT_DIR=${RESULT_DIR}"
  exit "${status}"
}
trap on_exit EXIT

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export JAX_ENABLE_X64=true
# JAX 0.12's CUDA plugin is registered as "cuda". A forced "gpu" platform
# makes this build probe the unavailable ROCm plugin; leave platform discovery automatic.
unset JAX_PLATFORMS
# Avoid reserving most of the MIG partition before the solver needs the memory.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "ACP H100 experiment job"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "project_root=${PROJECT_ROOT}"
echo "python=${PYTHON}"
echo "result_dir=${RESULT_DIR}"
echo "image_note=The job must use the same CUDA 12.4 H100 MIG image as the prepared .venv312 environment."

"${PYTHON}" - <<'PY'
import jax
import jaxlib
import numpy
import sys

devices = jax.devices()
if not devices:
    raise SystemExit("No JAX devices are visible")
device = devices[0]
kind = getattr(device, "device_kind", str(device))
if device.platform != "gpu" or "H100" not in kind.upper():
    raise SystemExit(f"Expected an H100 GPU/MIG device, detected {device!r}, kind={kind!r}")
print(f"PRECHECK python={sys.version.split()[0]} numpy={numpy.__version__}")
print(f"PRECHECK jax={jax.__version__} jaxlib={jaxlib.__version__}")
print(f"PRECHECK device={device} kind={kind}")
try:
    print(f"PRECHECK device_memory_stats={device.memory_stats()}")
except Exception as exc:
    print(f"PRECHECK device_memory_stats_unavailable={type(exc).__name__}: {exc}")
PY

echo "PHASE=pde_smoke_to_128"
"${PYTHON}" scripts/validate_local.py \
  --case all \
  --grid-sizes 16,32,64,128 \
  --steps 100 \
  --dt 1e-5 \
  --backend jax-cuda \
  --dtype float32 \
  --output "${RESULT_DIR}/pde-smoke-to-128.json"

echo "PHASE=p53_fixed_domain_grid_study_to_4096_at_t40"
"${PYTHON}" scripts/run_p53_grid_study.py \
  --output-dir "${RESULT_DIR}/p53-grid-study" \
  --grid-sizes 64,128,256,512,1024,2048,4096 \
  --domain-length 25.6 \
  --final-time 40 \
  --seed 2026 \
  --dtype float64 \
  --chunk-steps 10240 \
  --relative-change-tolerance 1e-5 \
  --consecutive-windows 3

echo "PHASE=checksums"
if command -v sha256sum >/dev/null 2>&1; then
  (
    cd "${RESULT_DIR}"
    find . -type f ! -name SHA256SUMS ! -name job.log -print0 \
      | sort -z \
      | xargs -0 sha256sum > SHA256SUMS
  )
fi

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "RESULT_DIR=${RESULT_DIR}"
