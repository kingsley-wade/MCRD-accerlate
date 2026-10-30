#!/usr/bin/env bash
# Matched p53 CPU/H100 short benchmark for non-interactive ACP jobs.
set -Eeuo pipefail

PROJECT_ROOT="/data/250010002/DNA-p53-MCRD-accelerated"
PYTHON="${PROJECT_ROOT}/.venv312/bin/python"
BACKEND="${BACKEND:-jax-cpu}"
CPU_CORES="${CPU_CORES:-8}"
GRID_SIZES="${GRID_SIZES:-64,128,256,512,1024,2048,4096}"
STEPS="${STEPS:-10240}"
REPEATS="${REPEATS:-3}"
WARMUPS="${WARMUPS:-1}"

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "ERROR: expected project directory does not exist: ${PROJECT_ROOT}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: expected Python/JAX environment does not exist: ${PYTHON}" >&2
  exit 2
fi
case "${BACKEND}" in
  jax-cpu)
    export JAX_PLATFORMS=cpu
    ;;
  jax-cuda)
    unset JAX_PLATFORMS
    ;;
  *)
    echo "ERROR: BACKEND must be jax-cpu or jax-cuda, got ${BACKEND}" >&2
    exit 2
    ;;
esac

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${PROJECT_ROOT}/reports/acp-p53-backend-${BACKEND}-${RUN_ID}"
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
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=true
export OMP_NUM_THREADS="${CPU_CORES}"
export MKL_NUM_THREADS="${CPU_CORES}"
export OPENBLAS_NUM_THREADS="${CPU_CORES}"
export NUMEXPR_NUM_THREADS="${CPU_CORES}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "ACP matched p53 backend benchmark"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "project_root=${PROJECT_ROOT}"
echo "python=${PYTHON}"
echo "backend=${BACKEND}"
echo "cpu_cores=${CPU_CORES}"
echo "grid_sizes=${GRID_SIZES}"
echo "steps=${STEPS} repeats=${REPEATS} warmups=${WARMUPS}"
echo "result_dir=${RESULT_DIR}"

"${PYTHON}" scripts/benchmark_p53_backend.py \
  --backend "${BACKEND}" \
  --output-dir "${RESULT_DIR}" \
  --grid-sizes "${GRID_SIZES}" \
  --steps "${STEPS}" \
  --repeats "${REPEATS}" \
  --warmups "${WARMUPS}" \
  --cpu-cores "${CPU_CORES}"

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
