#!/usr/bin/env bash
# Non-interactive ACP entry point for matched CPU or H100 full-workflow experiments.
# All files are kept under /data/250010002/DNA-p53-MCRD-accelerated.
set -Eeuo pipefail

PROJECT_ROOT="/data/250010002/DNA-p53-MCRD-accelerated"
PYTHON="${PROJECT_ROOT}/.venv312/bin/python"
BACKEND="${BACKEND:-jax-cpu}"
CPU_CORES="${CPU_CORES:-8}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export BACKEND CPU_CORES

if [[ "${BACKEND}" != "jax-cpu" && "${BACKEND}" != "jax-cuda" ]]; then
  echo "ERROR: BACKEND must be jax-cpu or jax-cuda" >&2
  exit 2
fi
if [[ ! -d "${PROJECT_ROOT}" || ! -x "${PYTHON}" ]]; then
  echo "ERROR: expected project and Python environment under ${PROJECT_ROOT}" >&2
  exit 2
fi

RESULT_DIR="${PROJECT_ROOT}/reports/acp-full-workflow-${BACKEND}-${RUN_ID}"
mkdir -p "${RESULT_DIR}"
LOG_FILE="${RESULT_DIR}/job.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
START_EPOCH="$(date +%s)"
STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
JOB_STATUS="failed"

on_exit() {
  status=$?
  FINISHED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  FINISH_EPOCH="$(date +%s)"
  if [[ ${status} -eq 0 ]]; then JOB_STATUS="success"; fi
  export RESULT_DIR BACKEND CPU_CORES STARTED_UTC FINISHED_UTC JOB_STATUS
  export WORKFLOW_WALL_SECONDS="$((FINISH_EPOCH - START_EPOCH))"
  "${PYTHON}" - <<'PY' || true
import json, os
from pathlib import Path
p = Path(os.environ['RESULT_DIR']) / 'workflow-manifest.json'
p.write_text(json.dumps({
    'schema_version': 1,
    'backend': os.environ['BACKEND'],
    'cpu_cores_requested': int(os.environ['CPU_CORES']) if os.environ['BACKEND'] == 'jax-cpu' else None,
    'started_utc': os.environ['STARTED_UTC'],
    'finished_utc': os.environ['FINISHED_UTC'],
    'wall_seconds': int(os.environ['WORKFLOW_WALL_SECONDS']),
    'status': os.environ['JOB_STATUS'],
    'result_dir': os.environ['RESULT_DIR'],
}, indent=2) + '\n')
PY
  if [[ -f "${RESULT_DIR}/workflow-manifest.json" ]] && command -v sha256sum >/dev/null 2>&1; then
    (cd "${RESULT_DIR}" && find . -type f ! -name SHA256SUMS ! -name job.log -print0 \
      | sort -z | xargs -0 sha256sum > SHA256SUMS)
  fi
  echo "JOB_STATUS=${JOB_STATUS} exit_code=${status}"
  echo "RESULT_DIR=${RESULT_DIR}"
  exit "${status}"
}
trap on_exit EXIT

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${CPU_CORES}" OPENBLAS_NUM_THREADS="${CPU_CORES}"
export MKL_NUM_THREADS="${CPU_CORES}" NUMEXPR_NUM_THREADS="${CPU_CORES}"
if [[ "${BACKEND}" == "jax-cpu" ]]; then
  export JAX_PLATFORMS=cpu
  CPU_SET="$("${PYTHON}" -c 'import os; a=sorted(os.sched_getaffinity(0)); n=int(os.environ.get("CPU_CORES", "8")); print(",".join(map(str,a[:n])) if len(a)>=n else "")')"
  if [[ -z "${CPU_SET}" ]]; then
    echo "ERROR: ACP exposes fewer than ${CPU_CORES} CPU cores" >&2
    exit 2
  fi
  RUN=(taskset -c "${CPU_SET}" "${PYTHON}")
  echo "cpu_affinity=${CPU_SET}"
else
  unset JAX_PLATFORMS
  RUN=("${PYTHON}")
fi

printf 'ACP full workflow\nbackend=%s\nstarted_utc=%s\nproject_root=%s\npython=%s\nresult_dir=%s\n' \
  "${BACKEND}" "${STARTED_UTC}" "${PROJECT_ROOT}" "${PYTHON}" "${RESULT_DIR}"
if [[ "${BACKEND}" == "jax-cuda" ]]; then
  echo "image_note=Use the CUDA 12.4 H100 MIG image matching the prepared .venv312 environment."
fi
"${RUN[@]}" - <<'PY'
import jax, jaxlib, numpy, os, sys
from mcrd_accel.backends import get_jax_device
from mcrd_accel.config import Backend
device = get_jax_device(Backend(os.environ['BACKEND']))
expected = 'cpu' if os.environ['BACKEND'] == 'jax-cpu' else 'gpu'
kind = getattr(device, 'device_kind', str(device))
if device.platform != expected:
    raise SystemExit(f"Expected {expected}, found {device.platform}: {device}")
if expected == 'gpu' and 'H100' not in kind.upper():
    raise SystemExit(f"Expected H100, found {kind}")
if not jax.config.read('jax_enable_x64'):
    raise SystemExit('JAX float64 is disabled')
print(f'PRECHECK python={sys.version.split()[0]} numpy={numpy.__version__}')
print(f'PRECHECK jax={jax.__version__} jaxlib={jaxlib.__version__}')
print(f'PRECHECK backend={os.environ["BACKEND"]} device={device} kind={kind}')
PY

echo "PHASE=pde_smoke_to_128"
"${RUN[@]}" scripts/validate_local.py \
  --case all --grid-sizes 16,32,64,128 --steps 100 --dt 1e-5 \
  --backend "${BACKEND}" --dtype float32 \
  --output "${RESULT_DIR}/pde-smoke-to-128.json"

echo "PHASE=p53_fixed_domain_grid_study_to_4096_at_t40"
"${RUN[@]}" scripts/run_p53_grid_study.py \
  --backend "${BACKEND}" --cpu-cores "${CPU_CORES}" \
  --output-dir "${RESULT_DIR}/p53-grid-study" \
  --grid-sizes 64,128,256,512,1024,2048,4096 \
  --domain-length 25.6 --final-time 40 --seed 2026 --dtype float64 \
  --chunk-steps 10240 --relative-change-tolerance 1e-5 --consecutive-windows 3

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "RESULT_DIR=${RESULT_DIR}"
