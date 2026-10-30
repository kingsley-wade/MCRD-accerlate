import ctypes, hashlib, importlib.metadata, json, platform, socket, subprocess, sys, time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from mcrd_accel.config import Backend, DType, SolverConfig
from mcrd_accel.mcrd import ReactionModel, run
from mcrd_accel.mcrd.solver import make_model_initial_state
import jax
root=Path("/data/250010002/DNA-p53-MCRD-accelerated")
out=root/"reports/h100-validation-20261030/mcrd-parity.json"
lib=ctypes.CDLL(str((root/".venv312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12").resolve()))
def mem():
    free,total=ctypes.c_size_t(),ctypes.c_size_t()
    rc=lib.cudaMemGetInfo(ctypes.byref(free),ctypes.byref(total))
    if rc: raise RuntimeError(f"cudaMemGetInfo failed: {rc}")
    return {"free_bytes":free.value,"total_bytes":total.value}
before=mem()
if before["free_bytes"] < 16*1024**3: raise SystemExit(f"GPU free memory below 16 GiB: {before}")
config=SolverConfig.fig2b(Nx=64,Ny=64,dx=0.2,dy=0.2,dt=0.02,n_steps=100,dtype=DType.FLOAT32,backend=Backend.NUMPY,seed=31415)
gpu_config=replace(config,backend=Backend.JAX_CUDA)
model=ReactionModel.p53_mcrd(config)
initial=make_model_initial_state(config,model)
input_hash=hashlib.sha256(initial.u.tobytes()+initial.v.tobytes()).hexdigest()
t=time.perf_counter(); cpu=run(config,initial_state=initial,checkpoint_stride=0,model=model); cpu_wall=time.perf_counter()-t
t=time.perf_counter(); gpu=run(gpu_config,initial_state=initial,checkpoint_stride=0,model=model); gpu_wall=time.perf_counter()-t
errors={}
for name in ("u","v"):
    expected=getattr(cpu.final_state,name); actual=getattr(gpu.final_state,name)
    diff=np.asarray(actual,dtype=np.float64)-np.asarray(expected,dtype=np.float64)
    errors[name]={"max_abs":float(np.max(np.abs(diff))),"relative_l2":float(np.linalg.norm(diff.ravel())/max(np.linalg.norm(expected.ravel()),np.finfo(float).tiny)),"allclose":bool(np.allclose(actual,expected,rtol=3e-5,atol=2e-6)),"finite":bool(np.all(np.isfinite(actual)))}
passed=all(v["allclose"] and v["finite"] for v in errors.values())
revision=subprocess.run(["git","--git-dir="+str(root/".git"),"--work-tree="+str(root),"rev-parse","HEAD"],capture_output=True,text=True).stdout.strip() or None
driver=subprocess.run(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],capture_output=True,text=True).stdout.strip()
report={"schema_version":1,"created_at_utc":datetime.now(timezone.utc).isoformat(),"purpose":"same-input NumPy reference vs JAX CUDA correctness parity","source_revision":revision,"host":{"hostname":socket.gethostname(),"platform":platform.platform(),"machine":platform.machine(),"python":sys.version.split()[0]},"runtime":{"numpy":np.__version__,"jax":jax.__version__,"jaxlib":jax.lib.__version__,"nvidia_cuda_runtime_package":importlib.metadata.version("nvidia-cuda-runtime-cu12"),"driver_version":driver,"device":str(jax.devices()[0]),"device_kind":jax.devices()[0].device_kind,"device_platform":jax.devices()[0].platform},"config":config.to_dict(),"same_initial_state_sha256":input_hash,"tolerance":{"rtol":3e-5,"atol":2e-6},"comparison":errors,"wall_seconds":{"numpy":cpu_wall,"jax_cuda_including_compile_and_transfer":gpu_wall},"solver_runtime_seconds":{"numpy":cpu.metadata.get("runtime_seconds"),"jax_cuda":gpu.metadata.get("runtime_seconds")},"relative_mass_drift":{"numpy":abs(cpu.metadata["final_mass"]-cpu.metadata["initial_mass"])/abs(cpu.metadata["initial_mass"]),"jax_cuda":abs(gpu.metadata["final_mass"]-gpu.metadata["initial_mass"])/abs(gpu.metadata["initial_mass"])},"cuda_memory_before":before,"cuda_memory_after":mem(),"status":"passed" if passed else "failed"}
out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
print(json.dumps({"status":report["status"],"source_revision":revision,"device_kind":report["runtime"]["device_kind"],"comparison":errors,"same_initial_state_sha256":input_hash},indent=2))
if not passed: raise SystemExit(1)
