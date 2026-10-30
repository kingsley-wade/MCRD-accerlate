import ctypes, hashlib, importlib.metadata, json, os, platform, socket, subprocess, sys, time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import jax
from mcrd_accel.config import Backend, DType, SolverConfig
from mcrd_accel.mcrd import ReactionModel, run
from mcrd_accel.mcrd.solver import make_model_initial_state
root=Path("/data/250010002/DNA-p53-MCRD-accelerated")
out=root/"reports/h100-validation-20261030/cpu-vs-cuda-benchmark.json"
lib=ctypes.CDLL(str((root/".venv312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12").resolve()))
def mem():
    free,total=ctypes.c_size_t(),ctypes.c_size_t()
    rc=lib.cudaMemGetInfo(ctypes.byref(free),ctypes.byref(total))
    if rc: raise RuntimeError(f"cudaMemGetInfo failed: {rc}")
    return {"free_bytes":free.value,"total_bytes":total.value}
def allocator_stats():
    try: return {k:int(v) for k,v in jax.devices()[0].memory_stats().items() if isinstance(v,(int,np.integer))}
    except Exception as exc: return {"unavailable":str(exc)}
before=mem()
if before["free_bytes"] < 16*1024**3: raise SystemExit(f"GPU free memory below 16 GiB: {before}")
revision=subprocess.run(["git","--git-dir="+str(root/".git"),"--work-tree="+str(root),"rev-parse","HEAD"],capture_output=True,text=True).stdout.strip() or None
driver=subprocess.run(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],capture_output=True,text=True).stdout.strip()
device=jax.devices()[0]
if device.platform != "gpu" or "H100" not in device.device_kind or "MIG" not in device.device_kind:
    raise SystemExit(f"Unexpected device: {device.platform} {device.device_kind}")
report={"schema_version":1,"created_at_utc":datetime.now(timezone.utc).isoformat(),"purpose":"same-input NumPy CPU vs JAX CUDA MCRD benchmark","source_revision":revision,"host":{"hostname":socket.gethostname(),"platform":platform.platform(),"machine":platform.machine(),"cpu_count":os.cpu_count(),"python":sys.version.split()[0]},"runtime":{"numpy":np.__version__,"jax":jax.__version__,"jaxlib":jax.lib.__version__,"nvidia_cuda_runtime_package":importlib.metadata.version("nvidia-cuda-runtime-cu12"),"driver_version":driver,"device":str(device),"device_kind":device.device_kind,"device_platform":device.platform},"method":{"model":"p53_mcrd","dtype":"float32","steps":200,"grids":[128,256],"seed":271828,"warmup_count":2,"repeats":5,"timing":"wall time for public run() call; CUDA includes host-to-device input and final output transfer; first CUDA call reported separately and excluded from steady-state","peak_memory":"JAX allocator peak stats; not whole-process or system memory"},"cuda_memory_before":before,"results":[]}
for n in (128,256):
    config=SolverConfig.fig2b(Nx=n,Ny=n,dx=0.2,dy=0.2,dt=0.02,n_steps=200,dtype=DType.FLOAT32,backend=Backend.NUMPY,seed=271828)
    gpu_config=replace(config,backend=Backend.JAX_CUDA)
    model=ReactionModel.p53_mcrd(config)
    initial=make_model_initial_state(config,model)
    input_hash=hashlib.sha256(initial.u.tobytes()+initial.v.tobytes()).hexdigest()
    def execute(cfg): return run(cfg,initial_state=initial,checkpoint_stride=0,model=model)
    t=time.perf_counter(); cpu_first=execute(config); cpu_first_wall=time.perf_counter()-t
    t=time.perf_counter(); gpu_first=execute(gpu_config); gpu_first_wall=time.perf_counter()-t
    errors={}
    for name in ("u","v"):
        expected=getattr(cpu_first.final_state,name); actual=getattr(gpu_first.final_state,name)
        diff=np.asarray(actual,dtype=np.float64)-np.asarray(expected,dtype=np.float64)
        errors[name]={"max_abs":float(np.max(np.abs(diff))),"relative_l2":float(np.linalg.norm(diff.ravel())/max(np.linalg.norm(expected.ravel()),np.finfo(float).tiny)),"allclose":bool(np.allclose(actual,expected,rtol=3e-5,atol=2e-6)),"finite":bool(np.all(np.isfinite(actual)))}
    parity_passed=all(v["allclose"] and v["finite"] for v in errors.values())
    if not parity_passed:
        raise SystemExit(f"CPU/CUDA parity failed on grid {n}: {errors}")
    if mem()["free_bytes"] < 16*1024**3: raise SystemExit(f"GPU free memory fell below 16 GiB after grid {n} parity run")
    for _ in range(2): execute(config)
    for _ in range(2): execute(gpu_config)
    cpu_times=[]; gpu_times=[]
    for _ in range(5):
        t=time.perf_counter(); execute(config); cpu_times.append(time.perf_counter()-t)
        t=time.perf_counter(); execute(gpu_config); gpu_times.append(time.perf_counter()-t)
    result={"grid":f"{n}x{n}","config":config.to_dict(),"same_initial_state_sha256":input_hash,"parity":{"tolerance":{"rtol":3e-5,"atol":2e-6},"fields":errors,"status":"passed"},"first_call_wall_seconds":{"numpy":cpu_first_wall,"jax_cuda_including_compile_and_transfer":gpu_first_wall},"warmup_count":2,"repeats":5,"steady_state_wall_seconds":{"numpy_runs":cpu_times,"jax_cuda_runs":gpu_times,"numpy_mean":float(np.mean(cpu_times)),"numpy_std":float(np.std(cpu_times,ddof=1)),"numpy_median":float(np.median(cpu_times)),"jax_cuda_mean":float(np.mean(gpu_times)),"jax_cuda_std":float(np.std(gpu_times,ddof=1)),"jax_cuda_median":float(np.median(gpu_times)),"speedup_mean_ratio":float(np.mean(cpu_times)/np.mean(gpu_times)),"speedup_median_ratio":float(np.median(cpu_times)/np.median(gpu_times))},"runtime_seconds_from_solver":{"numpy":cpu_first.metadata.get("runtime_seconds"),"jax_cuda_first_call":gpu_first.metadata.get("runtime_seconds")},"jax_allocator_stats_after_grid":allocator_stats(),"cuda_memory_after_grid":mem()}
    report["results"].append(result)
    print(json.dumps({"grid":result["grid"],"parity":"passed","cpu_mean_s":result["steady_state_wall_seconds"]["numpy_mean"],"cuda_mean_s":result["steady_state_wall_seconds"]["jax_cuda_mean"],"speedup":result["steady_state_wall_seconds"]["speedup_mean_ratio"],"cuda_free_gib":result["cuda_memory_after_grid"]["free_bytes"]/1024**3},sort_keys=True))
report["cuda_memory_after"]=mem()
out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
print(f"report={out}")
