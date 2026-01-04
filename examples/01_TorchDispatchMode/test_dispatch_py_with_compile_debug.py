import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils._python_dispatch import TorchDispatchMode


ROOT = Path(__file__).resolve().parent


def _setup_env(root: Path) -> Path:
    """
    注意：不要在模块 import 时就调用（否则会创建默认的 cache/debug_log/trace_log 目录）。
    只在真正跑某个 case 时调用。
    """
    os.chdir(root)

    # 只要打开 TORCH_COMPILE_DEBUG，compile 产物就会落盘；默认不打开海量日志。
    os.environ.setdefault("TORCH_COMPILE_DEBUG", "1")

    # 允许外部（例如多 case runner）传入独立 debug/cache dir，避免“串味”
    os.environ.setdefault("TORCH_COMPILE_DEBUG_DIR", str(root / "debug_log"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / "cache"))

    # profiler trace 也支持分 case 输出（默认复用原来的 trace_log）
    os.environ.setdefault("TRACE_LOG_DIR", str(root / "trace_log"))

    # 可选：需要排查 Dynamo/Inductor 细节时再打开，避免默认刷屏
    if os.environ.get("ENABLE_TORCH_LOGS", "0") == "1":
        os.environ.setdefault("TORCHDYNAMO_VERBOSE", "1")
        os.environ.setdefault("TORCH_LOGS", "+dynamo,+inductor")

    # 提前建目录，避免某些版本/路径下落盘失败
    Path(os.environ["TORCH_COMPILE_DEBUG_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TRACE_LOG_DIR"]).mkdir(parents=True, exist_ok=True)

    return Path(os.environ["TRACE_LOG_DIR"])


# Register the custom op with PyTorch using torch.library.custom_op
@torch.library.custom_op("mylib::numpy_matmul", mutates_args=())
def numpy_matmul_custom_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Custom matrix multiplication implementation using numpy.
    This replaces torch.mm with a numpy-based implementation.
    """
    print("Enter numpy_matmul_custom_op")

    # Convert torch tensors to numpy
    a_np = a.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()

    # Perform matrix multiplication using numpy
    result_np = np.matmul(a_np, b_np)

    # Convert back to torch tensor with same device and dtype as input
    device = a.device
    dtype = a.dtype
    result = torch.from_numpy(result_np).to(device=device, dtype=dtype)
    return result


@numpy_matmul_custom_op.register_fake
def numpy_matmul_custom_op_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Fake implementation for torch.compile compatibility.
    This is used during tracing to infer output shape and dtype without actual computation.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Both inputs must be 2D tensors for matrix multiplication")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes for matmul: {a.shape} and {b.shape}")
    output_shape = (a.shape[0], b.shape[1])
    return torch.zeros(output_shape, dtype=a.dtype, device=a.device)


class HookMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # func.overloadpacket contains operator metadata, e.g., torch.ops.aten.mm
        if func.overloadpacket == torch.ops.aten.mm:
            print(f"[Hook] Captured torch.mm! Shapes: {args[0].shape} x {args[1].shape}")
            print("[Hook] Replacing with numpy-based custom op implementation")
            return numpy_matmul_custom_op(args[0], args[1])
        return func(*args, **kwargs)


def _make_profile(trace_dir: Path):
    return profile(
        activities=[ProfilerActivity.CPU],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
        record_shapes=True,
        with_stack=True,
    )


def _inputs():
    x = torch.randn(4, 4)
    w = torch.randn(4, 4)
    return x, w


def _run_case(case: str):
    trace_dir = _setup_env(ROOT)
    x, w = _inputs()
    myprofile = _make_profile(trace_dir)

    print(f"[case] {case}")
    print(f"[env] TORCH_COMPILE_DEBUG_DIR={os.environ.get('TORCH_COMPILE_DEBUG_DIR')}")
    print(f"[env] TORCHINDUCTOR_CACHE_DIR={os.environ.get('TORCHINDUCTOR_CACHE_DIR')}")
    print(f"[env] TRACE_LOG_DIR={os.environ.get('TRACE_LOG_DIR')}")
    if os.environ.get("ENABLE_TORCH_LOGS", "0") == "1":
        print(f"[env] TORCHDYNAMO_VERBOSE={os.environ.get('TORCHDYNAMO_VERBOSE')}")
        print(f"[env] TORCH_LOGS={os.environ.get('TORCH_LOGS')}")

    if case == "try1_compile":
        print("Try1: torch.compile")

        def simple_function(x, w):
            for _ in range(5):
                x = torch.mm(x, w) + torch.mm(torch.sin(x), w)
                x = torch.relu(x)
            return x

        compiled_fn = torch.compile(simple_function)
        with torch.inference_mode():
            _ = compiled_fn(x, w)

        print("--- Profiling compiled function ---")
        with myprofile:
            with torch.inference_mode():
                for _ in range(2):
                    _ = compiled_fn(x, w)
        return

    if case == "try2_custom_op":
        print("Try2: torch.compile + torch.ops.mylib.numpy_matmul")

        def simple_function(x, w):
            for _ in range(5):
                x = torch.ops.mylib.numpy_matmul(x, w) + torch.ops.mylib.numpy_matmul(torch.sin(x), w)
                x = torch.relu(x)
            return x

        compiled_fn = torch.compile(simple_function)
        with torch.inference_mode():
            for _ in range(3):
                _ = compiled_fn(x, w=w)

        print("--- Profiling compiled function ---")
        with myprofile:
            with torch.inference_mode():
                for _ in range(2):
                    _ = compiled_fn(x, w=w)
        return

    if case == "try3_compile_hook":
        print("Try3: torch.compile + HookMode")

        def simple_function(x, w):
            for _ in range(5):
                x = torch.mm(x, w) + torch.mm(torch.sin(x), w)
                x = torch.relu(x)
            return x

        with HookMode():
            compiled_fn = torch.compile(simple_function)
            with torch.inference_mode():
                for _ in range(3):
                    _ = compiled_fn(x, w)

            print("--- Profiling compiled function ---")
            with myprofile:
                with torch.inference_mode():
                    for _ in range(2):
                        _ = compiled_fn(x, w)
        return

    if case == "try4_fullgraph_hook":
        print("Try4: torch.compile + HookMode + full graph capture")

        def simple_function(x, w):
            for _ in range(5):
                x = torch.mm(x, w) + torch.mm(torch.sin(x), w)
                x = torch.relu(x)
            return x

        with HookMode():
            compiled_fn_full = torch.compile(simple_function, fullgraph=True)
            with torch.inference_mode():
                for _ in range(3):
                    _ = compiled_fn_full(x, w)

            print("--- Profiling fullgraph compiled function ---")
            with myprofile:
                with torch.inference_mode():
                    for _ in range(2):
                        _ = compiled_fn_full(x, w)
        return

    raise ValueError(
        f"unknown case={case}. supported: try1_compile, try2_custom_op, try3_compile_hook, try4_fullgraph_hook"
    )


def main():
    # 无参数：依次跑所有 case（每个 case 用子进程隔离全局状态 & 分开落盘 compile 产物）
    if len(sys.argv) == 1:
        cases = [
            "try1_compile", 
            "try2_custom_op", 
            "try3_compile_hook", 
            "try4_fullgraph_hook"
        ]
        for c in cases:
            env = os.environ.copy()
            env["TORCH_COMPILE_DEBUG"] = "1"
            env["TORCH_COMPILE_DEBUG_DIR"] = str(ROOT / "debug_log_cases" / c)
            env["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "cache_cases" / c)
            env["TRACE_LOG_DIR"] = str(ROOT / "trace_log_cases" / c)
            # 可选：需要详细日志时再开
            #   ENABLE_TORCH_LOGS=1 python test_dispatch_py_with_compile_debug.py
            if env.get("ENABLE_TORCH_LOGS", "0") == "1":
                env["TORCHDYNAMO_VERBOSE"] = "1"
                env["TORCH_LOGS"] = "+dynamo,+inductor"
            print(f"\n\n##############################\n### RUN CASE (subprocess): {c}\n##############################")
            subprocess.run([sys.executable, __file__, c], check=True, env=env, cwd=str(ROOT))
        return

    case = sys.argv[1]
    try:
        _run_case(case)
    except Exception as e:
        # 保持行为和原脚本接近：不同版本/配置下 torch.compile 可能失败
        print(f"[error] case={case} failed: {e}")
        raise


if __name__ == "__main__":
    main()


