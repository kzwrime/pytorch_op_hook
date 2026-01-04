import torch
import numpy as np
from torch.utils._python_dispatch import TorchDispatchMode
from torch.profiler import profile, record_function, ProfilerActivity
# Register the custom op with PyTorch using torch.library.custom_op
@torch.library.custom_op("mylib::numpy_matmul", mutates_args=())
def numpy_matmul_custom_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Custom matrix multiplication implementation using numpy.
    This replaces torch.mm with a numpy-based implementation.
    """

    print(f"Enter numpy_matmul_custom_op")

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
    # Calculate output shape: (M, K) @ (K, N) -> (M, N)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Both inputs must be 2D tensors for matrix multiplication")

    # Infer output shape based on input shapes
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes for matmul: {a.shape} and {b.shape}")

    output_shape = (a.shape[0], b.shape[1])
    return torch.zeros(output_shape, dtype=a.dtype, device=a.device)

class HijackMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 1. Check if this is an operator we care about
        # func.overloadpacket contains operator metadata, e.g., torch.ops.aten.mm
        if func.overloadpacket == torch.ops.aten.mm:
            print(f"[Hijack] Captured torch.mm! Shapes: {args[0].shape} x {args[1].shape}")
            print(f"[Hijack] Replacing with numpy-based custom op implementation")

            # Replace with our numpy-based custom op
            return numpy_matmul_custom_op(args[0], args[1])

        # 2. Pass through: execute the original operator
        # Note: calling func in TorchDispatchMode automatically passes through without recursive infinite loop
        return func(*args, **kwargs)

# --- Test ---
x = torch.randn(4, 4)
w = torch.randn(4, 4)

myprofile = profile(
    activities=[ProfilerActivity.CPU],
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/'),
    record_shapes=True,
    with_stack=True
)

hmd = HijackMode()

# --- Test with torch.compile ---
print("\n--- Testing torch.compile ---")

# Test if torch.compile is available and test it with try-except
try:
    print("Try1: torch.compile")

    def simple_function(x, w):
        for _ in range(5):
            x = torch.mm(x, w) + torch.mm(torch.sin(x), w)
            x = torch.relu(x)
        return x
    compiled_fn = torch.compile(simple_function)

    with torch.inference_mode():
        y_compiled = compiled_fn(x, w)

    print("--- Testing compiled function normly ---")
    with myprofile:
        with torch.inference_mode():
            for i in range(2):
                y_compiled = compiled_fn(x, w)

    print("Compiled function test completed\n\n")
except Exception as e:
    print(f"torch.compile test failed (this is expected on some PyTorch versions): {e}")
    print("Skipping torch.compile test")

# Test if torch.compile is available and test it with try-except
try:
    print("Try2: torch.compile + torch.ops.mylib.numpy_matmul")

    def simple_function(x, w):
        for _ in range(5):
            x = torch.ops.mylib.numpy_matmul(x, w) + torch.ops.mylib.numpy_matmul(torch.sin(x), w)
            x = torch.relu(x)
        return x
    compiled_fn = torch.compile(simple_function)

    with torch.inference_mode():
        for i in range(3):
            y_compiled = compiled_fn(x, w=w)

    print("--- Testing compiled function with Hijack Mode ---")
    with myprofile:
        with torch.inference_mode():
            for i in range(2):
                y_compiled = compiled_fn(x, w=w)

    print("Compiled function test completed\n\n")
except Exception as e:
    print(f"torch.compile test failed (this is expected on some PyTorch versions): {e}")
    print("Skipping torch.compile test")
finally:
    # hmd.__exit__(None, None, None)
    pass

# Test if torch.compile is available and test it with try-except
try:
    print("Try3: torch.compile + HijackMode")

    hmd.__enter__()
    def simple_function(x, w):
        for _ in range(5):
            x = torch.mm(x, w) + torch.mm(torch.sin(x), w)
            x = torch.relu(x)
        return x
    compiled_fn = torch.compile(simple_function)

    with torch.inference_mode():
        for i in range(3):
            y_compiled = compiled_fn(x, w)

    print("--- Testing compiled function with Hijack Mode ---")
    with myprofile:
        with torch.inference_mode():
            for i in range(2):
                y_compiled = compiled_fn(x, w)
    hmd.__exit__(None, None, None)

    print("Compiled function test completed\n\n")
except Exception as e:
    print(f"torch.compile test failed (this is expected on some PyTorch versions): {e}")
    print("Skipping torch.compile test")
finally:
    # hmd.__exit__(None, None, None)
    pass

# Test compile with full graph capture
try:
    print("Try4: torch.compile + HijackMode + full graph capture")

    hmd.__enter__()

    def simple_function(x, w):
        for _ in range(5):
            x = torch.mm(x, w) + torch.mm(torch.sin(x), w)
            x = torch.relu(x)
        return x
    compiled_fn_full = torch.compile(simple_function, fullgraph=True)

    with torch.inference_mode():
        for i in range(3):
            y_full = compiled_fn_full(x, w)

    print("--- Testing fullgraph compiled function with Hijack Mode ---")
    with myprofile:
        with torch.inference_mode():
            for i in range(2):
                y_full = compiled_fn_full(x, w)
    hmd.__exit__(None, None, None)

    print("Fullgraph compiled function test completed\n\n")
except Exception as e:
    print(f"Fullgraph torch.compile test failed: {e}")
    print("Skipping fullgraph test")
finally:
    # hmd.__exit__(None, None, None)
    pass