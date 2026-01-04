import torch
import numpy as np
from torch.utils._python_dispatch import TorchDispatchMode

# Custom op implemented with numpy
def numpy_matmul_custom_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Custom matrix multiplication implementation using numpy.
    This replaces torch.mm with a numpy-based implementation.
    """
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
x = torch.randn(3, 4)
w = torch.randn(4, 5)

hmd = HijackMode()

# --- Test with torch.compile ---
print("\n--- Testing torch.compile ---")

# Test if torch.compile is available and test it with try-except
try:
    print("Try1: Attempting to compile function with torch.compile...")

    def simple_function(x, w):
        return torch.mm(x, w)
    compiled_fn = torch.compile(simple_function)

    print("--- Testing compiled function normly ---")
    with torch.inference_mode():
        y_compiled = compiled_fn(x, w)

    print("Compiled function test completed\n\n")
except Exception as e:
    print(f"torch.compile test failed (this is expected on some PyTorch versions): {e}")
    print("Skipping torch.compile test")

# Test if torch.compile is available and test it with try-except
try:
    print("Try2: Attempting to compile function with torch.compile...")

    hmd.__enter__()
    def simple_function(x, w):
        return torch.mm(x, w)
    compiled_fn = torch.compile(simple_function)

    print("--- Testing compiled function with Hijack Mode ---")
    with torch.inference_mode():
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
    print("Try3: Attempting to compile with full graph capture...")

    hmd.__enter__()

    def simple_function(x, w):
        return torch.mm(x, w)
    compiled_fn_full = torch.compile(simple_function, fullgraph=True)

    print("--- Testing fullgraph compiled function with Hijack Mode ---")
    with torch.inference_mode():
        y_full = compiled_fn_full(x, w)
    hmd.__exit__(None, None, None)

    print("Fullgraph compiled function test completed\n\n")
except Exception as e:
    print(f"Fullgraph torch.compile test failed: {e}")
    print("Skipping fullgraph test")
finally:
    # hmd.__exit__(None, None, None)
    pass