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

hmd.__enter__()

print("--- Entering Hijack Mode ---")
with torch.inference_mode():
    # Here we use regular torch.Tensor
    # Whether using torch.mm(x, w) or x.mm(w), or even Linear layers inside models, all will be captured
    y = torch.mm(x, w)
    z = x.mm(w)

hmd.__exit__(None, None, None)

print("\n--- Exiting Mode ---")
# This call will not be intercepted
y = torch.mm(x, w)

