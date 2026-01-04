import torch
from torch.utils._python_dispatch import TorchDispatchMode

class HijackMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 1. Check if this is an operator we care about
        # func.overloadpacket contains operator metadata, e.g., torch.ops.aten.mm
        if func.overloadpacket == torch.ops.aten.mm:
            print(f"[Hijack] Captured torch.mm! Shapes: {args[0].shape} x {args[1].shape}")

            # You can modify input args here, or return custom results directly
            # return torch.ones(args[0].shape[0], args[1].shape[1])

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

