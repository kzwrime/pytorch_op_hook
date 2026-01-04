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
        if func.overloadpacket == torch.ops.aten.mm:
            print(f"[Hijack] Captured torch.mm! Shapes: {args[0].shape} x {args[1].shape}")
            return numpy_matmul_custom_op(args[0], args[1])
        return func(*args, **kwargs)

# Verification test
print("=" * 60)
print("Verification Test: NumPy Custom Op vs Original torch.mm")
print("=" * 60)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create test tensors
x = torch.randn(3, 4)
w = torch.randn(4, 5)

print(f"\nInput shapes: x={x.shape}, w={w.shape}")

# Test 1: Original torch.mm
print("\n[Test 1] Original torch.mm:")
result_original = torch.mm(x, w)
print(f"Result shape: {result_original.shape}")
print(f"Result sample values:\n{result_original}")

# Test 2: NumPy custom op
print("\n[Test 2] NumPy custom op (direct call):")
result_numpy = numpy_matmul_custom_op(x, w)
print(f"Result shape: {result_numpy.shape}")
print(f"Result sample values:\n{result_numpy}")

# Test 3: Through HijackMode
print("\n[Test 3] Through HijackMode:")
hmd = HijackMode()
hmd.__enter__()
result_hijack = torch.mm(x, w)
hmd.__exit__(None, None, None)
print(f"Result shape: {result_hijack.shape}")
print(f"Result sample values:\n{result_hijack}")

# Verification
print("\n" + "=" * 60)
print("Verification Results:")
print("=" * 60)

# Check if results are close
tolerance = 1e-5
match_original_numpy = torch.allclose(result_original, result_numpy, rtol=tolerance, atol=tolerance)
match_original_hijack = torch.allclose(result_original, result_hijack, rtol=tolerance, atol=tolerance)
match_numpy_hijack = torch.allclose(result_numpy, result_hijack, rtol=tolerance, atol=tolerance)

print(f"Original vs NumPy direct:  {'✓ PASS' if match_original_numpy else '✗ FAIL'}")
print(f"Original vs HijackMode:    {'✓ PASS' if match_original_hijack else '✗ FAIL'}")
print(f"NumPy direct vs HijackMode: {'✓ PASS' if match_numpy_hijack else '✗ FAIL'}")

if match_original_numpy and match_original_hijack and match_numpy_hijack:
    print("\n✓ All tests PASSED! NumPy implementation matches torch.mm")
else:
    print("\n✗ Some tests FAILED! Results don't match")
    print(f"\nMax difference (original vs numpy): {(result_original - result_numpy).abs().max().item()}")
    print(f"Max difference (original vs hijack): {(result_original - result_hijack).abs().max().item()}")

# Test with different dtypes and devices
print("\n" + "=" * 60)
print("Additional Tests: Different Data Types and Devices")
print("=" * 60)

# Test with float64
print("\n[Test 4] Float64 dtype:")
x64 = x.to(torch.float64)
w64 = w.to(torch.float64)
result_original_64 = torch.mm(x64, w64)
result_numpy_64 = numpy_matmul_custom_op(x64, w64)
match_64 = torch.allclose(result_original_64, result_numpy_64, rtol=tolerance, atol=tolerance)
print(f"Float64 test: {'✓ PASS' if match_64 else '✗ FAIL'}")

# Test with CUDA if available
if torch.cuda.is_available():
    print("\n[Test 5] CUDA device:")
    x_cuda = x.cuda()
    w_cuda = w.cuda()
    result_original_cuda = torch.mm(x_cuda, w_cuda)
    result_numpy_cuda = numpy_matmul_custom_op(x_cuda, w_cuda)
    match_cuda = torch.allclose(result_original_cuda, result_numpy_cuda, rtol=tolerance, atol=tolerance)
    print(f"CUDA test: {'✓ PASS' if match_cuda else '✗ FAIL'}")
else:
    print("\n[Test 5] CUDA not available, skipping CUDA test")

print("\n" + "=" * 60)
print("Verification Complete!")
print("=" * 60)
