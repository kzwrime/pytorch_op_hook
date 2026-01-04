import torch
import numpy as np
from torch.utils._python_dispatch import TorchDispatchMode


def numpy_linear_custom_op(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """使用 NumPy 实现的线性层算子"""

    print(f"Func numpy_linear_custom_op")

    input_shape = input.shape
    in_features = input_shape[-1]
    batch_dims = input_shape[:-1]

    input_flat = input.reshape(-1, in_features)

    input_np = input_flat.detach().cpu().numpy()
    weight_np = weight.detach().cpu().numpy()

    # 根据形状判断是否需要转置
    if weight_np.shape[0] == input_np.shape[1]:
        result_np = np.matmul(input_np, weight_np)
    else:
        result_np = np.matmul(input_np, weight_np.T)

    if bias is not None:
        bias_np = bias.detach().cpu().numpy()
        result_np = result_np + bias_np

    device = input.device
    dtype = input.dtype
    result = torch.from_numpy(result_np).to(device=device, dtype=dtype)

    out_features = result_np.shape[1]
    output_shape = batch_dims + (out_features,)
    result = result.reshape(output_shape)

    return result


class HijackMode(TorchDispatchMode):
    """
    劫持 linear 操作的自定义模式
    注意：PyTorch 的 linear 在底层使用 addmm 算子实现
    """

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 拦截 linear 算子
        if func.overloadpacket == torch.ops.aten.linear:
            print(f"[Hijack] Captured torch.ops.aten.linear")
            print(f"[Hijack]   Input shape: {args[0].shape}")
            print(f"[Hijack]   Weight shape: {args[1].shape}")
            print(f"[Hijack]   Bias: {args[2] is not None}")

            input_tensor = args[0]
            weight = args[1]
            bias = args[2] if len(args) > 2 else None

            return numpy_linear_custom_op(input_tensor, weight, bias)

        return func(*args, **kwargs)


# --- Test ---
torch.manual_seed(42)
x = torch.randn(3, 4)
weight = torch.randn(5, 4)
bias = torch.randn(5)

# Test 1: Baseline test without hijacking
print("=" * 80)
print("Test 1: Baseline - Call simple_function directly (no hijacking)")
print("=" * 80)

def simple_function(x, weight, bias):
    return torch.nn.functional.linear(x, weight, bias)

print("Calling simple_function...")
with torch.inference_mode():
    result = simple_function(x, weight, bias)
print(f"Result shape: {result.shape}")
print("Test 1 completed\n")

# Test 2: HijackMode without compile
print("=" * 80)
print("Test 2: HijackMode without compile")
print("=" * 80)

with HijackMode():
    print("Calling simple_function with HijackMode...")
    with torch.inference_mode():
        result = simple_function(x, weight, bias)

print(f"Result shape: {result.shape}")
print("Test 2 completed\n")

# Test 3: torch.compile without hijacking
print("=" * 80)
print("Test 3: torch.compile without hijacking")
print("=" * 80)

compiled_fn = torch.compile(simple_function)

print("Calling compiled_fn...")
with torch.inference_mode():
    result = compiled_fn(x, weight, bias)

print(f"Result shape: {result.shape}")
print("Test 3 completed\n")

# Test 4: HijackMode + torch.compile
print("=" * 80)
print("Test 4: HijackMode + torch.compile")
print("=" * 80)

try:
    with HijackMode():
        compiled_fn = torch.compile(simple_function)

        print("Calling compiled_fn with HijackMode...")
        with torch.inference_mode():
            result = compiled_fn(x, weight, bias)

    print(f"Result shape: {result.shape}")
    print("Test 4 completed\n")
except Exception as e:
    print(f"Test 4 failed: {e}")
    print("This might indicate that HijackMode doesn't work well with torch.compile\n")

# Test 5: Full graph capture with HijackMode
print("=" * 80)
print("Test 5: Full graph capture with HijackMode")
print("=" * 80)

try:
    with HijackMode():
        compiled_fn_full = torch.compile(simple_function, fullgraph=True)

        print("Calling compiled_fn_full (fullgraph) with HijackMode...")
        with torch.inference_mode():
            result = compiled_fn_full(x, weight, bias)

    print(f"Result shape: {result.shape}")
    print("Test 5 completed\n")
except Exception as e:
    print(f"Test 5 failed: {e}")
    print("Skipping fullgraph test\n")


# --- Test with nn.Linear layer ---
print("\n" + "=" * 80)
print("Test 6-7: Testing torch.nn.Linear with torch.compile")
print("=" * 80)

torch.manual_seed(42)
linear_layer = torch.nn.Linear(4, 5, bias=True)
linear_layer.eval()

x = torch.randn(3, 4)

# Test 6: nn.Linear with compile (no hijacking)
print("\nTest 6: nn.Linear with torch.compile (no hijacking)")

def model_function(x):
    return linear_layer(x)

compiled_model = torch.compile(model_function)

print("Calling compiled_model...")
with torch.inference_mode():
    result = compiled_model(x)

print(f"Result shape: {result.shape}")
print("Test 6 completed\n")

# Test 7: nn.Linear with HijackMode + compile
print("Test 7: nn.Linear with HijackMode + torch.compile")

try:
    with HijackMode():
        compiled_model = torch.compile(model_function)

        print("Calling compiled_model with HijackMode...")
        with torch.inference_mode():
            result = compiled_model(x)

    print(f"Result shape: {result.shape}")
    print("Test 7 completed\n")
except Exception as e:
    print(f"Test 7 failed: {e}\n")


# --- Test with a more complex model ---
print("\n" + "=" * 80)
print("Test 8-9: Testing complex model with torch.compile")
print("=" * 80)

torch.manual_seed(42)


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 8, bias=True)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(8, 5, bias=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


model = SimpleModel()
model.eval()
x = torch.randn(3, 4)

# Test 8: Complex model with compile (no hijacking)
print("\nTest 8: Complex model with torch.compile (no hijacking)")

compiled_model = torch.compile(model)

print("Calling compiled_model (complex)...")
with torch.inference_mode():
    result = compiled_model(x)

print(f"Result shape: {result.shape}")
print("Test 8 completed\n")

# Test 9: Complex model with HijackMode + compile
print("Test 9: Complex model with HijackMode + torch.compile")
print("This should intercept both linear layers!")

try:
    with HijackMode():
        compiled_model = torch.compile(model)

        print("Calling compiled_model (complex) with HijackMode...")
        with torch.inference_mode():
            result = compiled_model(x)

    print(f"Result shape: {result.shape}")
    print("Test 9 completed\n")
except Exception as e:
    print(f"Test 9 failed: {e}\n")

print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)
print("This script tests whether linear operator hijacking works with torch.compile")
print("\nKey observations:")
print("  1. TorchDispatchMode may or may not work with torch.compile")
print("  2. Behavior depends on PyTorch version and compilation settings")
print("  3. Some modes (fullgraph) may have different compatibility")
print("=" * 80)
