# TorchDispatchMode 算子劫持

本目录展示如何使用 `TorchDispatchMode` 来劫持 PyTorch 算子。

## 概述

`TorchDispatchMode` 是 PyTorch 2.0 引入的一个强大工具，允许开发者在算子调度层面拦截和修改 PyTorch 操作。与传统的钩子方法相比，它提供了更细粒度的控制和更广泛的适用性。

## 核心特性

- ✅ **全面拦截** - 可以拦截几乎所有 PyTorch 算子，包括 `torch.mm`、`torch.matmul` 等
- ✅ **编译兼容** - 与 `torch.compile` 兼容，可以在编译模式下工作
- ✅ **透明替换** - 可以完全替换算子的实现，而无需修改模型代码
- ✅ **灵活控制** - 可以基于算子类型、输入形状等条件决定是否拦截
- ✅ **性能友好** - 拦截开销相对较小，适合调试和监控

## 文件说明

### test_dispatch_py.py

最基础的 TorchDispatchMode 使用示例，展示如何：

1. 创建自定义的 DispatchMode 类
2. 拦截 `torch.mm` 操作
3. 使用 NumPy 实现替换原生实现
4. 在模式内外对比行为差异

**运行：**
```bash
python test_dispatch_py.py
```

### test_dispatch_py_with_compile.py

展示 TorchDispatchMode 与 `torch.compile` 的兼容性测试，包括：

1. 普通编译模式测试
2. 在 DispatchMode 中编译函数
3. fullgraph 编译模式测试

**运行：**
```bash
python test_dispatch_py_with_compile.py
```

### verify_numpy_op.py

验证自定义 NumPy 实现的正确性，包含：

1. 与原生 `torch.mm` 的数值对比
2. 不同数据类型测试（float32, float64）
3. 不同设备测试（CPU, CUDA）
4. 通过 HijackMode 的集成测试

**运行：**
```bash
python verify_numpy_op.py
```

## 实现原理

### TorchDispatchMode 工作机制

```
用户代码
   ↓
TorchDispatchMode.__torch_dispatch__
   ↓
检查是否为目标算子
   ↓
   ├─ 是 → 执行自定义逻辑
   │         ↓
   │      返回自定义结果
   │
   └─ 否 → 调用原始算子 func(*args, **kwargs)
             ↓
          返回原始结果
```

### 关键代码解析

```python
from torch.utils._python_dispatch import TorchDispatchMode

class HijackMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 1. 检查算子类型
        if func.overloadpacket == torch.ops.aten.mm:
            # 2. 执行自定义逻辑
            return custom_implementation(args[0], args[1])

        # 3. 其他算子正常执行
        return func(*args, **kwargs)
```

**要点：**
- `func.overloadpacket` - 算子的唯一标识符
- `args[0]` - 第一个输入张量
- `args[1]` - 第二个输入张量
- 必须调用 `func()` 来执行原始算子（避免递归）

## 使用场景

### 1. 性能监控

```python
class ProfilingMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.overloadpacket}: {elapsed:.6f}s")
        return result
```

### 2. 输入输出检查

```python
class ValidationMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 检查输入
        for arg in args:
            if isinstance(arg, torch.Tensor):
                if torch.isnan(arg).any():
                    raise ValueError(f"Input contains NaN")

        result = func(*args, **kwargs)

        # 检查输出
        if isinstance(result, torch.Tensor):
            if torch.isnan(result).any():
                raise ValueError(f"Output contains NaN")

        return result
```

### 3. 自定义实现替换

```python
class CustomImplMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func.overloadpacket == torch.ops.aten.mm:
            return my_faster_matmul(args[0], args[1])
        return func(*args, **kwargs)
```

### 4. 调试和日志

```python
class DebugMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print(f"Calling {func.overloadpacket}")
        print(f"  Input shapes: {[a.shape for a in args if isinstance(a, torch.Tensor)]}")
        result = func(*args, **kwargs)
        if isinstance(result, torch.Tensor):
            print(f"  Output shape: {result.shape}")
        return result
```

## 常见算子的 overloadpacket

| 算子 | overloadpacket | 说明 |
|------|----------------|------|
| `torch.mm` | `torch.ops.aten.mm` | 矩阵乘法 |
| `torch.matmul` | `torch.ops.aten.matmul` | 广义矩阵乘法 |
| `torch.add` | `torch.ops.aten.add.Tensor` | 张量加法 |
| `torch.mul` | `torch.ops.aten.mul.Tensor` | 张量乘法 |
| `torch.nn.functional.linear` | `torch.ops.aten.linear` | 线性层 |
| `torch.conv2d` | `torch.ops.aten.conv2d` | 2D 卷积 |

完整的算子列表可以通过 `torch.ops` 查看。

## 限制和注意事项

### 1. 不能在 __torch_dispatch__ 中使用被拦截的算子

```python
# ❌ 错误：会导致无限递归
def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    if func.overloadpacket == torch.ops.aten.mm:
        return torch.mm(args[0], args[1])  # 会再次被拦截
```

```python
# ✅ 正确：使用 func() 调用原始实现
def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    if func.overloadpacket == torch.ops.aten.mm:
        return func(*args, **kwargs)  # 不会递归
```

### 2. 返回值必须与原始算子兼容

确保自定义实现的返回值类型、形状、dtype 与原始算子一致。

### 3. torch.compile 限制

虽然 TorchDispatchMode 与 torch.compile 基本兼容，但在某些复杂场景下可能遇到问题。建议充分测试。

### 4. 性能开销

虽然 TorchDispatchMode 的开销相对较小，但在高频调用的场景下仍需注意性能影响。

## 与其他方法的对比

| 特性 | TorchDispatchMode | Python Hooks | Impl Override |
|------|-------------------|--------------|---------------|
| 实现难度 | 低 | 低 | 中 |
| 拦截范围 | 所有算子 | 仅模型层 | ATen 算子 |
| torch.compile 兼容 | 是 | 部分 | 否 |
| 性能影响 | 小 | 小 | 中 |
| 适用场景 | 通用劫持 | 模型级操作 | 深度优化 |

## 进阶技巧

### 1. 条件拦截

```python
class ConditionalMode(TorchDispatchMode):
    def __init__(self, min_size=1024):
        super().__init__()
        self.min_size = min_size

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func.overloadpacket == torch.ops.aten.mm:
            # 只劫持大矩阵乘法
            if args[0].numel() > self.min_size:
                return custom_impl(args[0], args[1])
        return func(*args, **kwargs)
```

### 2. 嵌套模式

```python
mode1 = HijackMode1()
mode2 = HijackMode2()

with mode1:
    with mode2:
        y = torch.mm(x, w)  # mode2 先拦截，然后 mode1
```

### 3. 统计和监控

```python
class StatsMode(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.stats = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        op_name = str(func.overloadpacket)
        self.stats[op_name] = self.stats.get(op_name, 0) + 1
        return func(*args, **kwargs)

    def print_stats(self):
        for op, count in sorted(self.stats.items(), key=lambda x: -x[1]):
            print(f"{op}: {count} calls")
```

## 调试技巧

### 1. 打印所有被调用的算子

```python
class LoggingMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print(f"Called: {func.overloadpacket}")
        return func(*args, **kwargs)
```

### 2. 查看调用栈

```python
import traceback

class TracingMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func.overloadpacket == torch.ops.aten.mm:
            print("Call stack:")
            traceback.print_stack()
        return func(*args, **kwargs)
```

## 参考资料

- [PyTorch Dispatcher Tutorial](https://pytorch.org/tutorials/advanced/dispatcher.html)
- [TorchDispatchMode API Documentation](https://pytorch.org/docs/stable/dispatcher.html)
- [PyTorch Internals: Dispatcher](https://github.com/pytorch/pytorch/tree/main/torch/csrc/dispatcher)

## 贡献

欢迎提交改进建议、新的使用示例或 bug 报告！
