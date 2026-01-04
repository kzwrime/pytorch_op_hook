# PyTorch Operator Hook

> 探索和验证各种 PyTorch 算子劫持技术的实验性项目

## 项目宗旨

本项目旨在系统地探索、验证和比较各种劫持（hook）PyTorch 算子的方法。通过实际的代码示例和测试用例，帮助开发者理解：

- 如何在算子执行前后注入自定义逻辑
- 如何替换或修改算子的默认实现
- 如何在不修改模型代码的情况下优化或调试 PyTorch 程序
- 不同劫持方法的性能影响和兼容性差异

## 支持的劫持方法

### 已实现

- ✅ **TorchDispatchMode** - 通过 PyTorch 的调度机制拦截算子调用
  - 普通函数劫持
  - 编译模式兼容（torch.compile）
  - 自定义算子实现（如 NumPy 实现）

### 计划实现

- ⏳ **Impl Override** - 直接覆盖 ATen 算子的底层实现
- ⏳ **Pattern Rewrite** - 通过模式匹配和重写来替换算子
- ⏳ **Registering Custom Ops** - 注册自定义算子替换原生算子
- ⏳ **TorchDynamo Hooks** - 在 Dynamo 编译过程中拦截算子
- ⏳ **Backend Extensions** - 通过后端扩展机制劫持算子
- ⏳ **Python-based Hook** - 使用 Python 钩子函数（如 `register_forward_hook`）

## 项目结构

```
pytorch_op_hook/
├── examples/
│   ├── 01_TorchDispatchMode/     # TorchDispatchMode 示例
│   │   ├── test_dispatch_py.py           # 基础用法示例
│   │   ├── test_dispatch_py_with_compile.py  # 与 torch.compile 兼容性测试
│   │   └── verify_numpy_op.py             # NumPy 实现验证
│   ├── 02_ImplOverride/          # Impl 覆盖方法（待实现）
│   ├── 03_PatternRewrite/        # Pattern Rewrite 示例（待实现）
│   └── ...                        # 更多方法示例
├── docs/                          # 详细文档（待添加）
├── tests/                         # 单元测试（待添加）
└── README.md                      # 本文件
```

## 快速开始

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- NumPy

### 安装依赖

```bash
pip install torch numpy
```

### 运行示例

#### 示例 1: TorchDispatchMode 基础用法

```bash
python examples/01_TorchDispatchMode/test_dispatch_py.py
```

输出：
```
--- Entering Hijack Mode ---
[Hijack] Captured torch.mm! Shapes: torch.Size([3, 4]) x torch.Size([4, 5])
[Hijack] Replacing with numpy-based custom op implementation
[Hijack] Captured torch.mm! Shapes: torch.Size([3, 4]) x torch.Size([4, 5])
[Hijack] Replacing with numpy-based custom op implementation

--- Exiting Mode ---
```

#### 示例 2: 与 torch.compile 兼容性测试

```bash
python examples/01_TorchDispatchMode/test_dispatch_py_with_compile.py
```

#### 示例 3: 验证自定义实现的正确性

```bash
python examples/01_TorchDispatchMode/verify_numpy_op.py
```

## 使用示例

### TorchDispatchMode 示例

```python
import torch
import numpy as np
from torch.utils._python_dispatch import TorchDispatchMode

# 定义自定义算子实现
def numpy_matmul_custom_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """使用 NumPy 实现矩阵乘法"""
    a_np = a.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    result_np = np.matmul(a_np, b_np)
    result = torch.from_numpy(result_np).to(device=a.device, dtype=a.dtype)
    return result

# 定义劫持模式
class HijackMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func.overloadpacket == torch.ops.aten.mm:
            print(f"[Hijack] 捕获 torch.mm! 形状: {args[0].shape} x {args[1].shape}")
            return numpy_matmul_custom_op(args[0], args[1])
        return func(*args, **kwargs)

# 使用劫持模式
x = torch.randn(3, 4)
w = torch.randn(4, 5)

hmd = HijackMode()
with hmd:
    y = torch.mm(x, w)  # 会被 HijackMode 拦截并替换为 NumPy 实现
```

## 应用场景

算子劫持技术可以用于：

1. **性能优化** - 替换低效的算子实现
2. **调试和监控** - 记录算子调用信息、输入输出形状、执行时间等
3. **自定义实现** - 使用第三方库（如 NumPy、CuBLAS）替代原生实现
4. **模型修改** - 在不修改模型源码的情况下修改算子行为
5. **A/B 测试** - 比较不同实现方式的性能差异
6. **研究和实验** - 测试新的算子实现或优化策略

## 方法对比

| 方法 | 兼容性 | 性能影响 | 实现难度 | 适用场景 |
|------|--------|----------|----------|----------|
| TorchDispatchMode | ⭐⭐⭐⭐ | 低 | 低 | 通用劫持、调试 |
| Impl Override | ⭐⭐ | 中 | 中 | 底层优化 |
| Pattern Rewrite | ⭐⭐⭐ | 低 | 高 | 图优化 |
| TorchDynamo Hooks | ⭐⭐⭐⭐ | 低 | 中 | 编译时优化 |
| Backend Extensions | ⭐⭐⭐ | 低 | 高 | 生产环境 |
| Python Hooks | ⭐⭐⭐⭐⭐ | 低 | 低 | 模型级操作 |

## 贡献指南

欢迎贡献新的劫持方法示例、测试用例或改进建议！

### 如何贡献

1. Fork 本仓库
2. 创建新的分支：`git checkout -b feature/your-method`
3. 在 `examples/` 下创建新的目录，如 `04_YourMethod/`
4. 添加示例代码、测试脚本和 README
5. 提交 Pull Request

### 贡献类型

- 新的劫持方法示例
- 现有方法的改进
- 性能基准测试
- 文档改进
- Bug 修复

## 测试

```bash
# 运行所有测试（待实现）
pytest tests/

# 运行特定方法测试
pytest tests/test_torch_dispatch_mode.py
```

## 路线图

- [ ] 实现所有计划的劫持方法
- [ ] 添加性能基准测试
- [ ] 添加详细的技术文档
- [ ] 提供方法选择的决策指南
- [ ] 添加更多实际应用场景示例
- [ ] 支持更多 PyTorch 版本

## 常见问题

### Q: 这些方法可以在生产环境使用吗？

A: 部分方法（如 TorchDispatchMode）相对稳定，但建议在生产环境使用前充分测试。某些方法可能影响性能或兼容性。

### Q: 哪种方法最好？

A: 没有绝对最好的方法，取决于具体场景：
- 调试和监控：TorchDispatchMode 或 Python Hooks
- 性能优化：Impl Override 或 Backend Extensions
- 编译优化：Pattern Rewrite 或 TorchDynamo Hooks

### Q: torch.compile 兼容性如何？

A: 不同方法对 torch.compile 的兼容性不同。TorchDispatchMode 在大多数情况下可以工作，但某些方法可能会导致编译失败。建议在使用前进行测试。

## 参考资料

- [PyTorch Dispatcher Documentation](https://pytorch.org/docs/stable/dispatcher.html)
- [TorchDispatchMode Guide](https://pytorch.org/tutorials/advanced/dispatcher.html)
- [PyTorch Custom Operators Guide](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)

## 许可证

MIT License

## 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。

---

**注意**: 本项目为实验性项目，用于学习和研究目的。在生产环境使用前请充分测试。
