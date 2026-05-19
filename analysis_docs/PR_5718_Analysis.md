# FBGEMM PR #5718 分析报告

## 基本信息

| 属性 | 内容 |
|------|------|
| **标题** | Optimize jagged_unique_indices_cuda (binary-search length + custom cub pipeline) |
| **合并日期** | 2026-05-06 |
| **变更规模** | +541 / -130 行 |
| **涉及文件** | 3 个 |
| **关联 PR** | [pytorch/FBGEMM#5718](https://github.com/pytorch/FBGEMM/pull/5718), facebookresearch 内部 #2651 |

---

## 功能目标

对 `fbgemm_gpu` 中的 `jagged_unique_indices_cuda` 算子进行端到端性能优化，该算子用于推荐系统推理中的 jagged tensor 去重索引计算。优化目标平台为 NVIDIA H100 GPU，解决生产环境下 SM 利用率低、冗余内存扫描等问题。

---

## 原有实现的问题

原始 pipeline 由 4 个阶段组成，存在以下瓶颈：

```
线性化索引 → at::_unique → 反线性化 → 计算唯一索引长度
```

| 阶段 | 问题 |
|------|------|
| **linearize_index_wo_infos_kernel** | Grid = ⌈total_B / kMaxThreads⌉，生产 shape 下仅启动 ~5 个 block，H100 上 ~96% SM 空闲 |
| **at::_unique** | 使用 int64 key 做全位宽排序，即使 hash space 小于 INT32_MAX 也浪费算力；且是 PyTorch 通用实现，无法利用 domain 约束 |
| **delinearize_unique_index_kernel** | 遍历 total_indices（~数千万），通过 scatter 写入 unique_indices |
| **unique_indices_length_kernel** | 对 reverse_index 做 O(N) BlockReduce 扫描求 min/max，额外扫描 ~800MB 数据 |

---

## 优化方案详解

### 优化 1: 二分搜索重写 unique_indices_length_kernel

**核心思路**: 利用线性化索引的不变量 —— feature t 的唯一值在排序后占据连续区间 `[hash_size_cumsum[t], hash_size_cumsum[t+1])`，因此每个 feature 的长度可通过两次二分搜索直接计算。

| 维度 | 旧实现 | 新实现 |
|------|--------|--------|
| 算法 | O(N) BlockReduce 扫描 reverse_index 的 min/max | 两次 O(log U) 二分搜索 on sorted unique_indices |
| Block size | 1024（含共享内存 reduction） | 256（无 reduction，仅写输出） |
| 输入依赖 | reverse_index + offsets + max/min 模板参数 | hash_size_cumsum + linear_unique_indices |
| 内存开销 | 扫描 ~800MB reverse_index | 仅读 hash_size_cumsum + sorted unique_indices |

新增 `device_lower_bound` helper 函数，在 GPU 端执行标准 lower_bound 二分搜索。每个 block 处理一组 feature，计算 `total_length / num_lengths` 的整数除法和余数分配。

### 优化 2: Flat-grid 线性化索引 kernel

**核心思路**: 将 `linearize_index_wo_infos_kernel` 替换为 `linearize_index_flat_kernel`，Grid 从 `⌈total_B/kMaxThreads⌉` 改为 `total_B`（每个 (t, b) 样本一个 block）。

```
旧: Grid = ⌈total_B / 1024⌉ ≈ 5 blocks  → 96% SM 空闲
新: Grid = total_B ≈ 数千~数万 blocks    → 充分利用所有 SM
```

额外改进：
- 模板参数化为 `<index_t, key_t>`，直接写入 int32 或 int64 目标类型，省去后续 cast pass
- 去除 warp-cooperative shuffle 广播逻辑，改为单 block 内线程直接处理

### 优化 3: 自定义 CUB pipeline 替代 at::_unique

这是最核心的优化，用自定义 CUB 原语替代 PyTorch 的 `at::_unique`：

```
cub::DeviceRadixSort::SortPairs  →  cub::DeviceRunLengthEncode::Encode
    → adjacent_diff + cub::DeviceScan::InclusiveSum + scatter  →  构建 inverse index
```

**关键优化点**：

1. **int32/int64 key 分派**: 当 `total_hash_size < INT32_MAX`（实际几乎总是如此）时使用 int32 key，大幅减少排序 kernel 工作量
2. **end_bit 裁剪**: 基数排序的位数限制为 `ceil(log2(total_hash_size + 1) + 1)`，而非完整的 64 位，减少排序 pass 数
3. **逆向索引构建**: 参照 PyTorch `UniqueCub.cu` 的模式：
   - `jagged_unique_adjacent_diff_kernel`: 计算相邻差分标记新值出现位置
   - `cub::DeviceScan::InclusiveSum`: 前缀和生成逆映射位置
   - `jagged_unique_scatter_kernel`: scatter 写入最终 reverse_index
4. **D→H 同步**: 一次 `item().toLong()` 获取 total_hash_size 用于 end_bit 计算，与原 `at::_unique` 的隐式同步数量持平（net-neutral）

### 优化 4: 基于二分搜索的反线性化

将 `delinearize_unique_index_kernel` 替换为 `delinearize_unique_from_sorted_kernel`：

```
旧: 遍历 total_indices (~数千万)，scatter 写入 unique_indices
新: 遍历 num_unique (~数百万)，每个线程通过 device_lower_bound 在 hash_size_cumsum 上二分搜索恢复 feature id
```

**收益**: 迭代规模从 total_indices 降至 num_unique，通常缩小 5-10 倍，且无需 24M 元素的 scatter 操作。

---

## 完整新 Pipeline

```
输入: hash_size_cumsum, hash_size_offsets, offsets, indices
  │
  ├─ Step F: linearize_index_flat_kernel
  │   Grid=total_B, 写入 key_t (int32/int64)
  │
  ├─ Step 3a: cub::DeviceRadixSort::SortPairs
  │   end_bit 裁剪, int32 key 分派
  │
  ├─ Step 3b: cub::DeviceRunLengthEncode::Encode
  │   提取排序后的唯一 key
  │
  ├─ Step 3c: 逆向索引构建
  │   ├─ jagged_unique_adjacent_diff_kernel
  │   ├─ cub::DeviceScan::InclusiveSum
  │   └─ jagged_unique_scatter_kernel
  │
  ├─ Step E: delinearize_unique_from_sorted_kernel
  │   遍历 num_unique, 二分搜索恢复 feature-local index
  │
  ├─ unique_indices_length_kernel (重写)
  │   二分搜索替代 BlockReduce 扫描
  │
  └─ asynchronous_complete_cumsum_gpu
      → 输出: (output_lengths, output_offsets, unique_indices, reverse_index)
```

---

## 输入校验

新增前置校验确保 pipeline 安全：

| 校验 | 目的 |
|------|------|
| `hash_size_cumsum.dtype() == indices.dtype()` | 保证 AT_DISPATCH 绑定单一 index_t |
| `hash_size_cumsum.numel() >= 2` | 确保 T >= 1，避免 FixedDivisor 除零 |
| `N <= INT32_MAX` | CUB API 和 int32 positions 的安全约束 |
| `total_hash_size >= 0` | 基数排序 end_bit 的合法性 |
| `total_hash_size_bits <= 63` | 防止 CUB pass 计数溢出 |

---

## 测试覆盖

新增随机化差分测试 `test_jagged_unique_indices_randomized_differential`：

- 使用 Hypothesis 库进行参数空间探索（B, F, max_length, per_feature_hash_bits）
- 以 `torch.unique(linear_indices, return_inverse=True)` 作为参考实现
- **覆盖 int32/int64 双路径**: 通过 `per_feature_hash_bits` 参数扫描（3~30 bit），驱动 `use_int32_keys` 的两个分支
- 验证项：
  - unique 基数一致性
  - (input_index, reverse_index) 映射正确性
  - output_lengths 求和等于 num_unique
  - 每个 feature 的 unique_indices 落在对应 hash 范围内

`failures_dict.json` 中注册了 faketensor / aot_dispatch_dynamic 的 xfail，因为这些测试使用 `.item()/.tolist()` 做 host 端比较，与 fake tensor 不兼容。

---

## 公共接口兼容性

**完全保持不变**: op 仍然返回 `(output_lengths, output_offsets, unique_indices, reverse_index)`，dtype 不变。`reverse_index` 保持 int64 以匹配 `at::_unique` 的历史行为。`unique_indices_length_kernel` 可无修改复用。

---

## 变更文件汇总

| 文件 | 变更 | 说明 |
|------|------|------|
| `fbgemm_gpu/src/jagged_tensor_ops/jagged_unique_indices.cu` | +411 / -130 | 核心实现：5 个新 kernel + pipeline 函数 + 输入校验 |
| `fbgemm_gpu/test/jagged/unique_indices_test.py` | +117 | 随机化差分测试 |
| `fbgemm_gpu/test/jagged/failures_dict.json` | +8 | faketensor/aot dispatch xfail 注册 |

---

## 优化效果总结

| 优化点 | 机制 | 预期收益 |
|--------|------|----------|
| Flat-grid 线性化 | total_B 个 block 替代 ~5 个 block | SM 利用率从 ~4% 提升至接近 100% |
| int32 key 排序 | 实际 hash space 通常 < INT32_MAX | 排序 kernel 工作量减半 |
| end_bit 裁剪 | 仅排序有效位数 | 减少基数排序 pass 数 |
| 二分搜索 length | O(log U) 替代 O(N) 扫描 | 消除 ~800MB reverse_index 扫描 |
| num_unique 反线性化 | 遍历规模缩小 5-10x | 减少 scatter 开销 |
