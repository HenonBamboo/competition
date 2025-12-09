# MindNLP 模型优化详细说明 (Qwen)

本文档详细记录了针对 Qwen 模型的关键性能优化点，并附带了相应的核心代码实现。

## 1. Qwen2-VL 模型优化

### 1.1 切片操作优化 (针对 Qwen2-VL)

优化痛点：

1. **RoPE 切片操作**：在旋转位置编码（RoPE）的 `rotate_half` 函数中，使用切片索引 `x[..., :half]` 和 `x[..., half:]` 可能导致非连续内存访问，增加数据搬运开销。
2. **`_prepare_4d_causal_attention_mask_with_cache_position` 切片操作**：通过`ops.narrow()`和`view()`函数处理切片与增维逻辑，同时依据原有代码处理带`masked_fill()`的切片逻辑，存在非连续内存访问及操作冗余的问题。
3. **Qwen2RMSNorm 速度问题**：原 Qwen2RMSNorm 的计算逻辑分散拆分，未利用框架原生算子的融合能力，导致执行速度较慢。

改进方案：

1. **针对 RoPE 切片操作**：使用 `ops.split` 替代手动切片，在图编译层面更易优化，减少 strided slice 带来的开销。
2. **针对`_prepare_4d_causal_attention_mask_with_cache_position`切片操作**：简化`ops.narrow()`与`view()`的嵌套逻辑，优先选取连续维度执行切片；处理`masked_fill()`时直接在原 Tensor 上操作，移除不必要的内存拷贝。
3. **针对 Qwen2RMSNorm 速度问题**：采用框架自带的 `F.rms_norm` 算法进行融合计算，替换原有分散的计算逻辑，提升执行速度。

**源码实现** (`mindnlp/transformers/models/qwen2_vl/modeling_qwen2_vl.py`)：

**Python**

```python
# 1. rotate_half优化
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # 修改前: x1 = x[..., : x.shape[-1] // 2]; x2 = x[..., x.shape[-1] // 2 :]
    # 修改后: 使用 ops.split
    x1, x2 = ops.split(x, x.shape[-1] // 2, dim=-1)
    return ops.cat((-x2, x1), dim=-1)

# 2. _prepare_4d_causal_attention_mask_with_cache_position优化
# 修改前: 
padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
# 修改后：
padding_mask = ops.narrow(causal_mask, -1, 0, mask_length) + attention_mask.view(attention_mask.shape[0], 1,
                                                                                             1, attention_mask.shape[1])
if mask_length >= causal_mask.shape[-1]:
    causal_mask = ops.masked_fill(causal_mask, padding_mask, min_dtype)
else:
    causal_mask = ops.cat(
        [ops.masked_fill(ops.narrow(causal_mask, -1, 0, mask_length), padding_mask, min_dtype),
         ops.narrow(causal_mask, -1, mask_length, causal_mask.shape[-1] - mask_length)],
        dim=-1
    )

# 3. Qwen2RMSNorm算子融合
# 修改前:
input_dtype = hidden_states.dtype
hidden_states = hidden_states.to(mindspore.float32)
variance = ops.mean(hidden_states.pow(2), -1, keepdim=True)
hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
return self.weight * hidden_states.to(input_dtype)

# 修改后:
if not self.training and use_pyboost() and not ON_ORANGE_PI:
    return F.rms_norm(hidden_states, self.weight, self.variance_epsilon)
input_dtype = hidden_states.dtype
hidden_states = hidden_states.to(mindspore.float32)
variance = ops.mean(hidden_states.pow(2), -1, keepdim=True)
hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
return self.weight * hidden_states.to(input_dtype)
```


## 2. 评测结果

| 评测指标 | 平均得分 |
|---------|---------|
| 峰值显存得分 | 100 |
| Prefill时延得分 | 107.6535 |
| Decode时延得分 | 112.7979 |
| **总分** | **106.8171** |
