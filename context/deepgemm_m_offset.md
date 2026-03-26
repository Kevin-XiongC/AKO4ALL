# DeepGEMM m-offset Layout

## packed_layout Format

`packed_layout` is a `torch.int32` tensor of shape `[2 * num_groups]`:
- First half `[0..G-1]`: `m_offsets` — starting row index for each expert (aligned to 128)
- Second half `[G..2G-1]`: `m_counts` — actual token count for each expert

## Example

4 groups with token counts [1, 129, 127, 128]:

```
m_offsets: [0, 128, 384, 512]     # aligned to 128
m_counts:  [1, 129, 127, 128]     # actual counts
packed_layout: [0, 128, 384, 512, 1, 129, 127, 128]
```

## API

```python
deep_gemm.m_grouped_fp8_gemm_nt_moffset(
    a, b, d, packed_layout, expected_m_per_group,
    disable_ue8m0_cast=disable_ue8m0_cast
)
```

- `a`: input in m-offset layout `[total_M, K]`
- `b`: weight `[num_groups, N, K]`
- `d`: output in m-offset layout `[total_M, N]`
- `packed_layout`: as described above

## Key Properties

1. **Alignment**: m_offsets are multiples of 128; gaps between groups contain padding rows.
2. **Padding**: rows between `m_count` and the next aligned boundary are never read by gather —
   GEMM processes them but those output rows are discarded. This means `sorted_hidden` can
   use `torch.empty` (no need to zero the buffer).
3. **Pre-allocation**: `max_total_M = bs * topk + num_groups * (alignment - 1)` is an
   upper bound; the buffer is allocated once and reused across forward passes.
