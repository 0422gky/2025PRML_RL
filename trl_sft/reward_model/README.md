# Reward Model 训练和使用指南

## 训练 Reward Model

运行 `reward_model.py` 来训练一个基于 Qwen 的 reward model：

```bash
cd /workspace/trl_sft/reward_model
python reward_model.py
```

训练完成后，模型会保存在 `./Qwen2.5-0.5B-Instruct-Reward/` 目录。

## 在 PPO 训练中使用

训练好的 reward model 已经自动集成到 `baseline.py` 中：

1. **自动检测**: `baseline.py` 会自动检查 `./reward_model/Qwen2.5-0.5B-Instruct-Reward/` 是否存在
2. **Tokenizer 兼容**: 使用基于 Qwen 的 reward model 可以避免 tokenizer 不匹配的问题
3. **备选方案**: 如果训练好的模型不存在，会自动使用公开的 Pythia reward model（但可能有 tokenizer 不兼容问题）

## 优势

- ✅ **Tokenizer 兼容**: 与 Qwen policy model 使用相同的 tokenizer，避免 CUDA device-side assert 错误
- ✅ **领域适配**: 可以根据自己的数据集训练，更适合特定任务
- ✅ **模型一致性**: 使用相同的 base model，确保架构兼容

## 注意事项

- 确保 reward model 训练完成后再运行 `baseline.py`
- 如果训练中断，可以手动指定 reward model 路径
- 建议保存训练好的 reward model 到安全位置
