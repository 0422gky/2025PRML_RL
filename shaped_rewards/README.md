# Reward Shaping for PPO

这里存放的是 reward shaping 修改过后的 PPO trainer，主要修改是引入了重复惩罚和长度惩罚来防止 reward hacking。

## 改进内容

### 1. 奖励正则化（防止 hacking）

#### 长度惩罚
- **公式**: `scores -= length_penalty_coef * length`
- **作用**: 惩罚过长的响应，鼓励模型生成更简洁的回答
- **默认系数**: 0.01

#### 重复惩罚
- **公式**: `scores -= repetition_penalty_coef * repetition_ratio`
- **作用**: 基于 n-gram 重复率惩罚，防止模型生成重复内容
- **默认系数**: 0.1
- **默认 n-gram**: 3

### 2. 实现位置

在 `ppo_trainer_shaped.py` 的 `train()` 方法中，在计算 reward model scores 之后应用这些惩罚：

```python
# Response Processing 3 之后
# 应用长度惩罚和重复惩罚
length_penalty_coef = getattr(self.args, 'length_penalty_coef', 0.01)
repetition_penalty_coef = getattr(self.args, 'repetition_penalty_coef', 0.1)
```

## 使用方法

### 在 baseline.py 中使用

修改 `baseline.py` 的导入和配置：

```python
# 导入修改后的 trainer
import sys
sys.path.insert(0, './shaped_rewards')
from ppo_trainer_shaped import PPOTrainer

# 在 PPOConfig 中添加惩罚参数（可选，有默认值）
training_args = PPOConfig(
    ...
    length_penalty_coef=0.01,        # 长度惩罚系数
    repetition_penalty_coef=0.1,     # 重复惩罚系数
    repetition_n_gram=3,             # n-gram 大小
    ...
)
```

## 未来改进方向

1. **奖励塑性（Reward Shaping）**
   - 让 reward 不止在最后出现，每生成 k 个 token 给一次局部奖励
   - 用 verifier 给 dense reward

2. **更精细的重复检测**
   - 支持不同粒度的 n-gram
   - 考虑句子级别的重复
