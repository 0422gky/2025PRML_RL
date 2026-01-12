"""
RLVR (Reinforcement Learning with Verifiable Rewards) 训练脚本

基于 baseline.py，使用 GRPO 方法（不依赖 reward model）
使用自定义 reward function 来提供可验证的奖励

注意：虽然trl给出的参考代码是 PAPO_trainer_example.py，但这里使用 GRPO，
因为 PAPO 主要用于多模态任务，而 TL;DR 是文本任务。
如果需要多模态任务，可以使用 PAPOTrainer。
"""
import os
import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig

# 修复 wandb JSON decode 错误
os.environ.setdefault("WANDB_MODE", "offline")

# -------------------------
# 0) paths
# -------------------------
base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# SFT训练后的adapter保存到这个目录,从这里加载LoRA微调的SFT
# 使用绝对路径，确保从任何位置运行都能找到
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)
sft_adapter_dir = os.path.join(workspace_root, "trl_sft", "sft_tldr_lora", "checkpoint-63")

# -------------------------
# 1) tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb = BitsAndBytesConfig(load_in_8bit=True)

# -------------------------
# 2) policy (trainable): Qwen base + your LoRA adapter
# -------------------------
base_policy = AutoModelForCausalLM.from_pretrained(
    base_model_name, device_map="auto", quantization_config=bnb, trust_remote_code=True
)
policy_model = PeftModel.from_pretrained(base_policy, sft_adapter_dir, is_trainable=True)

# -------------------------
# 3) 数据集准备
# -------------------------
raw_dataset = load_dataset("trl-lib/tldr", split="train[:1000]")

def prepare_dataset(dataset, tokenizer):
    """pre-tokenize the dataset before training; only collate during training"""

    def tokenize(element):
        # GRPO 需要 "prompt" 列
        # TL;DR 数据集已经有 "prompt" 和 "label" 列
        return {
            "prompt": element["prompt"],
            "label": element.get("label", ""),  # 保存 ground truth 用于 reward function
        }

    return dataset.map(
        tokenize,
        remove_columns=[col for col in dataset.column_names if col not in ["prompt", "label"]],
    )

# Compute that only on the main process for faster data processing.
with PartialState().local_main_process_first():
    train_dataset = prepare_dataset(raw_dataset, tokenizer)
    # 过滤太长的 prompt（可选）
    # train_dataset = train_dataset.filter(lambda x: len(tokenizer(x["prompt"], padding=False)["input_ids"]) <= 512)

# 创建一个小的 eval_dataset
eval_dataset = train_dataset.select(range(min(10, len(train_dataset))))

# -------------------------
# 4) 自定义 Reward Function (RLVR)
# -------------------------
def tldr_reward_func(completions, label=None, **kwargs):
    """
    自定义 reward function for TL;DR 任务
    
    Args:
        completions: 生成的摘要列表
        label: ground truth 摘要列表（来自数据集）
        **kwargs: 其他参数
    
    Returns:
        rewards: 每个 completion 的奖励值列表
    """
    rewards = []
    
    for i, completion in enumerate(completions):
        # 获取对应的 ground truth
        gt = label[i] if label and i < len(label) else None
        
        if gt:
            # 简单的奖励策略：基于与 ground truth 的相似度
            # 这里使用简单的字符串匹配，你可以使用更复杂的指标（如 ROUGE, BLEU 等）
            completion_text = completion if isinstance(completion, str) else completion.get("content", "")
            gt_text = gt if isinstance(gt, str) else gt.get("content", "")
            
            # 简单的奖励计算：
            # 1. 如果 completion 包含关键信息，给予奖励
            # 2. 长度惩罚（鼓励简洁）
            # 3. 与 ground truth 的相似度
            
            # 基础奖励：如果 completion 不为空
            reward = 0.1 if completion_text.strip() else -0.5
            
            # 长度奖励：鼓励合理的长度（50-200 字符）
            length = len(completion_text)
            if 50 <= length <= 200:
                reward += 0.2
            elif length > 200:
                reward -= 0.1 * (length - 200) / 100  # 过长惩罚
            elif length < 50:
                reward -= 0.1 * (50 - length) / 50  # 过短惩罚
            
            # 相似度奖励（简单版本：基于共同词汇）
            if gt_text:
                completion_words = set(completion_text.lower().split())
                gt_words = set(gt_text.lower().split())
                if len(gt_words) > 0:
                    overlap = len(completion_words & gt_words) / len(gt_words)
                    reward += 0.7 * overlap  # 相似度贡献最大
            
            rewards.append(reward)
        else:
            # 如果没有 ground truth，使用简单的启发式奖励
            completion_text = completion if isinstance(completion, str) else completion.get("content", "")
            length = len(completion_text)
            
            # 基础奖励
            reward = 0.1 if completion_text.strip() else -0.5
            
            # 长度奖励
            if 50 <= length <= 200:
                reward += 0.3
            elif length > 200:
                reward -= 0.1 * (length - 200) / 100
            
            rewards.append(reward)
    
    return rewards


# 可选：使用更复杂的奖励函数（需要安装额外依赖）
# 例如使用 ROUGE 分数：
# try:
#     from rouge_score import rouge_scorer
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     
#     def tldr_reward_func_with_rouge(completions, label=None, **kwargs):
#         rewards = []
#         for i, completion in enumerate(completions):
#             completion_text = completion if isinstance(completion, str) else completion.get("content", "")
#             gt = label[i] if label and i < len(label) else None
#             
#             if gt:
#                 gt_text = gt if isinstance(gt, str) else gt.get("content", "")
#                 scores = scorer.score(gt_text, completion_text)
#                 # 使用 ROUGE-L F1 分数作为主要奖励
#                 reward = scores['rougeL'].fmeasure
#                 rewards.append(reward)
#             else:
#                 rewards.append(0.0)
#         return rewards
# except ImportError:
#     print("⚠️  rouge_score 未安装，使用简单奖励函数")

# -------------------------
# 5) GRPO config
# -------------------------
training_args = GRPOConfig(
    output_dir="./grpo_tldr_rlvr",
    learning_rate=1e-5,
    
    per_device_train_batch_size=4,      # 每个设备的batch size
    gradient_accumulation_steps=4,       # 梯度累积步数
    num_train_epochs=1,                  # 训练轮数（GRPO 使用 epochs 而不是 num_ppo_epochs）
    max_steps=200,                       # 或者使用 max_steps 来控制训练步数
    
    # 生成参数
    max_completion_length=128,            # 最大生成 completion 长度（GRPO 使用 max_completion_length 而不是 max_new_tokens）
    temperature=1.0,                     # 采样温度
    
    # GRPO 特定参数
    num_generations=4,                    # 每个 prompt 生成的样本数（用于 group relative rewards）
    scale_rewards="group",                # 奖励标准化方式："group"（组内标准化）或 "batch"（批次标准化）
    beta=0.0,                             # KL 系数，0.0 表示不使用 reference model（节省内存）
    
    # 其他参数
    report_to="wandb",  # 或 "none" 来禁用 wandb
    eval_strategy="no",                  # 不使用评估
    logging_steps=10,                     # 每 10 步记录一次
)

# -------------------------
# 6) GRPO Trainer (不依赖 reward model)
# -------------------------
trainer = GRPOTrainer(
    model=policy_model,                  # 使用已经加载的模型（带 LoRA）
    reward_funcs=tldr_reward_func,       # 使用自定义 reward function
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=None,  # 已经加载了PeftModel，不需要peft_config
)

# -------------------------
# 7) 开始训练
# -------------------------
print("🚀 开始 GRPO 训练（RLVR 方法，不依赖 reward model）...")
trainer.train()

# 保存模型
trainer.save_model(training_args.output_dir)
print(f"💾 模型已保存到: {training_args.output_dir}")

print("✅ 训练完成！")
