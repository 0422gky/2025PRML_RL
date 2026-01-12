"""
带 Reward Shaping 的 PPO 训练脚本

基于 baseline.py，使用 shaped_rewards 中的 PPOTrainer 实现长度惩罚和重复惩罚
"""
import os
import sys

# 确保优先使用本地的 trl 包（如果存在）
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
trl_path = os.path.join(workspace_root, "trl")
if os.path.exists(trl_path):
    sys.path.insert(0, trl_path)

# 添加 shaped_rewards 目录到路径，以便导入修改后的 trainer
sys.path.insert(0, os.path.dirname(__file__))

# 导入修改后的 trainer（带 reward shaping）
from ppo_trainer_shaped import PPOTrainer
from trl.experimental.ppo import PPOConfig  # 使用原始的 PPOConfig

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    AutoConfig,
)
from peft import PeftModel

# 修复 wandb JSON decode 错误
# 方案 1: 使用离线模式（避免网络问题）
os.environ.setdefault("WANDB_MODE", "offline")

# 方案 2: 如果需要在线模式，取消下面的注释并设置 API key
# os.environ["WANDB_API_KEY"] = "your-api-key"
# os.environ["WANDB_PROJECT"] = "your-project-name"

# 方案 3: 如果仍然有问题，可以在训练配置中设置 report_to="none" 来禁用 wandb

# 根据trl_sft加入了RLHF之后的模型
# -------------------------
# 0) paths
# -------------------------
base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# SFT训练后的adapter保存到这个目录,从这里加载LoRA微调的SFT
# 注意：路径相对于运行脚本的位置，从 shaped_rewards 目录运行时使用 ../trl_sft
# 如果从 workspace 根目录运行，使用 ./trl_sft
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)  # 获取 workspace 根目录

sft_adapter_dir = os.path.join(workspace_root, "trl_sft", "sft_tldr_lora", "checkpoint-63")

# Reward model 路径
# 选项 1: 使用自己训练的 reward model（推荐，与 Qwen tokenizer 兼容）
reward_model_path = os.path.join(workspace_root, "trl_sft", "reward_model", "Qwen2.5-0.5B-Instruct-Reward")

# 选项 2: 使用公开的 reward model（注意：tokenizer 可能不兼容）
# reward_model_path = "cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr"

# 如果训练好的 reward model 不存在，可以使用公开模型作为备选
if not os.path.exists(reward_model_path) or not os.listdir(reward_model_path):
    print(f"⚠️  警告: {reward_model_path} 不存在或为空，将使用公开模型")
    reward_model_path = "cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr"

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
# 3) ref (frozen): same init as policy, but frozen
# -------------------------
base_ref = AutoModelForCausalLM.from_pretrained(
    base_model_name, device_map="auto", quantization_config=bnb, trust_remote_code=True
)
ref_model = PeftModel.from_pretrained(base_ref, sft_adapter_dir, is_trainable=False)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

# -------------------------
# 4) reward model and value model (frozen)
# -------------------------
# 使用自己训练的 reward model（基于 Qwen，tokenizer 兼容）
print(f"📦 加载 reward model from: {reward_model_path}")

# 先加载 config 来检查 num_labels
try:
    reward_config = AutoConfig.from_pretrained(reward_model_path, trust_remote_code=True)
    # 如果 config 中有 num_labels，使用它；否则默认使用 1
    num_labels = getattr(reward_config, 'num_labels', 1)
    print(f"   检测到 num_labels: {num_labels}")
except Exception as e:
    print(f"   ⚠️  无法读取 config，使用默认 num_labels=1: {e}")
    num_labels = 1

# 加载 reward model
# 注意：如果 checkpoint 中的 num_labels 与指定值不匹配，会报错
# 解决方案：不指定 num_labels，让模型从 checkpoint 中自动读取
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_path, 
    # 不指定 num_labels，让模型从 checkpoint 的 config 中读取
    device_map="auto", 
    trust_remote_code=True,
)
reward_model.eval()
for p in reward_model.parameters():
    p.requires_grad = False

# value_model 通常和 reward_model 相同
value_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_path, 
    # 不指定 num_labels，让模型从 checkpoint 中自动读取
    device_map="auto", 
    trust_remote_code=True,
)
value_model.eval()
for p in value_model.parameters():
    p.requires_grad = False

# 如果 num_labels=2，需要创建包装器来适配 PPO trainer 的期望
# PPO trainer 期望 reward/value model 输出形状为 [batch, seq] 或 [batch, seq, 1]
if num_labels == 2:
    print("   ⚠️  检测到 num_labels=2，创建包装器以适配 PPO trainer")
    
    class RewardModelWrapper(torch.nn.Module):
        """包装器：将 num_labels=2 的输出转换为 num_labels=1 的输出"""
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.base_model_prefix = model.base_model_prefix
            
        def score(self, hidden_states):
            """只取第一个维度的输出（或者可以取平均，根据训练方式）"""
            scores = self.model.score(hidden_states)  # [batch, seq, 2]
            # 取第一个维度，或者根据训练方式调整
            # 如果是 preference learning，可能需要取 chosen - rejected
            # 这里假设第一个维度是 reward score
            return scores[..., 0:1]  # 保持维度 [batch, seq, 1]，这样 squeeze(-1) 后是 [batch, seq]
        
        def __getattr__(self, name):
            """转发其他属性到原始模型"""
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.model, name)
    
    # 包装 reward model 和 value model
    reward_model = RewardModelWrapper(reward_model)
    value_model = RewardModelWrapper(value_model)
    print("   ✅ Reward model 和 Value model 包装完成")

print("✅ Reward model 和 value model 加载完成")

# -------------------------
# 5) TL;DR dataset -> PPO queries
# -------------------------
raw_dataset = load_dataset("trl-lib/tldr", split="train[:1000]")

def prepare_dataset(dataset, tokenizer):
    """pre-tokenize the dataset before training; only collate during training"""

    def tokenize(element):
        input_ids = tokenizer(element["prompt"], padding=False)["input_ids"]
        return {"input_ids": input_ids, "lengths": len(input_ids)}

    return dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
    )

# Compute that only on the main process for faster data processing.
with PartialState().local_main_process_first():
    train_dataset = prepare_dataset(raw_dataset, tokenizer)
    # filtering
    train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 512)

assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"

# 创建一个小的 eval_dataset（用于 generate_completions，即使 eval_strategy="no"）
# PPO trainer 需要 eval_dataset 来生成示例，即使不使用评估
eval_dataset = train_dataset.select(range(min(10, len(train_dataset))))

# -------------------------
# 6) PPO config with Reward Shaping
# -------------------------
training_args = PPOConfig(
    output_dir="./ppo_tldr_rlhf_shaped",
    learning_rate=1e-5,
    
    per_device_train_batch_size=4,      # 每个设备的batch size
    gradient_accumulation_steps=4,       # 梯度累积步数
    local_rollout_forward_batch_size=16, # rollout阶段的forward batch size
    num_ppo_epochs=4,                    # PPO epochs
    
    # 生成参数
    response_length=128,                # 响应长度
    temperature=1.0,                     # 采样温度
    stop_token="eos",                    # 停止token
    
    kl_coef=0.1,                        # KL散度系数
    
    # 其他参数
    # 如果 wandb 仍有问题，可以改为 "none" 或 [] 来禁用
    report_to="wandb",  # 或 "none" 来禁用 wandb
    eval_strategy="no",                  # 不使用评估
    num_sample_generations=0,            # 禁用 generate_completions（需要 eval_dataset）
    
    # Reward Shaping 参数（可选，有默认值）
    # 注意：这些参数不在 PPOConfig 的正式定义中，但可以通过动态属性设置
    # 如果不在 PPOConfig 中设置，代码会使用默认值
    # length_penalty_coef=0.01,        # 长度惩罚系数，默认 0.01
    # repetition_penalty_coef=0.1,     # 重复惩罚系数，默认 0.1
    # repetition_n_gram=3,             # n-gram 大小，默认 3
)

# 如果需要在 PPOConfig 中设置 reward shaping 参数，可以使用 setattr
# 或者直接在创建时传递（虽然 PPOConfig 不正式支持，但可以通过 **kwargs 传递）
# 这里我们使用默认值，如果需要自定义，可以取消下面的注释：
# training_args.length_penalty_coef = 0.01      # 长度惩罚系数
# training_args.repetition_penalty_coef = 0.1   # 重复惩罚系数
# training_args.repetition_n_gram = 3           # n-gram 大小

# -------------------------
# 7) PPO Trainer (带 Reward Shaping)
# -------------------------
# 使用修改后的 PPOTrainer，自动应用长度惩罚和重复惩罚
trainer = PPOTrainer(
    args=training_args,
    processing_class=tokenizer,
    model=policy_model,
    ref_model=ref_model,
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # 提供 eval_dataset 用于 generate_completions
    peft_config=None,  # 已经加载了PeftModel，不需要peft_config
)

# -------------------------
# 8) 开始训练
# -------------------------
print("🚀 开始 PPO 训练（带 Reward Shaping）...")
trainer.train()

# 保存模型
trainer.save_model(training_args.output_dir)
print(f"💾 模型已保存到: {training_args.output_dir}")

print("✅ 训练完成！")
