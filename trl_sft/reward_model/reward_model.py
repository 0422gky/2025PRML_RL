import os
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset

# 修复 wandb JSON decode 错误 - 使用离线模式
os.environ.setdefault("WANDB_MODE", "offline")

# 设置输出目录
output_dir = "./Qwen2.5-0.5B-Instruct-Reward"

# 加载数据集
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# 配置训练参数
training_args = RewardConfig(
    output_dir=output_dir,
    # 可以根据需要调整这些参数
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=1,  # 根据需要调整
    save_strategy="epoch",
    report_to="wandb",
)

# 创建训练器
# 确保使用 num_labels=1 来创建 reward model
# 通过 model_init_kwargs 传递 num_labels
training_args.model_init_kwargs = {"num_labels": 1}

trainer = RewardTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    args=training_args,
    train_dataset=dataset,
)

# 开始训练
trainer.train()

# 保存模型
trainer.save_model(output_dir)
print(f"✅ Reward model 已保存到: {output_dir}")