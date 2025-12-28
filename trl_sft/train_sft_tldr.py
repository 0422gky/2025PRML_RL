from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# base model (8bit)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)

# LoRA config
peft_config = LoraConfig(
    r=16,                    # 4090 可以更大一点
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# dataset
dataset = load_dataset("trl-lib/tldr", split="train[:1000]")

# training args
training_args = SFTConfig(
    output_dir="./sft_tldr_lora",
    per_device_train_batch_size=2,   # 4090 可以 2
    gradient_accumulation_steps=8,   # effective batch = 16, 8次 forward/backward optimizer.step() 一次
    num_train_epochs=1, # 整个dataset只遍历一遍
    logging_steps=10, #  Number of update steps between two logs, trainer_state.json当中的global step代表总共更新的次数
    save_steps=500, # Number of updates steps before two checkpoint saves if `save_strategy="steps"`. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps.
    report_to="wandb",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()

print("show data example: ")
print(dataset[0]) # reddit tl;dr