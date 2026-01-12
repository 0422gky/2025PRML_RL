# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

import os
import shutil

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import ModelConfig, ScriptArguments, get_kbit_device_map, get_peft_config, get_quantization_config
from trl.experimental.ppo import PPOConfig, PPOTrainer


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


"""
python examples/scripts/ppo/ppo_tldr.py \
    --dataset_name trl-lib/tldr \
    --dataset_test_split validation \
    --learning_rate 3e-6 \
    --output_dir pythia-1b-deduped-tldr-preference-sft-trl-style-ppo \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --total_episodes 30000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --response_length 53 \
    --eval_strategy steps \
    --eval_steps 100 

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/ppo/ppo_tldr.py \
    --dataset_name trl-lib/tldr \
    --dataset_test_split validation \
    --output_dir pythia-1b-deduped-tldr-preference-sft-trl-style-ppo \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --total_episodes 1000000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --local_rollout_forward_batch_size 16 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --eval_strategy steps \
    --eval_steps 100
"""

"""
accelerate config 备用参数,不使用deepspeed.yaml,仅为了调试断点阅读源码
上面的加入report to wandb为了不让trackio在hugging face部署过多的东西浪费时间

# 1) 只关 Trackio 不关 wandb
unset TRACKIO_SPACE_ID
unset TRACKIO_DATASET_ID
env | grep -i trackio  # 确认已没有输出

# 2) wandb 正常用
export WANDB_PROJECT="trl-ppo"
# export WANDB_MODE=offline  # 没网就开

# 3) 运行, 123的命令最好每次训练前先看一眼,不然很容易爆gpu
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

调试命令：修改了 response length 53->32 和 total episode 30000 -> 1
--report_to wandb 等调试跑通了再开

remark: wandb 有时候也会卡住
unset TRACKIO_SPACE_ID TRACKIO_DATASET_ID TRACKIO_PROJECT TRACKIO_SPACE
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

python examples/scripts/ppo/ppo_tldr.py \
    --dataset_name trl-lib/tldr \
    --dataset_test_split validation \
    --learning_rate 3e-6 \
    --output_dir pythia-1b-deduped-tldr-preference-sft-trl-style-ppo \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --total_episodes 1 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --response_length 16 \
    --eval_strategy steps \
    --eval_steps 100

accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
  /workspace/trl/examples/scripts/ppo/ppo_tldr.py \
    --dataset_name trl-lib/tldr \
    --dataset_test_split validation \
    --learning_rate 3e-6 \
    --output_dir pythia-1b-deduped-tldr-preference-sft-trl-style-ppo \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --local_rollout_forward_batch_size 1 \
    --gradient_checkpointing True \
    --total_episodes 30000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --response_length 16 \
    --eval_strategy steps \
    --eval_steps 100 \
    
"""

"""
about the args(from copilot)

下面这批命令行参数在本脚本里会被 HfArgumentParser 解析到三个 dataclass：ScriptArguments、PPOConfig（training_args）和 ModelConfig（model_args）。对应关系如下。

ScriptArguments（script_args）

--dataset_name
--dataset_test_split
也包含未在示例里显式给出的：--dataset_train_split、--dataset_config

PPOConfig（training_args）

--learning_rate
--output_dir
--per_device_train_batch_size
--gradient_accumulation_steps
--total_episodes
--sft_model_path
--reward_model_path
--missing_eos_penalty
--stop_token
--response_length
--eval_strategy
--eval_steps
--local_rollout_forward_batch_size
代码里还用到：dataset_num_proc、push_to_hub（同属 PPOConfig）

ModelConfig（model_args）

--model_name_or_path
同类还包括（未在示例中列出）：--dtype、--attn_implementation、--model_revision、--trust_remote_code，以及量化/PEFT 相关选项
提示：你可以用 --help 查看各组参数的归属与默认值。
python /workspace/template_reading/example_ppo_tldr.py --help
通过 accelerate 运行时查看脚本自身的帮助（注意要用 -- 把参数传给脚本）
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
  /workspace/template_reading/example_ppo_tldr.py -- --help

"""


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig)) 
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    value_model = AutoModelForSequenceClassification.from_pretrained(
        # todo: 阅读reward_model, value_model架构
        training_args.reward_model_path, # --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
        **model_kwargs,
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, # --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
        **model_kwargs,
    )
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
        )
    else:
        ref_policy = None

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            input_ids = tokenizer(element["prompt"], padding=False)["input_ids"]
            return {"input_ids": input_ids, "lengths": len(input_ids)}

        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        if eval_dataset is not None:
            eval_dataset = prepare_dataset(eval_dataset, tokenizer)
        # filtering
        train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)

    assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"

    ################
    # Training
    ################
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()