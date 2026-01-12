---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
model_name: ppo_tldr_rlhf_baseline
tags:
- base_model:adapter:Qwen/Qwen2.5-0.5B-Instruct
- lora
- transformers
licence: license
pipeline_tag: text-generation
---

# Model Card for ppo_tldr_rlhf_baseline

This model is a fine-tuned version of [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with PPO, a method introduced in [Fine-Tuning Language Models from Human Preferences](https://huggingface.co/papers/1909.08593).

### Framework versions

- PEFT 0.18.0
- TRL: 0.26.2
- Transformers: 4.57.3
- Pytorch: 2.9.1+cu126
- Datasets: 4.4.2
- Tokenizers: 0.22.1

## Citations

Cite PPO as:

```bibtex
@article{mziegler2019fine-tuning,
    title        = {{Fine-Tuning Language Models from Human Preferences}},
    author       = {Daniel M. Ziegler and Nisan Stiennon and Jeffrey Wu and Tom B. Brown and Alec Radford and Dario Amodei and Paul F. Christiano and Geoffrey Irving},
    year         = 2019,
    eprint       = {arXiv:1909.08593}
}
```

Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```