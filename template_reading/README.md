# 这里存放了一些写代码时需要阅读的template

## 阅读reward model, value model
要看 reward/value 模型的源代码，有两种来源：Hugging Face Hub 上的模型仓库（若 trust_remote_code 为 True 会用仓库自带的 Python 模块），或 Transformers 内置模型类（默认从 config.architectures 推断，如 GPTNeoXForSequenceClassification）。

步骤建议：
- 克隆模型仓库，查看是否包含自定义代码
````bash
git lfs install
git clone https://huggingface.co/cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr /workspace/models/reward
ls -la /workspace/models/reward
# 关注：modeling_*.py、modeling.py、custom code、README、config.json
````

- 确认实际加载到的类（以及是否使用自定义代码）
````python
from transformers import AutoConfig, AutoModelForSequenceClassification
name = "cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr"
cfg = AutoConfig.from_pretrained(name)
print("architectures:", cfg.architectures, "model_type:", cfg.model_type)

m = AutoModelForSequenceClassification.from_pretrained(name, trust_remote_code=False)
print("loaded class:", type(m), "module:", m.__module__)
````

- 若没有自定义代码，则到 Transformers 查看对应内置类源码（例如 GPTNeoXForSequenceClassification）
````bash
python -c "import importlib; m=importlib.import_module('transformers.models.gpt_neox.modeling_gpt_neox'); print(m.__file__)"
# 打开上述文件路径，在 VS Code 里阅读 GPTNeoXForSequenceClassification 的实现
````

- 在线源码参考（Transformers）
  - GPTNeoXForSequenceClassification: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py

说明：
- 本脚本中 value_model 与 reward_model 都从同一个 checkpoint 加载：
  - AutoModelForSequenceClassification(..., num_labels=1)
- 是否使用仓库自定义代码取决于 trust_remote_code（由 model_args.trust_remote_code 决定）。如仓库内含 modeling*.py 且你设为 True，则优先使用仓库实现；否则使用 Transformers 内置实现。