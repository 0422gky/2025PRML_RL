# 2025 PRML pj -- RL
TODO:
1. 主流奖励
RLHF
RLVR
DPO
2. baseline
TRL SFT
3. 奖励机制分析
4. 改进奖励实现，对比性能
	正则化，塑性，多奖励权重平衡，更换设计

5. 论文
找related work


Timeline:
12/28 23:20 暂时完成SFT Baseline lora微调，还没有加入RLHF

TODO: 阅读PPO论文
PPO: proximal policy optimization

12/29 TODO: 阅读reward model, value model 架构
(in example_ppo-tldr.py) # --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \