"""
修复 wandb JSON decode 错误的解决方案
"""
import os
import json

# 方案 1: 设置 wandb 为离线模式（如果不需要在线同步）
os.environ["WANDB_MODE"] = "offline"

# 方案 2: 如果使用在线模式，确保设置正确的环境变量
# os.environ["WANDB_API_KEY"] = "your-api-key"  # 如果需要
# os.environ["WANDB_PROJECT"] = "your-project-name"  # 可选
# os.environ["WANDB_ENTITY"] = "your-entity"  # 可选

# 方案 3: 禁用 wandb（如果不需要记录）
# 在训练配置中设置 report_to="none" 或 report_to=[]

# 方案 4: 清理损坏的 wandb 缓存
def clean_wandb_cache():
    """清理可能损坏的 wandb 缓存文件"""
    import shutil
    wandb_dirs = [
        "./wandb",
        "./reward_model/wandb",
        "~/.cache/wandb",
    ]
    
    for dir_path in wandb_dirs:
        expanded_path = os.path.expanduser(dir_path)
        if os.path.exists(expanded_path):
            print(f"找到 wandb 目录: {expanded_path}")
            # 可以选择删除或重命名
            # shutil.rmtree(expanded_path)
            # print(f"已删除: {expanded_path}")

if __name__ == "__main__":
    print("Wandb 修复方案:")
    print("1. 离线模式: WANDB_MODE=offline")
    print("2. 禁用 wandb: report_to='none'")
    print("3. 清理缓存: 删除 wandb 目录")
    clean_wandb_cache()
