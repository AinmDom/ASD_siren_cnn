# run_training.py
# (已修复：添加了“智能”权重计算，不再卡住)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import config  # 1. 导入配置
import wandb
import os
import pandas as pd  # <-- 导入 pandas

# 2. 从我们的模块中导入【新】的 Dataset 类
from dataset import FSD50KDataset
from model import SimpleSirenCNN  # 你老的CNN
# from model import PannsCNN14  # (你可以稍后切换到这个)

from trainer import train_model, evaluate_model
from model import PannsCNN14

def main():
    wandb.init(
        project="siren-detector-FSD50K",  # 建议用个新名字
        config={
            "learning_rate": config.LEARNING_RATE,
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "n_mels": config.N_MELS
        }
    )

    # 自动创建模型保存目录
    model_save_dir = os.path.dirname(config.MODEL_SAVE_PATH)
    if not os.path.exists(model_save_dir) and model_save_dir:
        os.makedirs(model_save_dir)
        print(f"成功创建模型保存目录: {model_save_dir}")

    print(f"Using device: {config.device}")

    # --- ↓↓↓ “智能”权重计算 (修复了卡住的 Bug) ↓↓↓ ---

    # 1. 在创建 Dataset 之前，先加载 CSV 计算权重
    print("正在加载元数据 (dev.csv) 以计算类别权重...")
    train_metadata_file = os.path.join(config.FSD_METADATA_DIR, "dev.csv")

    try:
        df = pd.read_csv(train_metadata_file)
        mids_list = df['mids']
    except FileNotFoundError:
        print(f"!! 致命错误: 找不到 {train_metadata_file}")
        print("!! 请检查 config.py 中的 FSD_METADATA_DIR 路径是否正确 !!")
        return  # 停止执行

    siren_count = 0
    not_siren_count = 0
    siren_mids_set = set(config.SIREN_MIDS)  # 转换为 set 集合，查找更快

    # 这个循环只遍历文本，速度极快 (几秒钟)
    for mids_string in mids_list:
        # 使用和 dataset.py 完全相同的标签解析逻辑
        labels_mids = mids_string.split(',')

        found_siren = False
        for tag in labels_mids:
            if tag in siren_mids_set:  # 检查 tag 是否在 set 中
                found_siren = True
                break

        if found_siren:
            siren_count += 1
        else:
            not_siren_count += 1

    print(f"训练集中 Siren 样本数量 (label=1): {siren_count}")
    print(f"训练集中 Not Siren 样本数量 (label=0): {not_siren_count}")

    if siren_count == 0:
        print("!! 警告: 训练集中没有Siren样本！请检查 config.py 中的 SIREN_MIDS 是否正确 !!")
        weight_for_siren = 1.0
    else:
        # 权重 = 多数类 / 少数类
        weight_for_siren = not_siren_count / siren_count - 5

    print(f"为Siren类(label=1)设置的权重: {weight_for_siren:.2f}")
    weights = torch.tensor([1.0, weight_for_siren]).to(config.device)

    # --- ↑↑↑ 权重计算结束 ↑↑↑ ---

    # 2. 现在才创建 Dataset (这一步会很快)
    print("Loading training data (FSD50K Dev)...")
    train_dataset = FSD50KDataset(train_metadata_file,
                                  config.FSD_DEV_AUDIO_DIR,
                                  is_train=True)  # 开启 SpecAugment

    print("Loading validation data (FSD50K Eval)...")
    val_metadata_file = os.path.join(config.FSD_METADATA_DIR, "eval.csv")
    val_dataset = FSD50KDataset(val_metadata_file,
                                config.FSD_EVAL_AUDIO_DIR,
                                is_train=False)  # 关闭 SpecAugment

    # 3. 准备 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"Training data loaded: {len(train_dataset)} samples")
    print(f"Validation data loaded: {len(val_dataset)} samples")

    # 4. 初始化模型
    # model = SimpleSirenCNN(num_classes=2).to(config.device)
    model = PannsCNN14(num_classes=2).to(config.device)
    # 5. 初始化损失函数 (使用我们刚刚计算的权重)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 6. 开始训练
    print("Starting training on FSD50K...")
    for epoch in range(config.EPOCHS):
        train_model(model, train_loader, criterion, optimizer, epoch)
        evaluate_model(model, val_loader, criterion, epoch)

    print("Training finished.")
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

    wandb.finish()


if __name__ == "__main__":
    main()