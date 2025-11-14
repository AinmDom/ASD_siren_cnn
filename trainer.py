# trainer.py
import torch
import config
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score
import wandb  # 导入 wandb


def train_model(model, train_loader, criterion, optimizer, epoch):
    """
    执行一个 epoch 的训练。
    """
    model.train()  # 设置为训练模式
    running_loss = 0.0

    # 使用 enumerate 来跟踪批次索引
    for i, (inputs, labels) in enumerate(train_loader):
        # inputs 和 labels 应该已经在 dataset.py 中被 .to(device) 了

        # 1. 梯度清零
        optimizer.zero_grad()

        # 2. 前向传播
        outputs = model(inputs)

        # 3. 计算损失
        loss = criterion(outputs, labels)

        # 4. 反向传播
        loss.backward()

        # 5. 更新权重
        optimizer.step()

        running_loss += loss.item()

        # 打印并记录“步”级别的损失 (每100步一次)
        if (i + 1) % 100 == 0:
            step_loss = loss.item()
            print(f'Epoch [{epoch + 1}/{config.EPOCHS}], Step [{i + 1}/{len(train_loader)}], Loss: {step_loss:.4f}')
            # 【W&B】实时记录训练中的每一步损失
            wandb.log({"train_step_loss": step_loss})

    # 计算并记录“轮”级别的平均损失
    avg_epoch_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch + 1} Training Loss: {avg_epoch_loss:.4f}')

    # 【W&B】记录整个 epoch 的平均训练损失，并使用 'epoch' 作为 x 轴
    wandb.log({"train_epoch_loss": avg_epoch_loss, "epoch": epoch})


def evaluate_model(model, val_loader, criterion, epoch):
    """
    在验证集上评估模型。
    """
    model.eval()  # 设置为评估模式
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 评估时不需要计算梯度
        for inputs, labels in val_loader:
            # inputs 和 labels 应该已经在 dataset.py 中被 .to(device) 了

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 获取预测结果 (获取得分最高的那个类别的索引)
            _, predicted = torch.max(outputs.data, 1)

            # 收集所有预测和标签，用于稍后计算报告
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)

    # --- 【关键修改】 ---
    # 我们现在计算所有需要的指标，而不仅仅是打印报告

    # 目标类别 "Siren" 的 ID 是 1 (0 是 "Not Siren")
    target_label = 1

    # 计算 "Siren (Target)" 类的特定指标
    # zero_division=0: 防止在训练早期模型一个Siren都猜不对时, 发生除零错误
    siren_recall = recall_score(all_labels, all_preds, pos_label=target_label, zero_division=0)
    siren_precision = precision_score(all_labels, all_preds, pos_label=target_label, zero_division=0)
    siren_f1 = f1_score(all_labels, all_preds, pos_label=target_label, zero_division=0)

    # (可选) 你也可以计算宏平均值 (Macro Averages)
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # 1. 在控制台打印完整的报告 (和以前一样)
    print(f'Validation Loss: {avg_loss:.4f}')
    # print(classification_report(all_labels, all_preds, target_names=config.TARGET_NAMES, zero_division=0))

    # 2. 【W&B】将所有关键指标记录到 W&B 仪表板
    #    我们使用 "epoch" 作为 x 轴，这样 W&B 会自动将训练损失和验证指标对齐
    wandb.log({
        "epoch": epoch,
        "val_loss": avg_loss,
        "val_recall (Siren)": siren_recall,
        "val_precision (Siren)": siren_precision,
        "val_f1 (Siren)": siren_f1,
        "val_macro_recall": macro_recall,
        "val_macro_f1": macro_f1
    })