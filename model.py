# model.py
import torch
import torch.nn as nn
import sys
import os

# --- 1. 导入本地的 PANNs 库 (修正版) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
panns_repo_pytorch_path = os.path.join(current_dir, 'audioset_tagging_cnn', 'pytorch')

if panns_repo_pytorch_path not in sys.path:
    sys.path.append(panns_repo_pytorch_path)

try:
    # 直接从 models 模块导入 Cnn14 类
    from models import Cnn14
except ImportError as e:
    print(f"!! 导入错误: {e}")
    print(f"尝试加载的路径是: {panns_repo_pytorch_path}")
    print("请检查：")
    print("1. 'audioset_tagging_cnn' 文件夹是否在项目根目录下。")
    print("2. 'models.py' 文件是否存在于 'audioset_tagging_cnn/pytorch/' 目录中。")
    raise


# --- 你的 PANNs 迁移学习模型 (最终修正版) ---
class PannsCNN14(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # 2. 初始化 PANNs Cnn14
        # Cnn14 类的构造函数需要这些参数
        self.base_model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000,
                                classes_num=527)

        # 3. 加载本地权重文件
        # 权重文件现在应该和 model.py 在同一目录 (项目根目录)
        weights_path = os.path.join(current_dir, 'Cnn14_mAP=0.431.pth')

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"找不到权重文件: {weights_path}。请确保它在项目根目录下。")

        print(f"正在加载本地 PANNs 权重: {weights_path} ...")

        # 加载模型状态字典
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)

        # 加载权重到模型实例中。注意 PANNs 的权重文件格式通常是 {'model': state_dict, ...}
        self.base_model.load_state_dict(checkpoint['model'], strict=True)  # strict=True 确保完全匹配

        print("PANNs 权重加载成功。")

        # 4. 冻结主干 (Backbone)
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 5. 【魔改】适配频谱图输入
        # 你的 dataset 输出已经是频谱图了，但 PANNs 原本想要波形。
        # 我们把 PANNs 自带的“波形转频谱”层替换成“什么都不做” (Identity)，
        # 这样你的频谱图就能直接流进 CNN 层了。
        self.base_model.spectrogram_extractor = nn.Identity()
        self.base_model.logmel_extractor = nn.Identity()

        # 6. 替换分类头 (Head)
        # Cnn14 的特征维度是 2048，我们换成 2 分类
        self.base_model.fc_audioset = nn.Linear(2048, num_classes)

        # 确保新加的层是可训练的
        for param in self.base_model.fc_audioset.parameters():
            param.requires_grad = True

    def forward(self, x):
        # 输入 x 形状: [Batch, 1, N_Mels(64), Time(173)]

        # PANNs 的内部结构期望输入是: [Batch, 1, Time, N_Mels]
        # 所以我们需要交换维度 2 和 3
        x = x.transpose(2, 3)
        # 现在 x 形状: [Batch, 1, Time(173), N_Mels(64)]

        # 传入 PANNs
        output_dict = self.base_model(x)

        # 返回分类结果
        return output_dict['clipwise_output']
# --- 你老的 SimpleSirenCNN (保留着) ---
class SimpleSirenCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleSirenCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(64)
        self.dropout = nn.Dropout(p=0.5)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x