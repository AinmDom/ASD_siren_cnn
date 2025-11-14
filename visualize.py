import torch
from torchviz import make_dot
from model import SimpleSirenCNN  # 导入你的模型（替换为实际模型路径）

# 1. 初始化模型（保持和你的代码一致）
model = SimpleSirenCNN(num_classes=2)
model.eval()  # 避免 BatchNorm/Dropout 影响计算图

# 2. 创建 dummy 输入（务必匹配模型实际输入尺寸！）
# 示例：如果你的模型输入是 (batch_size, 3, 224, 224)（RGB图像），则改为：
# dummy_input = torch.randn(1, 3, 224, 224)
dummy_input = torch.randn(1, 1, 28, 28)  # 按你的实际输入调整

# 3. 生成计算图（指定格式为 PNG，强制使用 dot 引擎）
with torch.no_grad():  # 避免梯度计算，简化图结构
    output = model(dummy_input)

# 4. 生成并保存图片（指定引擎为 dot，确保兼容性）
dot = make_dot(
    output,
    params=dict(model.named_parameters()),
    show_attrs=False,  # 不显示属性（简化图）
    show_saved=False   # 不显示保存的张量（简化图）
)
dot.engine = 'dot'  # 强制使用 dot 引擎（Graphviz 核心引擎）
dot.format = 'png'  # 可选：png/jpg/svg/pdf（优先选 png）
dot.save('model_architecture_fixed.png')  # 保存新文件

print("图片已保存为 model_architecture_fixed.png")