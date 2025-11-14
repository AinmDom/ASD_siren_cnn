# config.py
import torch

# --- 1. 路径设置 (更新) ---
# FSD50K 路径
FSD_METADATA_DIR = "FSD50K.ground_truth"  # 包含 .csv 文件的目录
FSD_DEV_AUDIO_DIR = "FSD50K.dev_audio"    # 包含 .wav 文件的目录
FSD_EVAL_AUDIO_DIR = "FSD50K.eval_audio"  # 评估集的 .wav 目录
MODEL_SAVE_PATH = "models/fsd_siren_model.pth"   # 新的模型保存路径

# --- 2. 音频处理参数 (不变) ---
TARGET_SAMPLE_RATE = 22050
TARGET_NUM_SAMPLES = 88200  # 4 seconds * 22050 SR (保持 4 秒)
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

# --- 3. 训练参数 (不变) ---
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
# (FSD50K 已经划分好了 dev/eval，我们不再需要 'folds')

# --- 4. 评估设置 (更新) ---
# 目标样本对应的“mids”，即可以被视作异常声的标签
SIREN_MIDS = [
    "/m/03kmc9", # Siren
    "/m/07pp_mv", # Alarm
    "/m/07qqyl4", # Boom
    "/m/014zdl",  # Explosion
    "/m/032s66", # Gunshot_and_gunfire

    # "/m/07plct2",  # Crushing
    # "/m/07pjwq1", # Buzz
    # "/m/0395lw",  # Bell
    # "/m/0gy1t2s",  # Bicycle_bell
    # "/m/03wwcy", # Doorbell
    # "/m/03qc9zr"   # Screaming
]
TARGET_NAMES = ["Not Siren", "Siren (Target)"]

# --- 5. 设备设置 (不变) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")