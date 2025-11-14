# dataset.py
# (已修复：将 .split(';') 修正为 .split(','))

import torch
import torch.nn as nn
import torchaudio
import pandas as pd
import os
import config  # 导入我们的配置文件
import numpy as np


class FSD50KDataset(torch.utils.data.Dataset):
    """
    用于 FSD50K 数据集的 Dataset 类。
    它能处理多标签 CSV，并将其转换为二元分类 (Siren vs Not Siren)。
    """

    def __init__(self, metadata_file, audio_dir, is_train=False):

        self.audio_dir = audio_dir
        self.is_train = is_train
        self.target_sample_rate = config.TARGET_SAMPLE_RATE
        self.target_num_samples = config.TARGET_NUM_SAMPLES
        self.siren_mids = set(config.SIREN_MIDS)  # <-- 转换为 set (集合)，查找速度更快

        # 1. 加载 FSD50K ground truth .csv 文件
        try:
            self.metadata = pd.read_csv(metadata_file)
        except FileNotFoundError:
            print(f"错误：未找到元数据文件 {metadata_file}")
            print("!! 请检查 config.py 中的 FSD_METADATA_DIR 路径是否正确 !!")
            self.metadata = pd.DataFrame(columns=['fname', 'mids'])  # 返回空
            return

        # 2. 定义基础变换 (所有数据都要用)
        self.base_transformation = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self.target_sample_rate,
                n_fft=config.N_FFT,
                n_mels=config.N_MELS,
                hop_length=config.HOP_LENGTH
            ),
            torchaudio.transforms.AmplitudeToDB()
        ).to(config.device)

        # 3. 定义数据增强 (方案一：SpecAugment)
        if self.is_train:
            self.augmentation = nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=30),  # 遮挡 30 个 Mel 频带
                torchaudio.transforms.TimeMasking(time_mask_param=30)  # 遮挡 30 帧时间
            ).to(config.device)
        else:
            self.augmentation = nn.Identity().to(config.device)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        # 1. 获取元数据
        row = self.metadata.iloc[index]
        audio_path = os.path.join(self.audio_dir, f"{row['fname']}.wav")

        # 2. **关键：处理多标签**
        #    根据你的测试，mids 列是一个用【逗号】分隔的字符串
        mids_string = row['mids']  # e.g., '/m/02sgy,/m/0342h'

        # --- ↓↓↓ 这是修复 BUG 的代码 ↓↓↓ ---
        #    我们使用 .split(',') 来正确分割标签
        labels_mids = mids_string.split(',')
        # --- ↑↑↑ 修复结束 ↑↑↑ ---

        # 3. **关键：转换为二元标签**
        label = 0
        for tag in labels_mids:
            if tag in self.siren_mids:  # (在 set 中查找非常快)
                label = 1
                break  # 找到了，就是正样本

        # 4. 加载音频
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            # (如果你的 FSD_DEV_AUDIO_DIR 路径是错的，这里会报错)
            print(f"Error loading audio file {audio_path}: {e}")
            waveform = torch.zeros(1, self.target_num_samples)
            sample_rate = self.target_sample_rate
            label = 0  # 标记为负样本以防万一

        waveform = waveform.to(config.device)

        # 5. 统一采样率 (如果需要)
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate).to(
                config.device)
            waveform = resampler(waveform)

        # 6. 转为单通道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 7. **【重要】使用我们讨论过的“随机/中心裁剪”来处理不同时长**
        waveform = self._process_audio_length(waveform)

        # 8. 应用变换
        spectrogram = self.base_transformation(waveform)
        spectrogram = self.augmentation(spectrogram)  # 增强 (如果是验证集则什么都不做)

        return spectrogram, label

    def _process_audio_length(self, waveform):
        """
        一个“智能”的函数，替换掉 _pad_or_truncate。
        - 如果音频太短，就填充 (Pad)
        - 如果音频太长，就在训练时“随机裁剪”，在验证时“中心裁剪”
        """
        num_samples = waveform.shape[1]
        target_samples = self.target_num_samples  # 来自 config.py (例如 88200)

        if num_samples == target_samples:
            return waveform  # 1. 长度刚好，什么都不做

        if num_samples < target_samples:
            # 2. 音频太短 (e.g., 1s)，在末尾补 0
            num_padding = target_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, num_padding))
            return waveform

        if num_samples > target_samples:
            # 3. 音频太长 (e.g., 15s)

            if self.is_train:
                # **训练时：随机裁剪**
                max_start = num_samples - target_samples
                start_index = torch.randint(0, max_start + 1, (1,)).item()
                waveform = waveform[:, start_index: start_index + target_samples]
            else:
                # **验证时：中心裁剪**
                start_index = (num_samples - target_samples) // 2
                waveform = waveform[:, start_index: start_index + target_samples]

            return waveform