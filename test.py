# verify_bug.py
from operator import index

import pandas as pd
import config  # 1. 导入你真实的、已修复的 config.py
import os



def main():
    metadata_path = os.path.join(config.FSD_METADATA_DIR, "dev.csv")

    metadata = pd.read_csv(metadata_path)
    index = 460
    row = metadata.iloc[index]
    mids_string = row['mids']

    print("一个标签的例子：")
    print(mids_string)

    labels_mids = row['mids'].split(',')
    print("分离标签后：")
    print(labels_mids)
    print(labels_mids[3])

    # (来自 config.py)
    # siren_mids_from_config = config.SIREN_MIDS
    # print(f"\nconfig.py 中的目标 ID: {siren_mids_from_config[0]}")
    # print(f"目标 ID 的类型: {type(siren_mids_from_config[0])}")
    #
    # # 2. 从 FSD50K.ground_truth/dev.csv 中读取一行数据
    # # (来自 dev.csv)
    # row_mids_string = '"/m/03kmc9";"/m/07rrl"'
    # print(f"\nCSV 文件中的 'mids' 字符串: {row_mids_string}")
    # print(f"字符串的类型: {type(row_mids_string)}")
    #
    # # 只用了 .split(';')
    # buggy_labels_list = row_mids_string.split(';')
    #
    # print(f"split(';') 后的列表: {buggy_labels_list}")
    # print(f"列表中的第一项: {buggy_labels_list[0]}")
    # print(f"列表中第一项的类型: {type(buggy_labels_list[0])}")
    #
    # # 进行检查
    # label = 0
    # if siren_mids_from_config[0] in buggy_labels_list:
    #     label = 1
    #
    # print(f"\n检查: '{siren_mids_from_config[0]}' in {buggy_labels_list} ?")
    # print(f"检查结果: {siren_mids_from_config[0] in buggy_labels_list}")
    # print(f"最终标签 (label): {label}")  # <--- 这里会是 0
    #
    # # ---------------------------------------------
    # # 4. 【修复后的正确代码】
    # # (来自我建议的新版 dataset.py)
    # # ---------------------------------------------
    # print("\n--- 正在运行我建议的【修复】代码 ---")
    #
    # # 修复的代码：使用了 [m.strip('\"') for m in ...]
    # fixed_labels_list = [m.strip('"') for m in row_mids_string.split(';')]
    #
    # print(f"split(';') 并 strip('\"') 后的列表: {fixed_labels_list}")
    # print(f"列表中的第一项: {fixed_labels_list[0]}")
    # print(f"列表中第一项的类型: {type(fixed_labels_list[0])}")
    #
    # # 进行检查
    # label = 0
    # if siren_mids_from_config[0] in fixed_labels_list:
    #     label = 1
    #
    # print(f"\n检查: '{siren_mids_from_config[0]}' in {fixed_labels_list} ?")
    # print(f"检查结果: {siren_mids_from_config[0] in fixed_labels_list}")
    # print(f"最终标签 (label): {label}")  # <--- 这里会是 1


if __name__ == "__main__":
    main()