from util import batch_convert_json_to_csv
from audio_segment_DataGen import get_segment_vectors, get_segment_vectors_single
import os, glob, json, random
import pandas as pd
import numpy as np
from audio_segment_TrainTest import load_samples_from_paths, split_data, train, test


if __name__ == "__main__":
    '''
    input_folder = "random_samples"
    f0mode = "normal"  # 可选: "normal", "fft", "crepe", "pyin"
    segment_duration_ms = 20
    shiftmode = "ratio"  # 可选: "key", "ratio"
    shift_point = (1/3, 1/2, 1)
    label_adj = (1/5, 1/4, 1/3, 1/2, 2/3, 1, 3/2, 2, 3, 4, 5)

    # 构建输出特征文件夹名，遵循 get_segment_vectors 参数规则
    feature_folder = f"features_{f0mode}_{shiftmode}_{len(shift_point)}shift_{len(label_adj)}point"
    os.makedirs(feature_folder, exist_ok=True)

    print("🚀 开始遍历 CSV 并提取特征...")
    for filename in os.listdir(input_folder):
        if not filename.endswith(".csv"):
            continue
        csv_path = os.path.join(input_folder, filename)
        print(f"处理文件: {csv_path}")
        # 调用 get_segment_vectors 生成特征文件
        # 这里使用 get_segment_vectors_single 并在其基础上处理向量字段格式化
        segment_features = get_segment_vectors_single(
            csv_path=csv_path,
            audio_path=csv_path.replace('.csv', '.wav'),
            output_csv_path=os.path.join(feature_folder,f"{filename}_features.csv"),
            f0mode=f0mode,
            segment_duration_ms=segment_duration_ms,
            shiftmode=shiftmode,
            shift_point=shift_point,
            label_adj=label_adj
        )
        
    print(f"✅ 所有特征文件已生成，存放在 {feature_folder}")
    '''
    data_folder = "features_normal_ratio_3shift_11point"
    allowed_labels = [2, 4, 5]
    point = 11
    model_path = "f11_model.pt"
    all_csv_paths = glob.glob(os.path.join(data_folder, "*.csv"))
    random.seed(42)
    random.shuffle(all_csv_paths)
    half = len(all_csv_paths) // 2
    selected_train_files = all_csv_paths[:half]
    selected_test_files = all_csv_paths[half:]
    
    print(f"🔍 选中训练文件数: {len(selected_train_files)}")
    print(f"🔍 选中测试文件数: {len(selected_test_files)}")

    print("🚀 加载训练样本...")
    train_samples = load_samples_from_paths(selected_train_files, allowed_labels=allowed_labels, point=point)
    print(f"训练样本数: {len(train_samples)}")

    # 按比例划分训练集、验证集、测试集
    train_samples, val_samples, _ = split_data(train_samples, train_ratio=0.7, val_ratio=0.1)

    print("🚀 加载测试样本...")
    test_samples = load_samples_from_paths(selected_test_files, allowed_labels=allowed_labels, point=point)
    print(f"测试样本数: {len(test_samples)}")
    
    train_labels = [y for _, y in train_samples]
    print(f"训练集中实际使用的标签种类: {sorted(set(train_labels))}")

    train(train_samples=train_samples, val_samples=val_samples, max_samples= 50000, model_path=model_path, nc=11)
    test(test_samples=test_samples, model_path=model_path, nc=11)
