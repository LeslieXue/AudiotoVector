import builtins, inspect, os
_orig_print = builtins.print
def traced_print(*args, **kwargs):
    frm = inspect.stack()[1]
    _orig_print(f"[{os.path.basename(frm.filename)}:{frm.lineno}]", *args, **kwargs)
builtins.print = traced_print
from pydub import AudioSegment as PydubAudioSegment
import pandas as pd
import sys
import os
import glob
import json
import re
from util import point_to_index

def safe_json_list(val):
    """
    安全地将任意类型的数据（ndarray、字符串、列表、标量）转换为 JSON 格式的 list 字符串。
    """
    if isinstance(val, str):
        cleaned = re.sub(r'\s+', ',', val.strip("[] "))
        return json.dumps([float(x) for x in cleaned.split(',') if x])
    elif isinstance(val, (list, tuple, np.ndarray)):
        return json.dumps(np.asarray(val).tolist())
    else:
        return json.dumps([val])

sys.path.append('util')  
import matplotlib.pyplot as plt
import numpy as np
from util import(
    convert_value,
    cut_audio,
    process_pyinf0_and_label,
    shift_pyinf0_and_label
)
from audio_segment import AudioSegment
from audio_segment_features import AudioSegmentFeatures


def extract_features_predict(start_time, samples, sample_rate, base_f0):
    ASF = AudioSegmentFeatures(samples=samples, sample_rate=sample_rate, base_f0=base_f0)
    ASF.set_target_frequencies(mode='ratio')
    frequencies, magnitudes = ASF.perform_fft(normalize_by_frequency=True)
    frequencies_raw, magnitudes_raw = ASF.perform_fft(normalize_by_frequency=False)
    complex_frequencies, complex_result = ASF.perform_fft_complex()
    ASF.set_energy_vector_sum(frequencies=frequencies, magnitude=magnitudes)
    ASF.set_energy_vector_squared(frequencies=frequencies_raw, magnitude=magnitudes_raw)
    ASF.set_normalized_energy_vector(ASF.features['energy_vector_sum'], name='energy_vector_sum')
    ASF.set_normalized_energy_vector(ASF.features['energy_vector_squared'], name='energy_vector_squared')
    ASF.set_cosine_similarity_vector(frequencies=complex_frequencies, complex_values=complex_result)
    ASF.set_diff_vector()
    all_feats = ASF.get_all_features()
    if hasattr(all_feats, "iloc"):
        row = all_feats.iloc[0]
    else:
        row = all_feats
    features_dict = {
        "period_magnitude_deviation_vector": row["period_magnitude_deviation_vector"],
        "normalized_energy_vector_sum": row["normalized_energy_vector_sum"],
        "normalized_energy_vector_squared": row["normalized_energy_vector_squared"],
        "cosine_similarity_vector": row["cosine_similarity_vector"],
        "segment_start": start_time,
        "base_f0":base_f0
    }
    return features_dict


def get_segment_vectors_single(csv_path, audio_path, f0mode="normal", segment_duration_ms=20, 
                               shiftmode="ratio", shift_point=(1/3, 0.5, 1), 
                               label_adj=(1/5, 1/4, 1/3, 1/2, 2/3, 1, 1.5, 2, 3, 4, 5),
                               output_csv_path=None):
    """
    从单个 CSV 与对应音频中提取特征，并输出一个 *_features.csv 文件。

    参数
    ----
    csv_path : str
        输入的 CSV 路径（包含 start_time/time/onset 和 offset 列）。
    audio_path : str
        对应的音频文件路径（.wav）。
    f0mode : str
        选取基频的方法，"normal" | "fft" | "crepe" | "pyin"。
    segment_duration_ms : int
        每个音频段的持续时间（毫秒）。
    shiftmode : str
        偏移模式，"key" | "ratio"。
    shift_point : tuple
        需要生成的多个偏移点。
    label_adj : tuple
        标签调整集合。
    output_csv_path : str | None
        输出特征 CSV 路径；若为 None，则保存在输入 CSV 同目录，命名为 base_name+"_features.csv"。
    """
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    if output_csv_path is None:
        output_csv_path = os.path.join(os.path.dirname(csv_path), f"{base_name}_features.csv")

    # 初始化一个列表来存储所有特征数据
    all_features = []

    # Step 1: 读取csv信息并音频切块
    df_valid = pd.read_csv(csv_path)
    
    if "in_tune" not in df_valid.columns:
        print(f"未找到 in_tune 列，跳过文件: {csv_path}")
        return
    df_valid = df_valid[df_valid["in_tune"] == True].reset_index(drop=True)
    if df_valid.empty:
        print(f"过滤后无有效样本（in_tune=True）：{csv_path}")
        return
    
    # 检查列名
    if "start_time" in df_valid.columns:
        start_times = df_valid["start_time"].round().astype(int).tolist()
    elif "time" in df_valid.columns:
        print("Warning: 使用 'time' 列代替 'start_time'")
        start_times = df_valid["time"].round().astype(int).tolist()
    elif "onset" in df_valid.columns:
        print("Warning: 使用 'onset' 列代替 'start_time'")
        start_times = df_valid["onset"].round().astype(int).tolist()
    else:
        raise KeyError(f"未找到 'start_time' 列，CSV 列有: {list(df_valid.columns)}")

    if "offset" not in df_valid.columns:
        raise KeyError("未找到 'offset' 列，无法进行偏移计算。")
    offset_values = df_valid["offset"].tolist()

    # 切段输出目录可与批处理一致
    output_directory = "audios/cut_sheets"
    cut_audio(audio_path, output_directory, start_times, segment_duration_ms)

    # Step 2: Process each segment to extract energy vectors
    for idx, (start_time, offset) in enumerate(zip(start_times, offset_values)):
        segment_features = []

        # 选取基频
        if f0mode == "normal":
            base_f0 = df_valid.loc[idx, "frequency"]
        else:
            segment_file = f"{output_directory}/segment_{start_time // 1000}s.wav"
            audio = PydubAudioSegment.from_file(segment_file)
            samples = np.array(audio.get_array_of_samples())
            sample_rate = audio.frame_rate

            AS = AudioSegment(samples, sample_rate)

            if f0mode == "fft":
                base_f0 = AS.FFT_f0()
            elif f0mode == "crepe":
                base_f0 = AS.crepe_f0()
            elif f0mode == "pyin":
                base_f0 = AS.Pyin_f0()
            else:
                raise ValueError(f"Unrecognized mode: {f0mode}")

        # 偏移值转换
        if shiftmode == "ratio":
            diff = convert_value(offset)
        else:
            diff = offset

        if diff != 0:
            base_f0 = process_pyinf0_and_label(base_f0, diff, shiftmode)

        # 为每个shift_point创建数据
        for i in shift_point:
            shift_base_f0 = shift_pyinf0_and_label(base_f0, i, shiftmode)
            if shift_base_f0 < 100:
                continue  # 跳过低频样本

            segment_file = f"{output_directory}/segment_{start_time // 1000}s.wav"
            audio = PydubAudioSegment.from_file(segment_file)
            samples = np.array(audio.get_array_of_samples())
            sample_rate = audio.frame_rate

            ASF = AudioSegmentFeatures(samples=samples, sample_rate=sample_rate, base_f0=shift_base_f0)

            # 设置 target frequencies
            ASF.set_target_frequencies(mode = shiftmode, label_adj=label_adj)

            # 临时计算 FFT
            frequencies, magnitudes = ASF.perform_fft(normalize_by_frequency=True)
            frequencies_raw, magnitudes_raw = ASF.perform_fft(normalize_by_frequency=False)
            complex_frequencies, complex_result = ASF.perform_fft_complex()

            # 计算能量向量
            ASF.set_energy_vector_sum(frequencies=frequencies, magnitude=magnitudes)
            ASF.set_energy_vector_squared(frequencies=frequencies_raw, magnitude=magnitudes_raw)

            # 计算归一化能量向量
            ASF.set_normalized_energy_vector(ASF.features['energy_vector_sum'], name='energy_vector_sum')
            ASF.set_normalized_energy_vector(ASF.features['energy_vector_squared'], name='energy_vector_squared')

            # 计算COS向量
            ASF.set_cosine_similarity_vector(
                frequencies=complex_frequencies,
                complex_values=complex_result
            )

            # 计算周期绝对差
            ASF.set_diff_vector()

            # 获取所有特征
            features_df = ASF.get_all_features()

            # 将向量列转为合法 JSON 字符串，且仅对 ndarray 或 list 类型转换
            for col in ["period_wave_abs_diff", 
                        "normalized_energy_vector_sum", 
                        "normalized_energy_vector_squared", 
                        "cosine_similarity_vector"]:
                if col in features_df.columns:
                    features_df[col] = features_df[col].apply(safe_json_list)

            features_df["segment_start"] = start_time
            features_df["shift_base_f0"] = shift_base_f0
            features_df["shift_fre_point"] = point_to_index(i)

            segment_features.append(features_df)

        # 合并当前segment的所有shift_point特征
        if segment_features:
            segment_df = pd.concat(segment_features, ignore_index=True)
            all_features.append(segment_df)

    # 保存当前文件的所有特征到一个CSV
    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)
        final_df.to_csv(output_csv_path, index=False)
        print(f"所有特征已保存到 {output_csv_path}")
    else:
        print(f"未生成有效特征: {base_name}")


def get_segment_vectors(input_folder, f0mode="normal", segment_duration_ms=20, 
                       shiftmode="ratio", shift_point=(1/3, 0.5, 1), 
                       label_adj=(1/5, 1/4, 1/3, 1/2, 2/3, 1, 1.5, 2, 3, 4, 5)):
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
    
    for valid_csv in csv_files:
        base_name = os.path.splitext(os.path.basename(valid_csv))[0]
        input_audio_path = os.path.join(input_folder, base_name + '.wav')
        output_csv_path = os.path.join(input_folder, f"{base_name}_features.csv")
        
        # 初始化一个列表来存储所有特征数据
        all_features = []
        
        # Step 1: 读取csv信息并音频切块
        df_valid = pd.read_csv(valid_csv)
        if "in_tune" not in df_valid.columns:
            print(f"未找到 in_tune 列，跳过文件: {base_name}")
            return
        df_valid = df_valid[df_valid["in_tune"] == True].reset_index(drop=True)
        if df_valid.empty:
            print(f"过滤后无有效样本（in_tune=True）：{base_name}")
            return
        # 检查列名
        if "start_time" in df_valid.columns:
            start_times = df_valid["start_time"].round().astype(int).tolist()
        elif "time" in df_valid.columns:
            print("Warning: 使用 'time' 列代替 'start_time'")
            start_times = df_valid["time"].round().astype(int).tolist()
        elif "onset" in df_valid.columns:
            print("Warning: 使用 'onset' 列代替 'start_time'")
            start_times = df_valid["onset"].round().astype(int).tolist()
        else:
            raise KeyError(f"未找到 'start_time' 列，CSV 列有: {list(df_valid.columns)}")
        offset_values = df_valid["offset"].tolist()
        output_directory = "audios/cut_sheets"
        cut_audio(input_audio_path, output_directory, start_times, segment_duration_ms)

        # Step 2: Process each segment to extract energy vectors
        for idx, (start_time, offset) in enumerate(zip(start_times, offset_values)):
            segment_features = []
            
            # 选取基频
            if f0mode == "normal":
                base_f0 = df_valid.loc[idx, "frequency"]
            else:
                segment_file = f"{output_directory}/segment_{start_time // 1000}s.wav"
                audio = PydubAudioSegment.from_file(segment_file)
                samples = np.array(audio.get_array_of_samples())
                sample_rate = audio.frame_rate

                AS = AudioSegment(samples, sample_rate)

                if f0mode == "fft":
                    base_f0 = AS.FFT_f0()
                elif f0mode == "crepe":
                    base_f0 = AS.crepe_f0()
                elif f0mode == "pyin":
                    base_f0 = AS.Pyin_f0()
                else:
                    raise ValueError(f"Unrecognized mode: {f0mode}")

            # 偏移值转换
            if shiftmode == "ratio":
                diff = convert_value(offset)
            else:
                diff = offset

            if diff != 0:
                base_f0 = process_pyinf0_and_label(base_f0, diff, shiftmode)

            # 为每个shift_point创建数据
            for i in shift_point:
                shift_base_f0 = shift_pyinf0_and_label(base_f0, i, shiftmode)
                if shift_base_f0 < 100:
                    continue  # 跳过低频样本
                    
                segment_file = f"{output_directory}/segment_{start_time // 1000}s.wav"
                audio = PydubAudioSegment.from_file(segment_file)
                samples = np.array(audio.get_array_of_samples())
                sample_rate = audio.frame_rate
                
                ASF = AudioSegmentFeatures(samples=samples, sample_rate=sample_rate, base_f0=shift_base_f0)

                # 设置 target frequencies
                ASF.set_target_frequencies(mode = shiftmode, label_adj=label_adj)

                # 临时计算 FFT
                frequencies, magnitudes = ASF.perform_fft(normalize_by_frequency=True)
                frequencies_raw, magnitudes_raw = ASF.perform_fft(normalize_by_frequency=False)
                complex_frequencies, complex_result = ASF.perform_fft_complex()

                # 计算能量向量
                ASF.set_energy_vector_sum(frequencies=frequencies, magnitude=magnitudes)
                ASF.set_energy_vector_squared(frequencies=frequencies_raw, magnitude=magnitudes_raw)

                # 计算归一化能量向量
                ASF.set_normalized_energy_vector(ASF.features['energy_vector_sum'], name='energy_vector_sum')
                ASF.set_normalized_energy_vector(ASF.features['energy_vector_squared'], name='energy_vector_squared')

                # 计算COS向量
                ASF.set_cosine_similarity_vector(
                    frequencies=complex_frequencies,
                    complex_values=complex_result
                )

                # 计算周期绝对差
                ASF.set_diff_vector()

                # 获取所有特征并添加到当前segment的特征列表
                features_df = ASF.get_all_features()
                features_df["segment_start"] = start_time
                features_df["shift_base_f0"] = shift_base_f0
                features_df["shift_fre_point"] = i
                    # 将向量列转为合法 JSON 字符串
                for col in ["period_magnitude_deviation_vector", 
                            "normalized_energy_vector_sum", 
                            "normalized_energy_vector_squared", 
                            "cosine_similarity_vector"]:
                    if col in features_df.columns:
                        features_df[col] = features_df[col].apply(safe_json_list)
                segment_features.append(features_df)

            
            # 合并当前segment的所有shift_point特征
            if segment_features:
                segment_df = pd.concat(segment_features, ignore_index=True)
                all_features.append(segment_df)
        
        # 保存当前文件的所有特征到一个CSV
        if all_features:
            final_df = pd.concat(all_features, ignore_index=True)
            final_df.to_csv(output_csv_path, index=False)
            print(f"所有特征已保存到 {output_csv_path}")
        else:
            print(f"未生成有效特征: {base_name}")
