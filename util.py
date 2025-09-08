from pydub import AudioSegment
import os
import json
import pandas as pd
import numpy as np
from scipy.io import wavfile
import crepe

def point_to_index(point):
    mapping = {
        1/5 : 0,
        1/4 : 1,
        1/3 : 2,
        2/3 : 3,
        1/2 : 4,
        1 : 5, 
        3/2 : 6,
        2 : 7,
        3 : 8,
        4 : 9,
        5 : 10
    }
    return mapping.get(point, None)  

def index_to_point(index):
    reverse_mapping = {
        0: 1/5,
        1: 1/4,
        2: 1/3,
        3: 1/2,
        4: 2/3,
        5: 1,
        6: 3/2,
        7: 2,
        8: 3,
        9: 4,
        10: 5
    }
    return reverse_mapping.get(index, None)

def convert_value(offset):
    mapping = {
        12: 2,
        -4: 0.8,
        -5: 0.8,
        -7: float(2/3),
        -12: 0.5,
        -19: float(1/3),
        -28: 0.2,
        0: 1
    }
    return mapping.get(offset, None)  # 如果没有匹配，返回None


def process_pyinf0_and_label(pyinf0, diff, mode="key"):
    """
    根据mode处理label并计算normed_pyinf0。

    参数:
        pyinf0 (array-like): pyinf0 列的值
        label (array-like): label 列的值
        mode (str): "key" 使用映射转换label，"ratio" 保持原label值

    返回:
        numpy.ndarray: normed_pyinf0 数组
    """
    # 根据 mode 处理 label
    if mode == "key":
        normed_pyinf0 = pyinf0 * (2 ** (diff / 12))
    elif mode == "ratio":
        normed_pyinf0 = pyinf0 * diff
    else:
        raise ValueError("mode 必须是 'key' 或 'ratio'")

    return normed_pyinf0


def shift_pyinf0_and_label(pyinf0, shift, mode="key"):
    """
    根据给定的shiftvalue对pyinf0进行移调处理。

    参数:
        pyinf0 (array-like): 原始pyinf0值
        label (array-like): 标签值，用于计算基准频率
        shiftvalue (float): 移调值，以半音为单位
        mode (str): 处理label的模式，"key"或"ratio"

    返回:
        numpy.ndarray: 移调后的pyinf0数组
    """
    # 根据 mode 处理 label
    if mode == "key":
        shift_base_f0 = pyinf0 * (2 ** ( (24-shift) / 12))
    elif mode == "ratio":
        shift_base_f0 = pyinf0 / shift
    else:
        raise ValueError("mode 必须是 'key' 或 'ratio'")

    return shift_base_f0


# Get Frequency features by crepe

def feat_crepe(path, step_size=20):
    '''
    -----------------------------------
    Param : 
    -------
    step_size : int
        The stepsize of the sample (ms)
    -----------------------------------
    '''
    sr, audio = wavfile.read(path)
    t, frequency, confidence, activation = crepe.predict(audio, sr, step_size=step_size, viterbi=True)

    frequency = frequency.reshape((-1, 1))
    return frequency[1:], confidence[1:], activation[1:]


def cut_audio(file_path, output_dir, start_times_ms, segment_duration_ms=20):
    """
    Cut an audio file into multiple segments starting from specified times and save them.

    :param file_path: Path to the input audio file.
    :param output_dir: Directory to save the cut audio segments.
    :param start_times_ms: List of start times in milliseconds for the segments.
    :param segment_duration_ms: Duration of each segment in milliseconds (default: 100ms).
    """
    # Load the audio file
    audio = AudioSegment.from_file(file_path)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract and save each segment
    for start_time_ms in start_times_ms:
        segment = audio[start_time_ms:start_time_ms + segment_duration_ms]
        segment_file_name = f"segment_{start_time_ms // 1000}s.wav"
        segment_file_path = os.path.join(output_dir, segment_file_name)
        segment.export(segment_file_path, format="wav")


def batch_convert_json_to_csv(json_folder: str = 'random_samples', output_folder: str = 'random_samples', drop_zero_neighbors: int = 3, mode: str = "key", crepe_enabled: bool = False, crepe_step_ms: int = 10) -> list:
    """
    批量将 `json_folder` 中的音高检测 JSON 文件转换为 CSV，保存到 `output_folder`。

    处理流程：
    1. 读取每个 .json 文件中的 DetectedPitches.pitchUnitList，提取 (time, freq)
    2. 时间单位从秒转为毫秒
    3. 丢弃 time < 0 的行
    4. 删除频率为 0 的索引及其前后 `drop_zero_neighbors` 个样本
    5. 添加 label_1=1 和 label_0=0 两列

    参数：
        json_folder (str): 输入 JSON 文件夹路径
        output_folder (str): 输出 CSV 文件夹路径
        drop_zero_neighbors (int): 频率为 0 时向前后扩展丢弃的样本数，默认 3

    返回：
        List[str]: 成功保存的 CSV 文件路径列表
    """
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)

    saved_paths = []

    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            json_path = os.path.join(json_folder, filename)
            output_filename = os.path.splitext(filename)[0] + '.csv'
            output_path = os.path.join(output_folder, output_filename)

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 提取时间和频率
            pitch_units = data.get('DetectedPitches', {}).get('pitchUnitList', [])
            rows = [(item.get('time'), item.get('freq')) for item in pitch_units if ('time' in item and 'freq' in item)]

            # 时间单位从秒转为毫秒
            rows = [(t * 1000, f) for t, f in rows]

            # 转为 DataFrame 便于处理
            df = pd.DataFrame(rows, columns=['time', 'frequency'])

            # 丢弃非法时间并重置为连续索引
            df = df[df['time'] >= 0].reset_index(drop=True)

            # 在“连续索引”上寻找频率为 0 的行，并扩展邻域后删除
            bad_indices = df.index[df['frequency'] == 0]
            expanded_indices = set()
            for idx in bad_indices:
                start = max(0, idx - drop_zero_neighbors)
                end = min(len(df), idx + drop_zero_neighbors + 1)
                expanded_indices.update(range(start, end))

            # 只删除实际存在的索引，避免 KeyError
            safe_indices = [i for i in expanded_indices if i in df.index]
            df = df.drop(index=safe_indices, errors='ignore').reset_index(drop=True)

            if mode == "key":
                df['offset'] = 0
            elif mode == "ratio":
                df['offset'] = 1

            # 如果开启 crepe，则读取对应 wav 并写入 CrepeFreq / CrepeConf
            if crepe_enabled:
                base_name = os.path.splitext(filename)[0]
                wav_path = os.path.join(json_folder, f"{base_name}.wav")
                if not os.path.exists(wav_path):
                    print(f"⚠️ 找不到对应的音频文件: {wav_path}，跳过CREPE写入")
                else:
                    try:
                        frequency_arr, confidence_arr, _ = feat_crepe(wav_path, step_size=crepe_step_ms)
                        num_steps = len(frequency_arr)
                        crepe_times = np.arange(num_steps) * crepe_step_ms  # 毫秒

                        query_times = df['time'].values
                        # 为每个 query_time 寻找最近的 crepe 时间索引
                        idx_closest = np.abs(crepe_times[:, None] - query_times).argmin(axis=0)
                        matched_freq = frequency_arr[idx_closest].flatten()
                        matched_conf = confidence_arr[idx_closest].flatten()

                        df['CrepeFreq'] = matched_freq
                        df['CrepeConf'] = matched_conf

                        # 计算频率比值列（原始 frequency 与 CREPE 频率的比值）
                        # 避免除零/无穷：先计算，再将无穷替换为 NaN
                        df['freq_ratio'] = df['frequency'] / df['CrepeFreq']
                        df['freq_ratio'] = df['freq_ratio'].replace([np.inf, -np.inf], np.nan)

                        # ±100 音分范围对应的比值上下界：2^(±100/1200)
                        cents_tol = 100
                        lower = 2 ** (-cents_tol / 1200)
                        upper = 2 ** (cents_tol / 1200)

                        # 打标签：比值在 ±100 音分内 且 confidence > 0.8
                        df['in_tune'] = (
                            (df['CrepeConf'] > 0.8)
                            & (df['freq_ratio'] >= lower)
                            & (df['freq_ratio'] <= upper)
                        )
                    except Exception as e:
                        print(f"❌ CREPE 处理失败 {wav_path}: {e}")

            # 写入 CSV
            df.to_csv(output_path, index=False)
            saved_paths.append(output_path)

    return saved_paths