from audio_segment import AudioSegment
from audio_segment_DataGen import extract_features_predict
import sys
sys.path.append('util')  
import torch
import torch.nn.functional as F
from CNN_model import F0CNN
import numpy as np
import pydub

def FeaturePredictSegment(audio_file, base_frequency=None, model_path="f1_model.pt"):
    """
    对输入的音频文件进行特征提取和预测。
    参数:
        audio_file: 音频文件路径
        base_frequency: (可选) 基频，如果为None，则自动估算
        model_path: CNN模型权重文件路径
    返回:
        dict，包含预测标签和概率
    """
    # 1. 读取音频文件
    audio = pydub.AudioSegment.from_file(audio_file)
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate

    # 2. 基频处理
    if base_frequency is None:
        segment = AudioSegment(samples, sample_rate)
        base_frequency = segment.pyin_f0()
    else:
        base_frequency = base_frequency

    # 3. 特征提取
    features = extract_features_predict(samples, sample_rate, base_frequency)
    # features 应该是一个 dict 或类似结构，确保其格式与模型输入一致

    # 4. 加载模型
    model = F0CNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 5. 转换为模型输入
    # 假定 extract_features_from_segment 返回 dict，含有以下键
    data = torch.tensor([
        features["period_magnitude_deviation_vector"],
        features["normalized_energy_vector_sum"],
        features["normalized_energy_vector_squared"],
        features["cosine_similarity_vector"]
    ], dtype=torch.float32).T  # shape: (N, 4)
    # 如果模型需要 batch 维度
    data = data.unsqueeze(0)  # shape: (1, N, 4)

    # 6. 推理
    with torch.no_grad():
        outputs = model(data)
        probs = F.softmax(outputs, dim=1)
        pred_label = torch.argmax(probs, dim=1)

    # 7. 返回结果
    result = {
        "pred_label": int(pred_label.item()),
        "probabilities": probs.squeeze().cpu().numpy().tolist()
    }
    return result


