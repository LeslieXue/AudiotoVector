import numpy as np
from scipy.signal.windows import hann
from scipy.fftpack import fft
import librosa
import pandas as pd

class AudioSegmentFeatures:
    def __init__(self, samples = np.array([]), sample_rate = 0, base_f0 = 0):
        self._samples = samples
        self._sample_rate = sample_rate
        self._base_f0 = base_f0
        self._features = {}

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, value):
        self._samples = value

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self._sample_rate = value

    @property
    def base_f0(self):
        return self._base_f0

    @base_f0.setter
    def base_f0(self, value):
        self._base_f0 = value

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = value

    def get_target_frequencies(self, mode='ratio', label_adj =(1/5, 1/4, 1/3, 1/2, 2/3, 1, 1.5, 2, 3, 4, 5)):
        """
        获取目标频率列表，支持三种模式：
        1. 'key'：原始默认频率范围（61维）
        2. 'ratio'：使用基频倍数
        3. 'normal'：从 -24 到 +36 的 61 个半音频率
        """
        semitone_ratio = 2 ** (1/12)
        if mode == 'key':
            for value in label_adj:
                if not isinstance(value, int):
                    raise ValueError("label_adj 中的值必须是整数")
            return [self.base_f0 * (semitone_ratio ** (i - 24)) for i in label_adj]
        elif mode == 'ratio':
            return [self.base_f0 * r for r in label_adj]
        elif mode == 'normal':
            return [self.base_f0 * (semitone_ratio ** i) for i in range(-24, 37)]
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def set_target_frequencies(self, mode='ratio', label_adj=(1/5, 1/4, 1/3, 1/2, 2/3, 1, 1.5, 2, 3, 4, 5)):
        """
        设置目标频率列表到 features 中，支持三种模式：
        1. 'key'：原始默认频率范围
        2. 'ratio'：使用基频倍数
        3. 'normal'：从 -24 到 +36 的 61 个半音频率
        """
        self._features['target_frequencies'] = self.get_target_frequencies(mode=mode, label_adj=label_adj)
   
    def perform_fft(self, zero_padding_factor=20, normalize_by_frequency=True):
        """
        Perform FFT on the audio samples with optional zero padding and frequency normalization.

        :param zero_padding_factor: Factor by which to increase the sample length with zero padding.
        :param normalize_by_frequency: Whether to normalize the magnitude by frequency.
        :return: Tuple of (frequencies, magnitudes).
        """
        # Remove DC component (zero-mean)
        samples = self.samples - np.mean(self.samples)

        # Apply a Hann window
        window = hann(len(samples))
        samples = samples * window

        # Zero padding
        padded_length = len(samples) * zero_padding_factor
        padded_samples = np.pad(samples, (0, padded_length - len(samples)), mode='constant')

        # Perform FFT
        fft_result = fft(padded_samples)
        magnitude = np.abs(fft_result[:len(fft_result) // 2])  # Take the positive frequencies
        frequencies = np.fft.fftfreq(len(fft_result), d=1/self.sample_rate)[:len(fft_result) // 2]

        # Normalize by frequency if required
        if normalize_by_frequency:
            # Avoid division by zero for the DC component (frequency = 0)
            magnitude = magnitude / np.maximum(frequencies, 1e-10)

        return frequencies, magnitude
    
    def set_fft_features(self):
        frequencies, magnitudes = self.perform_fft(normalize_by_frequency=False)
        self._features['FFT_frequencies'] = frequencies
        self._features['FFT_magnitudes'] = magnitudes   
        frequencies_norm, magnitudes_norm = self.perform_fft(normalize_by_frequency=True)
        self._features['FFT_frequencies_norm'] = frequencies_norm
        self._features['FFT_magnitudes_norm'] = magnitudes_norm 

    def perform_fft_complex(self, zero_padding_factor=20):
        """
        Perform FFT on the audio samples and return the complex result.

        :param zero_padding_factor: Factor by which to increase the sample length with zero padding.
        :return: Tuple of (frequencies, complex_result).
        """
        # Remove DC component (zero-mean)
        samples = self.samples - np.mean(self.samples)

        # Apply a Hann window
        window = hann(len(samples))
        samples = samples * window

        # Zero padding
        padded_length = len(samples) * zero_padding_factor
        padded_samples = np.pad(samples, (0, padded_length - len(samples)), mode='constant')

        # Perform FFT
        fft_result = fft(padded_samples)
        frequencies = np.fft.fftfreq(len(fft_result), d=1/self.sample_rate)

        # Return only the positive frequencies and their corresponding complex values
        positive_frequencies = frequencies[:len(fft_result) // 2]
        positive_fft_result = fft_result[:len(fft_result) // 2]

        return positive_frequencies, positive_fft_result
    
    def set_fft_complex_features(self):
        frequencies, complex_result = self.perform_fft_complex()
        self._features['FFT_complex_frequencies'] = frequencies
        self._features['FFT_complex_result'] = complex_result

    def calculate_energy_vector_sum(self, frequencies = None, magnitude = None, target_frequencies = None, bandwidth_cents=50):
        """
        Calculate the energy vector by summing the normalized FFT magnitudes within the target frequency ranges.

        Uses self._features['FFT_frequencies_norm'], self._features['FFT_magnitudes_norm'], and self._features['target_frequencies'].
        :param bandwidth_cents: Bandwidth in cents for energy calculation (default: ±50 cents).
        :return: Energy vector (sum of magnitudes).
        """
        if frequencies is None:
            frequencies = self._features['FFT_frequencies_norm']
        if magnitude is None:   
            magnitude = self._features['FFT_magnitudes_norm']
        if target_frequencies is None:
            target_frequencies = self._features['target_frequencies']
       
        energy_vector_sum = []
        bandwidth_ratio = 2 ** (bandwidth_cents / 1200)  # Convert cents to frequency ratio

        for target_freq in target_frequencies:
            lower_bound = target_freq / bandwidth_ratio
            upper_bound = target_freq * bandwidth_ratio

            # Find indices within the frequency range
            indices = np.where((frequencies >= lower_bound) & (frequencies <= upper_bound))[0]

            # Sum the magnitudes within the range
            energy = np.sum(magnitude[indices])
            energy_vector_sum.append(energy)

        return energy_vector_sum
    
    def set_energy_vector_sum(self, frequencies=None, magnitude=None, target_frequencies=None, bandwidth_cents=50):
        energy_vector_sum = self.calculate_energy_vector_sum(
            frequencies=frequencies,
            magnitude=magnitude,
            target_frequencies=target_frequencies,
            bandwidth_cents=bandwidth_cents
        )
        self._features['energy_vector_sum'] = energy_vector_sum
    
    def calculate_energy_vector_squared(self, frequencies=None, magnitude=None, target_frequencies=None, bandwidth_cents=50):
        """
        Calculate the energy vector by summing the squared magnitudes within the target frequency ranges.

        Uses self._features['FFT_frequencies_norm'], self._features['FFT_magnitudes_norm'], and self._features['target_frequencies'] if no arguments provided.
        :param frequencies: Array of FFT frequencies.
        :param magnitude: Array of FFT magnitudes.
        :param target_frequencies: List of target frequencies.
        :param bandwidth_cents: Bandwidth in cents for energy calculation (default: ±50 cents).
        :return: Energy vector (sum of squared magnitudes).
        """
        if frequencies is None:
            frequencies = self._features['FFT_frequencies_norm']
        if magnitude is None:
            magnitude = self._features['FFT_magnitudes_norm']
        if target_frequencies is None:
            target_frequencies = self._features['target_frequencies']

        energy_vector_squared = []
        bandwidth_ratio = 2 ** (bandwidth_cents / 1200)  # Convert cents to frequency ratio

        for target_freq in target_frequencies:
            lower_bound = target_freq / bandwidth_ratio
            upper_bound = target_freq * bandwidth_ratio

            # Find indices within the frequency range
            indices = np.where((frequencies >= lower_bound) & (frequencies <= upper_bound))[0]

            # Sum the squared magnitudes within the range
            energy = np.sum(magnitude[indices] ** 2)
            energy_vector_squared.append(energy)

        return energy_vector_squared

    def set_energy_vector_squared(self, frequencies=None, magnitude=None, target_frequencies=None, bandwidth_cents=50):
        energy_vector_squared = self.calculate_energy_vector_squared(
            frequencies=frequencies,
            magnitude=magnitude,
            target_frequencies=target_frequencies,
            bandwidth_cents=bandwidth_cents
        )
        self._features['energy_vector_squared'] = energy_vector_squared

    def normalize_energy_vector(self, energy_vector, mode="f0"):
        """
        Normalize the energy vector.

        :param energy_vector: List or np.array of energy values. If None, use self._features['energy_vector'].
        :param mode: "f0" (each element divided by the 25th element) or "max" (each element divided by max value).
                     Defaults to "f0". Other values perform L2 norm normalization.
        :return: Normalized energy vector.
        """
        energy_vector = np.array(energy_vector)
        
        if mode == "f0":
            norm_value = np.abs(energy_vector[24]) if len(energy_vector) > 24 else 1e-8
            norm_value = norm_value if norm_value != 0 else 1e-8
            normalized_vector = energy_vector / norm_value
        elif mode == "max":
            norm_value = np.max(np.abs(energy_vector)) if len(energy_vector) > 0 else 1e-8
            norm_value = norm_value if norm_value != 0 else 1e-8
            normalized_vector = energy_vector / norm_value
        else:
            vector_length = np.linalg.norm(energy_vector)
            norm_value = vector_length if vector_length != 0 else 1e-8
            normalized_vector = energy_vector / norm_value

        return normalized_vector
    
    def set_normalized_energy_vector(self, energy_vector, mode="f0", name="energy_vector"):
        """
        Set the normalized energy vector in the features dictionary with a dynamic name.

        :param energy_vector: List or np.array of energy values.
        :param mode: Normalization mode ("f0", "max", or other for L2 norm).
        :param name: Name to use in the features dictionary (default: "energy_vector").
        """
        normalized_vector = self.normalize_energy_vector(energy_vector, mode)
        self._features[f'normalized_{name}'] = normalized_vector


    def calculate_cosine_similarity_vector(self, frequencies=None, complex_values=None, target_frequencies=None, fundamental_frequency=None, bandwidth_cents=50):
        """
        Calculate a 61-dimensional vector where each element is the cosine of the angle between the complex sum
        of a target frequency range and the fundamental frequency's complex result.

        Uses self._features['FFT_complex_frequencies'], self._features['FFT_complex_result'], 
        and self._features['target_frequencies'] if parameters are not provided.
        :param frequencies: Array of FFT frequencies.
        :param complex_values: Array of FFT complex values.
        :param target_frequencies: List of target frequencies (61 points).
        :param fundamental_frequency: The fundamental frequency in Hz.
        :param bandwidth_cents: Bandwidth in cents for energy calculation (default: ±50 cents).
        :return: 61-dimensional vector of cosine similarity values.
        """
        if frequencies is None:
            frequencies = self._features['FFT_complex_frequencies']
        if complex_values is None:
            complex_values = self._features['FFT_complex_result']
        if target_frequencies is None:
            target_frequencies = self._features['target_frequencies']
        if fundamental_frequency is None:
            fundamental_frequency = self.base_f0

        # Calculate the complex sum for the fundamental frequency
        bandwidth_ratio = 2 ** (bandwidth_cents / 1200)  # Convert cents to frequency ratio
        lower_bound_f0 = fundamental_frequency / bandwidth_ratio
        upper_bound_f0 = fundamental_frequency * bandwidth_ratio
        indices_f0 = np.where((frequencies >= lower_bound_f0) & (frequencies <= upper_bound_f0))[0]
        fundamental_complex_sum = np.sum(complex_values[indices_f0])

        # Initialize the cosine similarity vector
        cosine_similarity_vector = []

        for target_freq in target_frequencies:
            lower_bound = target_freq / bandwidth_ratio
            upper_bound = target_freq * bandwidth_ratio

            # Find indices within the frequency range
            indices = np.where((frequencies >= lower_bound) & (frequencies <= upper_bound))[0]

            # Sum the complex values within the range
            target_complex_sum = np.sum(complex_values[indices])

            # Calculate the cosine similarity
            dot_product = np.real(fundamental_complex_sum * np.conj(target_complex_sum))
            norm_fundamental = np.abs(fundamental_complex_sum)
            norm_target = np.abs(target_complex_sum)

            if norm_fundamental == 0 or norm_target == 0:
                cosine_similarity = 0  # Avoid division by zero
            else:
                cosine_similarity = dot_product / (norm_fundamental * norm_target)

            cosine_similarity_vector.append(cosine_similarity)

        return cosine_similarity_vector
    
    def set_cosine_similarity_vector(self, frequencies=None, complex_values=None, target_frequencies=None, fundamental_frequency=None, bandwidth_cents=50):
        """
        Set the cosine similarity vector in the features dictionary.

        :param frequencies: Array of FFT frequencies.
        :param complex_values: Array of FFT complex values.
        :param target_frequencies: List of target frequencies (61 points).
        :param fundamental_frequency: The fundamental frequency in Hz.
        :param bandwidth_cents: Bandwidth in cents for energy calculation (default: ±50 cents).
        """
        cosine_similarity_vector = self.calculate_cosine_similarity_vector(
            frequencies=frequencies,
            complex_values=complex_values,
            target_frequencies=target_frequencies,
            fundamental_frequency=fundamental_frequency,
            bandwidth_cents=bandwidth_cents
        )
        self._features['cosine_similarity_vector'] = cosine_similarity_vector

    def process_by_frequency_custom(self, frequency_list, audio_segment, sample_rate):
        """
        Apply custom processing to the audio segment by each frequency in frequency_list.

        :param frequency_list: List of frequencies to process.
        :param audio_segment: Optional audio segment array. Defaults to self.samples.
        :param sample_rate: Optional sample rate. Defaults to self.sample_rate.
        :return: List of average period waves for each frequency.
        """

        segment_len = len(audio_segment)
        avg_period_waves = []

        for frequency in frequency_list:
            period = 1000 / frequency
            period_samples = int(round(period * sample_rate / 1000))    # 计算每个周期的采样点数
            if period_samples == 0:
                avg_period_waves.append(np.zeros(1))
                continue

            whole_periods = segment_len // period_samples    # 计算完整周期数
            fraction = (segment_len / period_samples) - whole_periods  # 剩余比例
            n = whole_periods
            cut_point = int(fraction * period_samples)

            # 头部和尾巴
            parts_head = [audio_segment[i * period_samples : i * period_samples + cut_point]
                for i in range(n) if i * period_samples + cut_point <= segment_len]
            part_tail = audio_segment[-cut_point:] if cut_point > 0 else np.array([], dtype=audio_segment.dtype)
            all_heads = parts_head + ([part_tail] if cut_point > 0 else [])

            # 尾部（周期剩余部分）
            parts_body = [audio_segment[i * period_samples + cut_point : (i + 1) * period_samples]
                for i in range(n) if (i + 1) * period_samples <= segment_len and cut_point < period_samples]

            # 对齐长度
            valid_heads = [p for p in all_heads if isinstance(p, np.ndarray) and p.ndim == 1 and len(p) > 0]
            valid_bodies = [p for p in parts_body if isinstance(p, np.ndarray) and p.ndim == 1 and len(p) > 0]

            min_head_len = min([len(p) for p in valid_heads]) if valid_heads else 0
            min_body_len = min([len(p) for p in valid_bodies]) if valid_bodies else 0

            if min_head_len == 0 and min_body_len == 0:
                avg_period_waves.append(np.zeros(1))
                continue

            avg_head_wave = np.mean(np.stack([p[:min_head_len] for p in valid_heads], axis=0), axis=0) if min_head_len > 0 else np.array([])
            avg_body_wave = np.mean(np.stack([p[:min_body_len] for p in valid_bodies], axis=0), axis=0) if min_body_len > 0 else np.array([])

            avg_period_wave = np.concatenate([avg_head_wave, avg_body_wave])
            avg_period_waves.append(avg_period_wave)

        return avg_period_waves

    def period_wave_abs_diff(self, target_frequencies=None, audio_segment=None, sample_rate=None):
        """
        对每个频率点，将平均周期波形周期性铺满音频，与原始音频周期内做绝对差和，输出向量。
        默认使用 self.samples, self.sample_rate, self._features['target_frequencies']。
        """
        if audio_segment is None:
            audio_segment = self.samples
        if sample_rate is None:
            sample_rate = self.sample_rate
        if target_frequencies is None:
            target_frequencies = self._features['target_frequencies']

        avg_period_waves = self.process_by_frequency_custom(
            target_frequencies, audio_segment=audio_segment, sample_rate=sample_rate
        )

        segment_len = len(audio_segment)
        diff_vector = []
        period_counts = []

        for idx, frequency in enumerate(target_frequencies):
            avg_period_wave = avg_period_waves[idx]
            period_samples = len(avg_period_wave)
            if period_samples == 0:
                diff_vector.append(0.0)
                continue

            n_full = segment_len // period_samples
            remain = segment_len % period_samples
            n_periods = segment_len / period_samples
            period_counts.append(n_periods)

            total_diff = 0.0

            # 处理完整周期
            for i in range(n_full):
                start = i * period_samples
                end = start + period_samples
                orig = audio_segment[start:end]
                total_diff += np.sum(np.abs(orig - avg_period_wave))

            # 处理残余周期
            if remain > 0:
                start = n_full * period_samples
                end = segment_len
                orig = audio_segment[start:end]
                avg_part = avg_period_wave[:remain]
                total_diff += np.sum(np.abs(orig - avg_part))

            diff_vector.append(total_diff)

        diff_vector = np.array(diff_vector)
        period_counts = np.array(period_counts)
        diff_vector = diff_vector / np.log(np.e + np.sqrt(period_counts))

        # 周期不足 2 的掩码
        low_period_mask = period_counts < 2

        # 0-1 min-max 标准化
        min_val = np.min(diff_vector)
        max_val = np.max(diff_vector)
        if max_val - min_val > 1e-8:
            diff_vector = (diff_vector - min_val) / (max_val - min_val)
        else:
            diff_vector = np.zeros_like(diff_vector)

        # 设置周期不足 2 的维度为 1.0
        diff_vector[low_period_mask] = 1.0

        return diff_vector
    
    def set_diff_vector(self, target_frequencies=None, audio_segment=None, sample_rate=None):
        """
        Set the period wave absolute difference vector in the features dictionary.

        :param target_frequencies: List of target frequencies.
        :param audio_segment: Optional audio segment array. Defaults to self.samples.
        :param sample_rate: Optional sample rate. Defaults to self.sample_rate.
        """
        diff_vector = self.period_wave_abs_diff(
            target_frequencies=target_frequencies,
            audio_segment=audio_segment,
            sample_rate=sample_rate
        )
        self._features['period_wave_abs_diff'] = diff_vector


    def get_all_features(self):
        """
        返回处理后的所有 features，每个特征占一列，每列保存为字符串形式，适合直接写入 CSV。
        """
        if not self.features:
            raise ValueError("features 为空，无法返回")

        processed_features = {}
        for feat_name, feat_value in self.features.items():
            # 将每个特征转成字符串形式
            processed_features[feat_name] = [str(feat_value)]

        import pandas as pd
        df = pd.DataFrame(processed_features)
        return df









def main(audio_file_path, output_csv_path):
    # 1. 读取音频文件
    y, sr = librosa.load(audio_file_path, sr=None, mono=True)  # 保持原采样率

    # 2. 随机抽取 20ms
    duration_ms = 20
    duration_samples = int(sr * duration_ms / 1000)
    if len(y) <= duration_samples:
        start_idx = 0
    else:
        start_idx = np.random.randint(0, len(y) - duration_samples)
    samples_segment = y[start_idx:start_idx + duration_samples]

    # 3. 初始化 AudioSegmentFeatures
    base_f0 = 440.0  # Hz
    segment_features = AudioSegmentFeatures(samples=samples_segment, sample_rate=sr, base_f0=base_f0)

    # 4. 设置 target frequencies
    segment_features.set_target_frequencies(mode = 'ratio', label_adj=(1/5, 1/4, 1/3, 1/2, 2/3, 1, 1.5, 2, 3, 4, 5))

    # 临时计算 FFT
    frequencies, magnitudes = segment_features.perform_fft(normalize_by_frequency=True)
    frequencies_raw, magnitudes_raw = segment_features.perform_fft(normalize_by_frequency=False)
    complex_frequencies, complex_result = segment_features.perform_fft_complex()

    # 计算能量向量
    segment_features.set_energy_vector_sum(frequencies=frequencies, magnitude=magnitudes)
    segment_features.set_energy_vector_squared(frequencies=frequencies_raw, magnitude=magnitudes_raw)

    # 计算归一化
    segment_features.set_normalized_energy_vector(segment_features.features['energy_vector_sum'], name='energy_vector_sum')
    segment_features.set_normalized_energy_vector(segment_features.features['energy_vector_squared'], name='energy_vector_squared')

    # 计算余弦相似度
    segment_features.set_cosine_similarity_vector(
        frequencies=complex_frequencies,
        complex_values=complex_result
    )

    # 计算周期绝对差
    segment_features.set_diff_vector()
    # 6. 获取所有特征 DataFrame
    df_features = segment_features.get_all_features()

    # 7. 保存到 CSV
    df_features.to_csv(output_csv_path, index=False)
    print(f"特征已保存到 {output_csv_path}")

if __name__ == "__main__":
    audio_file_path = "1.wav"  # 替换为你的音频文件路径
    output_csv_path = "output_features.csv"
    main(audio_file_path, output_csv_path)


    # 读取 CSV
    df = pd.read_csv("output_features.csv")

    # 打印整个 DataFrame
    print(df)

    # 如果想逐列打印，可以用：
    for col in df.columns:
        print(f"{col}: {df[col].values[0]}")  # 因为每行是一个音频片段