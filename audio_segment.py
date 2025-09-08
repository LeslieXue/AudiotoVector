from scipy.signal.windows import hann
import numpy as np
from scipy.fft import fft
from scipy.signal import find_peaks
import crepe
import pandas as pd

class AudioSegment:
    def __init__(self, samples=np.array([]), sample_rate=0):
        self._samples = samples    
        self._sample_rate = sample_rate 
        self._features = {}

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError("samples 必须是 numpy.ndarray")
        self._samples = value

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("sample_rate 必须是数字")
        if value <= 0:
            raise ValueError("sample_rate 必须是正数")
        self._sample_rate = value
        self._features['sample_rate'] = value

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        if not isinstance(value, dict):
            raise TypeError("features 必须是字典")
        self._features = value

    def set_FFT(self):
        self.features['FFT'] = self.FFT_f0(self.samples, self.sample_rate)

    def FFT_f0(self, samples, sample_rate):
        samples = samples - np.mean(samples)
        window = hann(len(samples))
        samples = samples * window
        padded_length = len(samples) * 20
        padded_samples = np.pad(samples, (0, padded_length - len(samples)), mode='constant')

        fft_result = fft(padded_samples)
        magnitude = np.abs(fft_result[:len(fft_result)//2])
        frequencies = np.fft.fftfreq(len(fft_result), d=1/sample_rate)[:len(fft_result)//2]

        peaks, _ = find_peaks(magnitude)
        if len(peaks) == 0:
            return None
        peak_freq = frequencies[peaks[np.argmax(magnitude[peaks])]]
        return peak_freq
    
    def set_crepe(self):
        time, frequency, confidence, activation = self.crepe_f0(self.samples, self.sample_rate)
        self.features['crepe_frequency'] = frequency.flatten()
        self.features['crepe_confidence'] = confidence.flatten()

    def crepe_f0(self, samples, sample_rate, step_size=20):
        time, frequency, confidence, activation = crepe.predict(samples, sample_rate, step_size=step_size, viterbi=True)
        frequency = frequency.reshape((-1, 1))
        return time[1:], frequency[1:], confidence[1:], activation[1:]
    
    def set_Pyin(self):
        self.features['pyin_f0'] = self.Pyin_f0(self.samples, self.sample_rate)

    def Pyin_f0(self, audio_data, sample_rate, frequency_range=(50.0, 2000.0), harmonic_thresh=0.2):
        half_buffer_size = len(audio_data) // 2
        yin_buffer = np.zeros(half_buffer_size)
        cumulative_sum_buffer = np.zeros(half_buffer_size)
        cumulative_sum = 0.0

        for tau in range(half_buffer_size):
            diff = audio_data[:half_buffer_size] - audio_data[tau:tau + half_buffer_size]
            yin_buffer[tau] = np.sum(diff ** 2)
            cumulative_sum += yin_buffer[tau]
            cumulative_sum_buffer[tau] = cumulative_sum

        cumulative_mean_buffer = np.ones(half_buffer_size)
        for tau in range(1, half_buffer_size):
            if cumulative_sum_buffer[tau] != 0:
                cumulative_mean_buffer[tau] = yin_buffer[tau] / (cumulative_sum_buffer[tau] / tau)

        tau_min = int(sample_rate / frequency_range[1])
        tau_max = int(sample_rate / frequency_range[0])
        tau_max = min(tau_max, half_buffer_size)
        
        f0 = 0.0
        for tau_index in range(tau_min, tau_max):
            if cumulative_mean_buffer[tau_index] < harmonic_thresh:
                while (tau_index + 1 < tau_max and
                       cumulative_mean_buffer[tau_index + 1] < cumulative_mean_buffer[tau_index]):
                    tau_index += 1

                if tau_index - 1 < 0 or tau_index + 1 >= len(cumulative_mean_buffer):
                    f0 = sample_rate / tau_index
                    break
                y0 = cumulative_mean_buffer[tau_index - 1]
                y1 = cumulative_mean_buffer[tau_index]
                y2 = cumulative_mean_buffer[tau_index + 1]
                denominator = 2 * y1 - y2 - y0
                if denominator == 0:
                    interpolated_tau = tau_index
                else:
                    interpolated_tau = tau_index + 0.5 * (y2 - y0) / denominator
                f0 = sample_rate / interpolated_tau
                break
        return f0


    def save_features_to_csv(self):
        """
        将 features 写入到传入的已打开文件对象 file_obj 中。
        file_obj 应该是支持写操作的类文件对象，比如 open(...) 返回的对象。
        """
        if not self.features:
            raise ValueError("features 为空，无法保存")

        print("当前 features 中各项长度和形状：")
        for k, v in self.features.items():
            arr = np.array(v)
            print(f"特征名: {k}, 长度: {len(arr.flatten())}, 形状: {arr.shape}")

        processed_features = {}
        for k, v in self.features.items():
            arr = np.array(v)
            if arr.ndim > 1:
                arr = arr.flatten()
            processed_features[k] = arr

        try:
            import pandas as pd
            df = pd.DataFrame(processed_features)
        except Exception as e:
            raise ValueError(f"转换 features 为 DataFrame 出错: {e}")

        return df



import numpy as np
import librosa
from audio_segment import AudioSegment  # 你的类文件名和类名，确认正确

def extract_random_20ms_sample(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    segment_length = int(sr * 0.02)  # 20ms对应采样点数
    if len(y) <= segment_length:
        raise ValueError("音频太短，无法提取20ms样本")
    start_idx = np.random.randint(0, len(y) - segment_length)
    segment = y[start_idx : start_idx + segment_length]
    return segment, sr

def main(audio_file, output_csv):
    # 提取随机20ms样本
    sample, sample_rate = extract_random_20ms_sample(audio_file)

    # 创建特征提取器对象
    extractor = AudioSegment()
    extractor.samples = sample
    extractor.sample_rate = sample_rate

    # 计算各特征并保存到features字典
    extractor.set_FFT()
    extractor.set_crepe()
    extractor.set_Pyin()

    df = extractor.save_features_to_csv()  # 返回 DataFrame
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    audio_file = "1.wav"     # 你的音频路径
    output_csv = "features_output1.csv"  # 你想保存的CSV路径
    main(audio_file, output_csv)