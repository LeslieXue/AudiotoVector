import sys
sys.path.append('util')
import os, glob, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from Transformer import TransformerEncoderModel
from util import point_to_index, index_to_point


# Dataset 类
class ArrayDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# 规范列名：去除 BOM/零宽空格/不间断空格，并 strip
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = []
    for c in df.columns:
        s = str(c)
        s = s.replace("\ufeff", "")   # BOM
        s = s.replace("\u200b", "")   # zero-width space
        s = s.replace("\xa0", " ")    # non-breaking space
        s = s.strip()
        cleaned.append(s)
    df.columns = cleaned
    return df

# 工具函数：解析 JSON 向量列
def parse_vector_column(series, expected_len=11):
    def parse_string(s):
        try:
            val = json.loads(s)
            arr = np.array(val)
            if arr.shape == (expected_len,):
                return arr
        except Exception:
            pass
        try:
            s_clean = s.strip().strip('"').strip("'")
            if not s_clean.startswith("["):
                s_clean = "[" + s_clean
            if not s_clean.endswith("]"):
                s_clean = s_clean + "]"
            arr = np.array(json.loads(s_clean))
            if arr.shape == (expected_len,):
                return arr
            else:
                print(f"⚠️ 尺寸不符: 期望 {expected_len}, 实际 {arr.shape} ← 内容: {s}")
        except Exception as e:
            print(f"❌ 无法解析: {s}，错误: {e}")
        return np.array([])
    return series.apply(parse_string)

def load_samples(data_folder, allowed_labels=None, point=61):
    all_csv_paths = glob.glob(os.path.join(data_folder, "*.csv"))
    if not all_csv_paths:
        from audio_segment_DataGen import get_segment_vectors
        print(f"⚠️ No CSV files found in {data_folder}, generating with get_segment_vectors...")
        get_segment_vectors(input_folder=data_folder)
        all_csv_paths = glob.glob(os.path.join(data_folder, "*.csv"))
    all_records = []
    for path in all_csv_paths:
        df = pd.read_csv(path, encoding="utf-8-sig", engine="python", on_bad_lines="skip")
        df = _normalize_columns(df)
        records = df.to_dict(orient="records")
        all_records.extend(records)

    if allowed_labels is not None:
        all_records = [r for r in all_records if r["shift_fre_point"] in allowed_labels]

    df = pd.DataFrame(all_records)
    for col in ["period_wave_abs_diff", 
                "normalized_energy_vector_sum",
                "normalized_energy_vector_squared",
                "cosine_similarity_vector"]:
        if col in df.columns:
            df[col] = parse_vector_column(df[col].astype(str), expected_len=point)

    samples = []
    for _, row in df.iterrows():
        vectors = [row["period_wave_abs_diff"],
                   row["normalized_energy_vector_sum"],
                   row["normalized_energy_vector_squared"],
                   row["cosine_similarity_vector"]]
        if all(v.shape == (point,) for v in vectors):
            features = np.stack(vectors, axis=1)
            label = int(row["shift_fre_point"])
            samples.append((features, label))
    return samples

#def load_samples_from_paths(file_paths, allowed_labels=None, point=61):
    all_records = []
    for path in file_paths:
        df = pd.read_csv(path, encoding="utf-8-sig", engine="python", on_bad_lines="skip")
        df = _normalize_columns(df)
        records = df.to_dict(orient="records")
        all_records.extend(records)

    if allowed_labels is not None:
        all_records = [r for r in all_records if r["shift_fre_point"] in allowed_labels]

    df = pd.DataFrame(all_records)
    for col in ["period_wave_abs_diff", 
                "normalized_energy_vector_sum",
                "normalized_energy_vector_squared",
                "cosine_similarity_vector"]:
        if col in df.columns:
            df[col] = parse_vector_column(df[col].astype(str), expected_len=point)

    samples = []
    for _, row in df.iterrows():
        vectors = [row["period_wave_abs_diff"],
                   row["normalized_energy_vector_sum"],
                   row["normalized_energy_vector_squared"],
                   row["cosine_similarity_vector"]]
        if all(v.shape == (point,) for v in vectors):
            features = np.stack(vectors, axis=1)
            label = int(row["shift_fre_point"])
            samples.append((features, label))
    return samples

def load_samples_from_paths(file_paths, allowed_labels=None, point=61):
    all_records = []
    for path in file_paths:
        df = pd.read_csv(path)
        df = _normalize_columns(df)
        print(f"📄 正在处理文件: {path}，总行数: {len(df)}")
        # 检查向量列是否为合法 JSON 格式
        for col in ["period_wave_abs_diff", 
                    "normalized_energy_vector_sum",
                    "normalized_energy_vector_squared",
                    "cosine_similarity_vector"]:
            invalid_json_count = 0
            for i, val in enumerate(df[col].astype(str)):
                try:
                    arr = json.loads(val)
                    if not isinstance(arr, list):
                        raise ValueError("不是列表")
                except Exception as e:
                    print(f"❌ 非法 JSON: {col} 第 {i} 行 内容: {val}，错误: {e}")
                    invalid_json_count += 1
            print(f"🔍 {col} 列非法 JSON 数量: {invalid_json_count}")
        records = df.to_dict(orient="records")
        all_records.extend(records)

    if allowed_labels is not None:
        before = len(all_records)
        all_records = [r for r in all_records if r.get("shift_fre_point") in allowed_labels]
        after = len(all_records)
        print(f"✅ 筛选 shift_fre_point 成功: {after}/{before} 条记录保留")

    df = pd.DataFrame(all_records)

    for col in ["period_wave_abs_diff", 
                "normalized_energy_vector_sum",
                "normalized_energy_vector_squared",
                "cosine_similarity_vector"]:
        if col not in df.columns:
            print(f"❌ 缺少列: {col}")
            continue
        df[col] = parse_vector_column(df[col], expected_len=point)

    samples = []
    for _, row in df.iterrows():
        vectors = [row["period_wave_abs_diff"],
                   row["normalized_energy_vector_sum"],
                   row["normalized_energy_vector_squared"],
                   row["cosine_similarity_vector"]]
        print("vectors = ", [v.shape for v in vectors], "|", [v.tolist() for v in vectors])
        if all(v.shape == (point,) for v in vectors):
            label = int(row["shift_fre_point"])
            features = np.stack(vectors, axis=1)
            samples.append((features, label))
        else:
            print(f"⚠️ 向量维度不匹配: {[v.shape for v in vectors]}")

    print(f"✅ 有效样本数: {len(samples)}")
    return samples

def split_data(samples, train_ratio=0.7, val_ratio=0.1):
    random.shuffle(samples)
    n = len(samples)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    return train_samples, val_samples, test_samples

# -------------------- 训练函数 --------------------
def train(train_samples, val_samples, model_path="f1_model.pt", max_samples=None, nc = 61):
    if max_samples is not None and max_samples < len(train_samples):
        train_samples = random.sample(train_samples, max_samples)

    train_loader = DataLoader(ArrayDataset(train_samples), batch_size=32, shuffle=True)
    val_loader = DataLoader(ArrayDataset(val_samples), batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerEncoderModel(input_dim=4, embed_dim=32, num_heads=4, num_layers=2, num_classes= nc, dropout=0.1)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(10):
        model.train()
        total_loss = 0
        correct_train, total_train = 0, 0
        for x, y in train_loader:
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("❌ 输入 x 存在 NaN 或 Inf")
            if torch.isnan(y).any() or torch.isinf(y).any():
                print("❌ 标签 y 存在 NaN 或 Inf")
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            if torch.isnan(out).any() or torch.isinf(out).any():
                print("❌ 模型输出存在 NaN 或 Inf")
                print(f"→ 输入 x = {x[0]}")
                continue

            loss = criterion(out, y)
            if torch.isnan(loss) or torch.isinf(loss):
                print("❌ 损失函数为 NaN 或 Inf")
                print(f"→ 标签 y = {y}")
                print(f"→ 模型输出 out = {out}")
                continue

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = torch.argmax(out, dim=1)
            correct_train += (pred == y).sum().item()
            total_train += y.size(0)

        train_acc = correct_train / total_train if total_train > 0 else 0

        # 验证
        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                out_val = model(x_val)
                pred_val = torch.argmax(out_val, dim=1)
                correct_val += (pred_val == y_val).sum().item()
                total_val += y_val.size(0)
        val_acc = correct_val / total_val if total_val > 0 else 0

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"✅ 模型已保存到 {model_path}")

# -------------------- 测试函数 --------------------
def test(test_samples, model_path="f1_model.pt", nc = 61):
    test_loader = DataLoader(ArrayDataset(test_samples), batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerEncoderModel(input_dim=4, embed_dim=32, num_heads=4, num_layers=2, num_classes= nc, dropout=0.1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    correct, total = 0, 0
    all_true, all_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = torch.argmax(out, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_true.extend(y.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())

    acc = correct / total if total > 0 else 0
    print(f"\n✅ Test Accuracy: {acc:.4f}")

    # 混淆矩阵
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    unique_labels = sorted(set(all_true) | set(all_pred))
    label_names = [str(l) for l in unique_labels]
    label2idx = {l: i for i, l in enumerate(unique_labels)}

    conf_mat = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
    for t, p in zip(all_true, all_pred):
        conf_mat[label2idx[p], label2idx[t]] += 1

    df_count = pd.DataFrame(conf_mat, 
                            index=[f'Pred_{l}' for l in label_names], 
                            columns=[f'True_{l}' for l in label_names])
    print("\n📊 数量统计表：")
    print(df_count)

    col_sums = conf_mat.sum(axis=0, keepdims=True)
    df_ratio = df_count / col_sums
    print("\n📈 比例统计表：")
    print(df_ratio.round(3))

    print(f"\n📦 测试样本总数: {len(all_true)}")


'''
if __name__ == "__main__":
    
    data_folder = "features_normal_key_3shift_11point"
    allowed_labels = [5, 12, 24]
    allowed_label_indices = [point_to_index(p) for p in allowed_labels]
    point = 11
    model_path = "f3_model.pt"
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

    # 限制样本数量为最多 50000 条
    max_train_samples = 50000
    if len(train_samples) > max_train_samples:
        train_samples = random.sample(train_samples, max_train_samples)

    # 按比例划分训练集、验证集、测试集
    train_samples, val_samples, _ = split_data(train_samples, train_ratio=0.7, val_ratio=0.1)

    print("🚀 加载测试样本...")
    test_samples = load_samples_from_paths(selected_test_files, allowed_labels=allowed_labels, point=point)
    print(f"测试样本数: {len(test_samples)}")
    
    train_labels = [y for _, y in train_samples]
    print(f"训练集中实际使用的标签种类: {sorted(set(train_labels))}")

    train(train_samples=train_samples, val_samples=val_samples, model_path=model_path, nc=11)
    test(test_samples=test_samples, model_path=model_path, nc=11)
    '''