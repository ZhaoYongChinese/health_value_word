import numpy as np
import pandas as pd

# ========== 参数设置 ==========
fs = 32000          # 采样频率
T = 2.0             # 信号时长（秒）
fault_freq = 104.16 # 模拟内圈故障频率
impact_amp = 5.0    # 冲击幅度
noise_std = 0.2     # 背景噪声标准差

# ========== 生成信号 ==========
t = np.arange(0, T, 1/fs)
N = len(t)
signal = np.random.normal(0, noise_std, N)

# 冲击参数
impact_time = 1 / fault_freq
jitter = 0.01 * impact_time
resonance_freq = 3000
decay_factor = 200
duration = 5 / decay_factor           # 冲击持续时间（秒）
impulse_samples = int(duration * fs)  # 冲击样本数

num_impacts = int(T * fault_freq) + 1
for i in range(num_impacts):
    t0 = i * impact_time + np.random.uniform(-jitter, jitter)
    # 计算冲击起始样本索引（可能为负或超过 N）
    start_idx = int(t0 * fs)
    end_idx = start_idx + impulse_samples
    
    # 跳过完全不重叠的冲击
    if end_idx <= 0 or start_idx >= N:
        continue
    
    # 计算与信号有效重叠的部分
    sig_start = max(start_idx, 0)
    sig_end = min(end_idx, N)
    length = sig_end - sig_start
    if length <= 0:
        continue

    # 生成完整的指数衰减正弦波
    t_local = np.arange(impulse_samples) / fs
    sign = np.random.choice([-1, 1])
    full_impulse = sign * impact_amp * np.sin(2 * np.pi * resonance_freq * t_local) * np.exp(-decay_factor * t_local)

    # 截取对应部分叠加到信号
    offset = sig_start - start_idx
    signal[sig_start:sig_end] += full_impulse[offset:offset+length]

# ========== 保存 CSV ==========
df = pd.DataFrame({'time': t, 'amplitude': signal})
df.to_csv('wave19_32KHz.csv', header=False, index=False)
print("模拟振动数据已保存为 wave19_32KHz.csv")