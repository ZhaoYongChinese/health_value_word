import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SignalVisualizer:
    def __init__(self, output_dir="output_images"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _add_top_peaks(self, ax, xf, yf):
        """核心寻峰算法：剔除5Hz以下直流及低频，标注Top3极值点"""
        valid_idx = np.where(xf > 5.0)[0]
        if len(valid_idx) == 0: return

        xf_valid, yf_valid = xf[valid_idx], yf[valid_idx]

        # 寻找局部峰值，高度需超过区域均值的1.5倍，横向间距限制防止同一波峰重复抓取
        peaks, properties = find_peaks(yf_valid, height=np.mean(yf_valid)*1.5, distance=5)
        
        if len(peaks) == 0:
            # 兜底：若无明显独立峰值，强行找三个最高点
            peaks_idx = np.argsort(yf_valid)[-3:][::-1]
        else:
            # 依高度排序取前三
            top_indices = np.argsort(properties['peak_heights'])[-3:][::-1]
            peaks_idx = peaks[top_indices]

        colors = ['#d62728', '#2ca02c', '#9467bd'] # 红、绿、紫
        labels = []
        for j, p_idx in enumerate(peaks_idx):
            real_idx = valid_idx[p_idx]
            f_val, a_val = xf[real_idx], yf[real_idx]
            # 图上打点
            ax.scatter(f_val, a_val, color=colors[j % 3], zorder=5, s=40, edgecolors='white', linewidth=1)
            labels.append(f"Top {j+1}: {f_val:.1f}Hz (Amp: {a_val:.3f})")
            
        if labels:
            # 右上角绘制数据标签盒 (半透明防遮挡)
            ax.text(0.98, 0.95, "\n".join(labels), transform=ax.transAxes,
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='#cccccc'))

    def plot_worst_case(self, worst_data: dict, display_metrics: list, detail_scores: dict, prefix: str = ""):
        signal_data = worst_data['signal_data']
        fs = worst_data['fs']
        channels = list(signal_data.keys())
        num_ch = len(channels)
        
        metric_texts = [f"{metric}: {detail_scores[metric]}" for metric in display_metrics if metric in detail_scores]
        text_str = "\n".join(metric_texts) if metric_texts else "无额外指标"
        paths = {}
        line_colors = ['#1f77b4', '#ff7f0e', '#8c564b']

        # [修改点] 引入了前缀拼接机制，防止在多报警并发下生成的图像发生名称重写与覆盖
        file_prefix = f"{prefix}_" if prefix else ""

        # 1. 原始信号时域波形
        fig, axes = plt.subplots(num_ch, 1, figsize=(10, 2.8 * num_ch), squeeze=False)
        for i, ch in enumerate(channels):
            ax = axes[i, 0]
            sig = signal_data[ch]
            t = np.arange(len(sig)) / fs
            ax.plot(t, sig, color=line_colors[i % 3], linewidth=0.5, label=f"Channel {ch}")
            ax.set_ylabel('幅值')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            if i == 0:
                ax.set_title("原始信号时域波形")
                ax.text(0.01, 0.95, text_str, transform=ax.transAxes, ha='left', va='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[-1, 0].set_xlabel('时间 (s)')
        plt.tight_layout()
        p1 = os.path.join(self.output_dir, f"{file_prefix}time_domain.png")
        plt.savefig(p1, dpi=150); paths['time'] = p1; plt.close()

        # 2. 原始信号频谱 (FFT)
        fig, axes = plt.subplots(num_ch, 1, figsize=(10, 2.8 * num_ch), squeeze=False)
        for i, ch in enumerate(channels):
            ax = axes[i, 0]
            sig = signal_data[ch] - np.mean(signal_data[ch]) # 去基线
            N = len(sig)
            yf = 2.0/N * np.abs(np.fft.fft(sig)[:N//2])
            xf = np.linspace(0.0, fs/2.0, N//2)
            ax.plot(xf, yf, color=line_colors[i % 3], linewidth=0.8)
            ax.set_ylabel('幅值')
            ax.grid(True, alpha=0.3)
            if i == 0: ax.set_title("原始信号频谱图 (FFT)")
            self._add_top_peaks(ax, xf, yf) # 智能寻峰标注
        axes[-1, 0].set_xlabel('频率 (Hz)')
        plt.tight_layout()
        p2 = os.path.join(self.output_dir, f"{file_prefix}spectrum.png")
        plt.savefig(p2, dpi=150); paths['spectrum'] = p2; plt.close()

        # 3. 包络解调谱
        fig, axes = plt.subplots(num_ch, 1, figsize=(10, 2.8 * num_ch), squeeze=False)
        for i, ch in enumerate(channels):
            ax = axes[i, 0]
            sig = signal_data[ch] - np.mean(signal_data[ch])
            envelope = np.abs(hilbert(sig))
            envelope -= np.mean(envelope)
            N = len(envelope)
            yf = 2.0/N * np.abs(np.fft.fft(envelope)[:N//2])
            xf = np.linspace(0.0, fs/2.0, N//2)
            ax.plot(xf, yf, color=line_colors[i % 3], linewidth=0.8)
            ax.set_ylabel('幅值')
            ax.set_xlim(0, 500) # 包络图聚焦低频故障
            ax.grid(True, alpha=0.3)
            if i == 0: ax.set_title("包络解调谱 (Hilbert)")
            
            # 寻峰时约束在绘制的 xlim 范围内
            mask = xf <= 500
            self._add_top_peaks(ax, xf[mask], yf[mask])
            
        axes[-1, 0].set_xlabel('频率 (Hz)')
        plt.tight_layout()
        p3 = os.path.join(self.output_dir, f"{file_prefix}envelope.png")
        plt.savefig(p3, dpi=150); paths['envelope'] = p3; plt.close()

        return paths