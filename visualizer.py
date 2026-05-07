import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SignalVisualizer:
    def __init__(self, output_dir="output_images"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def plot_worst_case(self, worst_data: dict, display_metrics: list, detail_scores: dict):
        """
        根据最差项的数据，生成时域、频域、包络三张图，并在图上打印配置的关键指标
        """
        signal_data = worst_data['signal_data'] # dict: {'X': array, 'Y': array, 'Z': array}
        fs = worst_data['fs']
        channels = list(signal_data.keys())
        num_ch = len(channels)
        
        # 组装要在图上显示的文字
        metric_texts = []
        for metric in display_metrics:
            if metric in detail_scores:
                val = detail_scores[metric]
                metric_texts.append(f"{metric}: {val}")
        text_str = "\n".join(metric_texts) if metric_texts else "无额外指标"

        paths = {}
        
        # 1. 原始信号时域波形
        fig, axes = plt.subplots(num_ch, 1, figsize=(10, 2.5 * num_ch), squeeze=False)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, ch in enumerate(channels):
            ax = axes[i, 0]
            sig = signal_data[ch]
            t = np.arange(len(sig)) / fs
            ax.plot(t, sig, color=colors[i % 3], linewidth=0.5, label=f"Channel {ch}")
            ax.set_ylabel('幅值')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            if i == 0:
                ax.set_title("原始信号时域波形")
                # 在第一张图上打上指标
                ax.text(0.01, 0.95, text_str, transform=ax.transAxes, 
                        ha='left', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[-1, 0].set_xlabel('时间 (s)')
        plt.tight_layout()
        p1 = os.path.join(self.output_dir, "time_domain.png")
        plt.savefig(p1, dpi=150)
        paths['time'] = p1
        plt.close()

        # 2. 原始信号频谱 (FFT)
        fig, axes = plt.subplots(num_ch, 1, figsize=(10, 2.5 * num_ch), squeeze=False)
        for i, ch in enumerate(channels):
            ax = axes[i, 0]
            sig = signal_data[ch]
            N = len(sig)
            yf = 2.0/N * np.abs(np.fft.fft(sig)[:N//2])
            xf = np.linspace(0.0, fs/2.0, N//2)
            ax.plot(xf, yf, color=colors[i % 3], linewidth=0.8)
            ax.set_ylabel('幅值')
            ax.grid(True, alpha=0.3)
            if i == 0: ax.set_title("原始信号频谱图 (FFT)")
        axes[-1, 0].set_xlabel('频率 (Hz)')
        plt.tight_layout()
        p2 = os.path.join(self.output_dir, "spectrum.png")
        plt.savefig(p2, dpi=150)
        paths['spectrum'] = p2
        plt.close()

        # 3. 包络解调谱
        fig, axes = plt.subplots(num_ch, 1, figsize=(10, 2.5 * num_ch), squeeze=False)
        for i, ch in enumerate(channels):
            ax = axes[i, 0]
            sig = signal_data[ch]
            # 简单的希尔伯特包络
            envelope = np.abs(hilbert(sig))
            envelope -= np.mean(envelope)
            N = len(envelope)
            yf = 2.0/N * np.abs(np.fft.fft(envelope)[:N//2])
            xf = np.linspace(0.0, fs/2.0, N//2)
            ax.plot(xf, yf, color=colors[i % 3], linewidth=0.8)
            ax.set_ylabel('幅值')
            ax.set_xlim(0, 500) # 通常包络谱看低频
            ax.grid(True, alpha=0.3)
            if i == 0: ax.set_title("包络解调谱")
        axes[-1, 0].set_xlabel('频率 (Hz)')
        plt.tight_layout()
        p3 = os.path.join(self.output_dir, "envelope.png")
        plt.savefig(p3, dpi=150)
        paths['envelope'] = p3
        plt.close()

        return paths