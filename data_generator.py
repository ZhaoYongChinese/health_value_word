import os
import json
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

class FaultDataGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("电梯数字孪生 - 故障波形仿真生成器")
        self.root.geometry("480x520")
        
        # 定义可选的故障类型及其在代码中的映射
        self.mech_options = {
            "电机异常 (转子不平衡)": {"type": "motor_fault", "short": "电机异常"},
            "轴承磨损 (内圈剥落)": {"type": "bearing_fault", "short": "轴承磨损"},
            "底座固定螺栓松动": {"type": "bolt_loose", "short": "螺栓松动"},
            "钢丝绳打滑/磨损": {"type": "wire_rope", "short": "钢丝绳磨损"},
            "轿厢异常抖动 (导靴偏磨)": {"type": "car", "short": "轿厢抖动"},
            "导轨过度磨损": {"type": "guide_rail", "short": "导轨磨损"}
        }
        
        self.env_options = {
            "底坑/机房水浸": {"key": "water", "short": "水浸"},
            "机房高温 (>65℃)": {"key": "temperature", "short": "高温"},
            "设备异常位移 (>15cm)": {"key": "displacement", "short": "位移"}
        }
        
        self.mech_vars = {name: tk.BooleanVar() for name in self.mech_options}
        self.env_vars = {name: tk.BooleanVar() for name in self.env_options}
        
        self.fs = 10000.0 # 默认采样率 10kHz
        self.duration = 1.0 # 数据时长 1秒 (10000个点)
        
        self._build_gui()

    def _build_gui(self):
        # 机械故障区
        lf_mech = ttk.LabelFrame(self.root, text="机械本体故障注入 (将生成特定的异常高频振动)")
        lf_mech.pack(padx=20, pady=10, fill="x")
        for name, var in self.mech_vars.items():
            ttk.Checkbutton(lf_mech, text=name, variable=var).pack(anchor="w", padx=10, pady=5)
            
        # 环境故障区
        lf_env = ttk.LabelFrame(self.root, text="环境与电气故障注入 (将修改 env_data.json)")
        lf_env.pack(padx=20, pady=10, fill="x")
        for name, var in self.env_vars.items():
            ttk.Checkbutton(lf_env, text=name, variable=var).pack(anchor="w", padx=10, pady=5)
            
        # 操作区
        frame_action = ttk.Frame(self.root)
        frame_action.pack(pady=20)
        
        ttk.Button(frame_action, text="清除所有选择", command=self._clear_all).pack(side="left", padx=10)
        btn_generate = tk.Button(frame_action, text="生成测试数据包", bg="#1f77b4", fg="white", 
                                 font=("SimHei", 12, "bold"), command=self.generate_data)
        btn_generate.pack(side="left", padx=10)

    def _clear_all(self):
        for var in self.mech_vars.values(): var.set(False)
        for var in self.env_vars.values(): var.set(False)

    def _simulate_signal(self, fault_type, is_faulty):
        """核心仿真引擎：基于物理公式生成带有特定特征的伪造波形"""
        N = int(self.fs * self.duration)
        t = np.arange(N) / self.fs
        
        # 基础健康波形：低频运转基波 (25Hz=1500RPM) + 轻微白噪声 (RMS约0.05)
        base_noise = np.random.randn(N) * 0.05
        base_signal = base_noise + np.sin(2 * np.pi * 25 * t) * 0.02
        
        if not is_faulty:
            return base_signal
            
        # 故障注入逻辑
        if fault_type == 'motor_fault':
            # 电机异常：增加基频(25Hz)及其谐波的巨大振幅，RMS拉高
            return base_signal + np.sin(2 * np.pi * 25 * t) * 0.6 + np.sin(2 * np.pi * 50 * t) * 0.2
            
        elif fault_type == 'bearing_fault':
            # 轴承内圈磨损：生成 120Hz (BPFI) 的包络调制信号，载波为 3000Hz 共振区
            carrier = np.sin(2 * np.pi * 3000 * t)
            envelope = np.maximum(0, np.cos(2 * np.pi * 120 * t)) ** 4 # 锐利的冲击波形
            return base_signal + 0.8 * envelope * carrier
            
        elif fault_type == 'bolt_loose':
            # 螺栓松动：随机产生宽频的剧烈冲击脉冲
            impulses = (np.random.rand(N) > 0.995).astype(float) * np.random.randn(N) * 3.0
            return base_signal + impulses
            
        elif fault_type == 'wire_rope':
            # 钢丝绳：整体随机宽频噪音放大 (代表磨损或干摩擦)
            return np.random.randn(N) * 0.4
            
        elif fault_type == 'guide_rail':
            # 导轨：极低频大位移晃动代表磨损导致的不平顺
            return base_signal + np.sin(2 * np.pi * 2 * t) * 0.8
            
        elif fault_type == 'car':
            # 轿厢：注入极少数离散的巨型尖峰，极大拉高波峰因数(Crest Factor)，但不增加总RMS
            car_sig = base_signal.copy()
            spike_indices = np.random.choice(N, 10, replace=False)
            car_sig[spike_indices] = np.random.choice([1, -1], 10) * 8.0 
            return car_sig
            
        return base_signal

    def generate_data(self):
        # 1. 搜集勾选项，组成文件夹名称
        selected_shorts = []
        active_mech_faults = []
        
        for name, var in self.mech_vars.items():
            if var.get():
                selected_shorts.append(self.mech_options[name]["short"])
                active_mech_faults.append(self.mech_options[name]["type"])
                
        env_status = {"temperature": 25.0, "water": 0, "displacement": 0.0, "motor_current": 0, "noise_ratio": 0.5}
        for name, var in self.env_vars.items():
            if var.get():
                selected_shorts.append(self.env_options[name]["short"])
                key = self.env_options[name]["key"]
                if key == "water": env_status["water"] = 1
                elif key == "temperature": env_status["temperature"] = 68.5
                elif key == "displacement": env_status["displacement"] = 18.2

        folder_name = "+".join(selected_shorts) if selected_shorts else "无故障"
        
        # 限制文件夹名称长度
        if len(folder_name) > 50:
            folder_name = "+".join(selected_shorts[:4]) + "等复合故障"

        # 2. 创建输出目录
        output_dir = os.path.join("data", folder_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 3. 必须生成全套传感器的 CSV 供系统读取 (包含健康项和故障项)
        standard_sensors = [
            ("81025", "motor_fault", ["X", "Y", "Z"]),
            ("81025", "bearing_fault", ["X", "Y", "Z"]),
            ("81025", "bolt_loose", ["X", "Y", "Z"]),
            ("81026", "wire_rope", ["Ch1", "Ch2"]),
            ("81027", "car", ["X", "Y", "Z"]),
            ("81028", "guide_rail", ["X", "Y"])
        ]

        try:
            for sensor_id, fault_type, channels in standard_sensors:
                is_faulty = (fault_type in active_mech_faults)
                
                # 生成所有通道的波形
                data_dict = {}
                for ch in channels:
                    data_dict[ch] = self._simulate_signal(fault_type, is_faulty)
                
                # 构建 DataFrame 并按规范保存
                df = pd.DataFrame(data_dict)
                csv_path = os.path.join(output_dir, f"{sensor_id}_{fault_type}.csv")
                
                with open(csv_path, 'w', encoding='utf-8') as f:
                    f.write(f"{sensor_id}, SENSOR_ID\n")
                    f.write(f"{fault_type}, FAULT_TYPE\n")
                    f.write(f"{int(self.fs)}, SAMPLE_RATE_HZ\n")
                    f.write(", ".join(channels) + "\n")
                
                df.to_csv(csv_path, mode='a', index=False, header=False)

            # 4. 生成 env_data.json
            with open(os.path.join(output_dir, "env_data.json"), 'w', encoding='utf-8') as f:
                json.dump(env_status, f, indent=4, ensure_ascii=False)

            messagebox.showinfo("生成成功", f"仿真数据已生成！\n\n文件夹: ./data/{folder_name}\n包含: 6个物理波形CSV文件 + 1个环境JSON\n\n您现在可以运行 main.py 并在GUI中选择该文件夹进行测试了！")

        except Exception as e:
            messagebox.showerror("生成失败", f"发生错误: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaultDataGenerator(root)
    root.mainloop()