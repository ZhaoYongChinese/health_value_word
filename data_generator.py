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
        self.root.geometry("500x620")
        
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
        
        self.health_mode = tk.StringVar(value="fault_inject")
        self.mech_vars = {name: tk.BooleanVar() for name in self.mech_options}
        self.env_vars = {name: tk.BooleanVar() for name in self.env_options}
        
        self.fs = 10000.0 
        # [关键修改]：为了满足 2.0s滑窗且需要连续3个触发窗口，必须保证信号长度 >= 6.0s
        self.duration = 8.0 
        
        self.mech_cbs = []
        self.env_cbs = []
        
        self._build_gui()
        self._toggle_state() # 初始化状态

    def _build_gui(self):
        # 模式选择区
        lf_mode = ttk.LabelFrame(self.root, text="第一步：选择数据包生成模式")
        lf_mode.pack(padx=20, pady=10, fill="x")
        
        ttk.Radiobutton(lf_mode, text="高度健康 (全优状态，所有分项得分 > 90)", variable=self.health_mode, value="highly_healthy", command=self._toggle_state).pack(anchor="w", padx=10, pady=5)
        ttk.Radiobutton(lf_mode, text="一般健康 (轻微退化，各项得分 80 ~ 90 之间)", variable=self.health_mode, value="healthy", command=self._toggle_state).pack(anchor="w", padx=10, pady=5)
        ttk.Radiobutton(lf_mode, text="自定义故障注入 (严重异常，触发告警与 H4)", variable=self.health_mode, value="fault_inject", command=self._toggle_state).pack(anchor="w", padx=10, pady=5)

        # 机械故障区
        lf_mech = ttk.LabelFrame(self.root, text="第二步：机械本体故障注入 (仅在自定义模式可用)")
        lf_mech.pack(padx=20, pady=10, fill="x")
        for name, var in self.mech_vars.items():
            cb = ttk.Checkbutton(lf_mech, text=name, variable=var)
            cb.pack(anchor="w", padx=10, pady=2)
            self.mech_cbs.append(cb)
            
        # 环境故障区
        lf_env = ttk.LabelFrame(self.root, text="第二步：环境与电气故障注入 (仅在自定义模式可用)")
        lf_env.pack(padx=20, pady=10, fill="x")
        for name, var in self.env_vars.items():
            cb = ttk.Checkbutton(lf_env, text=name, variable=var)
            cb.pack(anchor="w", padx=10, pady=2)
            self.env_cbs.append(cb)
            
        # 操作区
        frame_action = ttk.Frame(self.root)
        frame_action.pack(pady=15)
        
        ttk.Button(frame_action, text="清除所有选择", command=self._clear_all).pack(side="left", padx=10)
        btn_generate = tk.Button(frame_action, text="生成测试数据包", bg="#1f77b4", fg="white", 
                                 font=("SimHei", 12, "bold"), command=self.generate_data)
        btn_generate.pack(side="left", padx=10)

    def _toggle_state(self):
        """根据选择的模式，启用或禁用底部的复选框"""
        state = "normal" if self.health_mode.get() == "fault_inject" else "disabled"
        for cb in self.mech_cbs + self.env_cbs:
            if state == "disabled":
                # 禁用时顺便取消勾选
                for var in self.mech_vars.values(): var.set(False)
                for var in self.env_vars.values(): var.set(False)
            cb.config(state=state)

    def _clear_all(self):
        for var in self.mech_vars.values(): var.set(False)
        for var in self.env_vars.values(): var.set(False)

    def _simulate_signal(self, fault_type, is_faulty, mode):
        """核心仿真引擎：根据配置的 baseline(1.0) 和惩罚系数精准生成波形"""
        N = int(self.fs * self.duration)
        t = np.arange(N) / self.fs
        
        # --- 模式一：高度健康 ---
        # Baseline为1.0。将RMS控制在0.5左右，永远达不到触发阈值，所有分数保持100分。
        base_noise = np.random.randn(N) * 0.5 
        base_signal = base_noise + np.sin(2 * np.pi * 25 * t) * 0.05
        
        if mode == "highly_healthy":
            return base_signal

        # --- 模式二：一般健康 (模拟轻微退化) ---
        # 需要RMS略大于1.0才能触发扣分机制，但扣分又不能太多
        if mode == "healthy":
            if fault_type in ['motor_fault', 'bearing_fault']:
                # RMS=1.25, 惩罚率40 -> 扣分: (1.25-1.0)*40 = 10分 -> 得分: 90分
                return np.random.randn(N) * 1.25
            elif fault_type == 'car':
                # 同上，加入少量冲击以触发CF因数超标
                car_sig = np.random.randn(N) * 1.25
                spike_idx = np.random.choice(N, 5, replace=False)
                car_sig[spike_idx] = np.random.choice([1, -1], 5) * 5.0 
                return car_sig
            elif fault_type == 'bolt_loose':
                # RMS=1.5, 惩罚率20 -> 扣分: (1.5-1.0)*20 = 10分 -> 得分: 90分
                return np.random.randn(N) * 1.5
            elif fault_type in ['wire_rope', 'rope_fault']:
                # 钢丝绳采用分段计分：2~6区间的得分为 100~70。RMS=3.0 -> 得分 92.5分
                return np.random.randn(N) * 3.0
            elif fault_type == 'guide_rail':
                # 导轨特殊：一旦RMS>1.0触发，磨损率就拉满0分，所以必须强制让导轨低于1.0
                return base_signal
            return base_signal

        # --- 模式三：自定义故障注入 ---
        if not is_faulty:
            return base_signal
            
        # 故障注入逻辑：产生巨大的RMS，彻底击穿底线
        if fault_type == 'motor_fault':
            return base_signal + np.sin(2 * np.pi * 25 * t) * 2.0 + np.sin(2 * np.pi * 50 * t) * 1.5
        elif fault_type == 'bearing_fault':
            carrier = np.sin(2 * np.pi * 3000 * t)
            envelope = np.maximum(0, np.cos(2 * np.pi * 120 * t)) ** 4
            return base_signal + 3.0 * envelope * carrier
        elif fault_type == 'bolt_loose':
            impulses = (np.random.rand(N) > 0.55).astype(float) * np.random.randn(N) * 8.0
            return base_signal + impulses
        elif fault_type == 'wire_rope':
            return np.random.randn(N) * 10.0 # 绝对打滑
        elif fault_type == 'guide_rail':
            return base_signal + np.sin(2 * np.pi * 2 * t) * 2.5
        elif fault_type == 'car':
            car_sig = np.random.randn(N) * 2.5
            spike_indices = np.random.choice(N, 20, replace=False)
            car_sig[spike_indices] = np.random.choice([1, -1], 20) * 15.0 
            return car_sig
            
        return base_signal

    def generate_data(self):
        mode = self.health_mode.get()
        selected_shorts = []
        active_mech_faults = []
        env_status = {"temperature": 0, "water": 0, "displacement": 0, "motor_current": 0, "noise_ratio": 0}

        # 1. 搜集状态配置与决定文件夹名
        if mode == "highly_healthy":
            folder_name = "批次_高度健康_全优运行"
        elif mode == "healthy":
            folder_name = "批次_一般健康_轻微退化"
        else:
            for name, var in self.mech_vars.items():
                if var.get():
                    selected_shorts.append(self.mech_options[name]["short"])
                    active_mech_faults.append(self.mech_options[name]["type"])
            for name, var in self.env_vars.items():
                if var.get():
                    selected_shorts.append(self.env_options[name]["short"])
                    env_status[self.env_options[name]["key"]] = 1
                    
            folder_name = "批次_故障_" + "+".join(selected_shorts) if selected_shorts else "批次_无故障基准"
            if len(folder_name) > 50:
                folder_name = "批次_故障_" + "+".join(selected_shorts[:4]) + "等复合异常"

        # 2. 创建输出目录
        output_dir = os.path.join("data", folder_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 3. 生成全套传感器的 CSV
        standard_sensors = [
            ("81025", "motor_fault", ["Ch1"]),
            ("81025", "bearing_fault", ["Ch1"]),
            ("81025", "bolt_loose", ["Ch1"]),
            ("81026", "wire_rope", ["Ch1"]),
            ("81027", "car", ["X", "Y", "Z"]),
            ("81028", "guide_rail", ["Ch1"])
        ]

        try:
            for sensor_id, fault_type, channels in standard_sensors:
                is_faulty = (fault_type in active_mech_faults)
                
                data_dict = {}
                for ch in channels:
                    data_dict[ch] = self._simulate_signal(fault_type, is_faulty, mode)
                
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

            messagebox.showinfo("生成成功", f"仿真数据已生成！\n\n模式: {mode}\n时长: {self.duration}秒\n文件夹: ./data/{folder_name}\n\n您现在可以运行 main.py 进行测试。")

        except Exception as e:
            messagebox.showerror("生成失败", f"发生错误: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaultDataGenerator(root)
    root.mainloop()