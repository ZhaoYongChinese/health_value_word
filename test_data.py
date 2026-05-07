import os
import numpy as np
import pandas as pd

def create_test_csv(file_path, sensor_id, fault_type, fs, channels, data_matrix):
    """
    生成符合诊断系统要求的 CSV 文件
    第一行：传感器编号
    第二行：故障类型
    第三行：采样率
    第四行：通道标志 (none 或 X,Y,Z 等)
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"{sensor_id}\n")
        f.write(f"{fault_type}\n")
        f.write(f"{fs}\n")
        f.write(f"{','.join(channels)}\n")
    
    # 追加写入具体的数据矩阵
    df = pd.DataFrame(data_matrix)
    df.to_csv(file_path, mode='a', header=False, index=False)
    print(f"[+] 已生成测试文件: {file_path} (形态: {data_matrix.shape})")

def main():
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # ==========================================
    # 测试例 1：曳引机电机（单通道，正常信号）
    # ==========================================
    fs1 = 16000
    t1 = np.linspace(0, 1, fs1, endpoint=False) # 1秒的数据
    # 模拟 50Hz 工频 + 一些轻微的随机噪声
    signal1 = 0.05 * np.sin(2 * np.pi * 50 * t1) + np.random.normal(0, 0.01, fs1)
    
    create_test_csv(
        os.path.join(data_dir, "test_motor_81025.csv"),
        sensor_id="81025",
        fault_type="motor_fault",
        fs=fs1,
        channels=["none"],
        data_matrix=signal1.reshape(-1, 1)
    )

    # ==========================================
    # 测试例 2：钢丝绳（单通道，注入严重高振幅异常）
    # ==========================================
    # 目的是让钢丝绳的 RMS 超标，触发系统抓取它作为"最差项"画图
    fs2 = 8000
    t2 = np.linspace(0, 1, fs2, endpoint=False)
    # 注入振幅高达 0.8 的冲击信号（模拟钢丝绳严重断丝或打滑）
    signal2 = 0.8 * np.sin(2 * np.pi * 10 * t2) + np.random.normal(0, 0.2, fs2)
    
    create_test_csv(
        os.path.join(data_dir, "test_wirerope_81026.csv"),
        sensor_id="81026",
        fault_type="wire_rope",
        fs=fs2,
        channels=["none"],
        data_matrix=signal2.reshape(-1, 1)
    )

    # ==========================================
    # 测试例 3：轿厢（多通道 X/Y/Z，中等波动）
    # ==========================================
    fs3 = 2000
    t3 = np.linspace(0, 1, fs3, endpoint=False)
    # 模拟三个方向的振动信号
    sig_x = 0.02 * np.sin(2 * np.pi * 2 * t3) + np.random.normal(0, 0.01, fs3)
    sig_y = 0.03 * np.sin(2 * np.pi * 3 * t3) + np.random.normal(0, 0.015, fs3)
    sig_z = 0.05 * np.sin(2 * np.pi * 5 * t3) + np.random.normal(0, 0.02, fs3)
    
    # 组合成 3 列
    signal3 = np.column_stack((sig_x, sig_y, sig_z))
    
    create_test_csv(
        os.path.join(data_dir, "test_car_81027.csv"),
        sensor_id="81027",
        fault_type="car",
        fs=fs3,
        channels=["X", "Y", "Z"],
        data_matrix=signal3
    )

    print("\n所有模拟数据生成完毕！请运行 `python main.py` 进行自动诊断测试。")

if __name__ == "__main__":
    main()