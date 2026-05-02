import numpy as np

def read_sensor_data(csv_file):
    """
    读取 CSV 文件，解析元数据和传感器数据。
    文件格式：
        第1行：采样频率，如 "100 Hz"
        第2行：总采样点数，整数
        第3行：采样时间，浮点数（单位秒）
        后续行：每个采样点的数值
    返回:
        sampling_freq (float): 采样频率 (Hz)
        total_points (int): 总采样点数
        sampling_time (float): 采样时间 (s)
        data (np.ndarray): 传感器数据数组
    """
    with open(csv_file, 'r') as f:
        lines = f.readlines()

    if len(lines) < 4:
        raise ValueError("文件行数不足，至少需要4行（3行元数据 + 至少1个数据点）")

    # 解析元数据
    sampling_freq_str = lines[0].strip()
    try:
        sampling_freq = float(sampling_freq_str.split()[0])  # 提取数值部分
    except:
        raise ValueError("采样频率格式错误，应为如 '100 Hz'")

    total_points = int(lines[1].strip())
    sampling_time = float(lines[2].strip())

    # 读取数据
    data = []
    for line in lines[3:]:
        line = line.strip()
        if line:
            try:
                data.append(float(line))
            except ValueError:
                print(f"警告：无法解析行 '{line}'，跳过")
    data = np.array(data)

    # 验证数据长度是否与 total_points 一致
    if len(data) != total_points:
        print(f"警告：实际数据点数 {len(data)} 与声明的总采样点数 {total_points} 不一致")

    return sampling_freq, total_points, sampling_time, data

def vibration_health(data, threshold):
    """
    振动健康度计算：基于均方根（RMS）值
    参数:
        data (np.ndarray): 振动加速度数据（单位：m/s² 或 g）
        threshold (float): 报警阈值（RMS 超过该值则健康度为0）
    返回:
        health (float): 0~100，数值越高越健康
    """
    rms = np.sqrt(np.mean(data**2))
    # 健康度线性衰减，RMS=0时100分，RMS=threshold时0分
    health = 100 * (1 - min(1, rms / threshold))
    print(f"  振动 RMS = {rms:.4f}, 阈值 = {threshold:.4f}, 健康度 = {health:.2f}")
    return health

def temperature_health(data, low, high):
    """
    温度健康度计算：基于均值与正常范围的偏差
    参数:
        data (np.ndarray): 温度数据（单位：℃）
        low (float): 正常工作下限
        high (float): 正常工作上限
    返回:
        health (float): 0~100，数值越高越健康
    """
    mean_temp = np.mean(data)
    if low <= mean_temp <= high:
        health = 100
    else:
        # 超出范围，按偏离程度线性衰减
        if mean_temp < low:
            deviation = low - mean_temp
            max_dev = low  # 假设偏离超过下限值则健康度为0
        else:
            deviation = mean_temp - high
            max_dev = 100 - high if high < 100 else 50  # 简单示例
        health = 100 * max(0, 1 - deviation / max_dev)
        health = min(100, health)
    print(f"  温度均值 = {mean_temp:.2f}℃, 正常范围 = [{low}, {high}], 健康度 = {health:.2f}")
    return health

def current_health(data, rated_current):
    """
    电流健康度计算：基于均方根（RMS）与额定值的比值
    参数:
        data (np.ndarray): 电流数据（单位：A）
        rated_current (float): 额定电流（A）
    返回:
        health (float): 0~100，数值越高越健康
    """
    rms = np.sqrt(np.mean(data**2))
    # 健康度随电流增大而降低，电流达到额定值2倍时健康度为0
    max_ratio = 2.0
    ratio = rms / rated_current
    health = 100 * max(0, 1 - (ratio / max_ratio))
    health = min(100, health)
    print(f"  电流 RMS = {rms:.2f} A, 额定电流 = {rated_current:.2f} A, 健康度 = {health:.2f}")
    return health

def main():
    print("=== 扶梯健康度生成系统 ===")
    print("说明：本程序根据多传感器数据计算综合健康度。")
    print("每个传感器数据需为 CSV 文件，格式：\n"
          "  第1行：采样频率 (例如 100 Hz)\n"
          "  第2行：总采样点数\n"
          "  第3行：采样时间 (s)\n"
          "  后续行：传感器数值\n")

    # 1. 获取传感器数量
    n_sensors = int(input("请输入传感器种类数量: ").strip())

    sensor_health = []  # 存储每个传感器的健康度
    sensor_weights = [] # 存储权重

    for i in range(n_sensors):
        print(f"\n--- 传感器 {i+1} ---")
        file_path = input("CSV 文件路径: ").strip()
        sensor_type = input("传感器类型（振动/温度/电流）: ").strip().lower()

        try:
            # 读取数据
            _, _, _, data = read_sensor_data(file_path)
        except Exception as e:
            print(f"读取文件失败: {e}，跳过该传感器")
            continue

        # 根据类型获取阈值并计算健康度
        if sensor_type == "振动":
            threshold = float(input("请输入振动报警阈值 (RMS, 单位同数据): ").strip())
            health = vibration_health(data, threshold)
        elif sensor_type == "温度":
            low = float(input("请输入温度正常范围下限 (℃): ").strip())
            high = float(input("请输入温度正常范围上限 (℃): ").strip())
            health = temperature_health(data, low, high)
        elif sensor_type == "电流":
            rated_current = float(input("请输入额定电流 (A): ").strip())
            health = current_health(data, rated_current)
        else:
            print("不支持的传感器类型，跳过")
            continue

        # 询问权重（默认为1）
        weight = float(input("请输入该传感器的权重（直接回车则默认为1）: ").strip() or "1")
        sensor_health.append(health)
        sensor_weights.append(weight)

    if not sensor_health:
        print("没有有效的传感器数据，无法计算健康度。")
        return

    # 2. 计算加权平均健康度
    weighted_sum = sum(h * w for h, w in zip(sensor_health, sensor_weights))
    total_weight = sum(sensor_weights)
    overall_health = weighted_sum / total_weight if total_weight > 0 else 0

    print("\n=== 综合健康度 ===")
    print(f"各传感器健康度: {sensor_health}")
    print(f"权重: {sensor_weights}")
    print(f"综合健康度 = {overall_health:.2f} (0~100, 越高越健康)")

    # 3. 可选：保存结果
    save = input("\n是否保存健康度结果到文件？(y/n): ").strip().lower()
    if save == 'y':
        out_file = input("输出文件名（默认 health_result.txt）: ").strip() or "health_result.txt"
        with open(out_file, 'w') as f:
            f.write("传感器健康度:\n")
            for i, (h, w) in enumerate(zip(sensor_health, sensor_weights)):
                f.write(f"  传感器{i+1}: {h:.2f} (权重 {w})\n")
            f.write(f"\n综合健康度: {overall_health:.2f}\n")
        print(f"结果已保存到 {out_file}")

if __name__ == "__main__":
    main()