import numpy as np
from scipy.signal import hilbert

def extract_features(fault_type: str, data_dict: dict, fs: float = 10000.0, bearing_params: dict = None) -> dict:
    """
    接收多通道数据字典，根据故障类型计算对应的特征指标。
    """
    all_signals = list(data_dict.values())
    
    # [修复隐蔽漏洞B]：防崩溃保护，如果传感器数据为空，直接返回空特征
    if not all_signals or len(all_signals) == 0 or len(all_signals[0]) == 0:
        return {}

    rms_values = [np.sqrt(np.mean(sig**2)) for sig in all_signals]
    max_rms = max(rms_values) if rms_values else 0.0

    features = {}
    
    if fault_type == 'motor_fault':
        baseline = 0.1
        ratio = max_rms / baseline if baseline > 0 else 1.0
        confidence = min(1.0, max(0.0, (ratio - 1) * 0.2))
        features = {"rms_ratio": round(ratio, 2), "confidence": round(confidence, 2)}
        
    elif fault_type == 'bearing_fault':
        baseline = 0.1
        ratio = max_rms / baseline if baseline > 0 else 1.0
        
        params = bearing_params or {}
        rpm = params.get('rpm', 1500.0)
        n_rollers = params.get('n_rollers', 8)
        d_roller = params.get('d_roller', 10.0)
        D_pitch = params.get('D_pitch', 50.0)
        beta = params.get('beta', 0.0)
        
        fr = rpm / 60.0       
        cos_beta = np.cos(beta)
        ratio_d_D = d_roller / D_pitch
        
        location_dict = {
            "inner": 0.5 * n_rollers * fr * (1 + ratio_d_D * cos_beta),
            "outer": 0.5 * n_rollers * fr * (1 - ratio_d_D * cos_beta),
            "cage": 0.5 * fr * (1 - ratio_d_D * cos_beta),
            "ball": (D_pitch / (2 * d_roller)) * fr * (1 - (ratio_d_D * cos_beta) ** 2)
        }
        
        best_sig = all_signals[np.argmax(rms_values)]
        best_sig_centered = best_sig - np.mean(best_sig) 
        envelope = np.abs(hilbert(best_sig_centered))
        envelope -= np.mean(envelope)
        
        spectrum = np.abs(np.fft.rfft(envelope))
        freqs = np.fft.rfftfreq(len(envelope), 1/fs)
        
        sub_confidences = {}
        window = 2
        for key, target_f in location_dict.items():
            if len(freqs) < 2:
                sub_confidences[key] = 0.0
                continue
            idx = np.argmin(np.abs(freqs - target_f))
            amp = spectrum[idx]
            start, end = max(0, idx - window), min(len(spectrum), idx + window + 1)
            local_region = np.concatenate((spectrum[start:idx], spectrum[idx+1:end])) if end > start else []
            local_mean = np.mean(local_region) if len(local_region) > 0 else 1e-6
            
            snr = amp / local_mean
            conf = min(1.0, max(0.0, (snr - 3.0) / 5.0))
            sub_confidences[key] = round(conf, 2)
            
        features = {
            "rms_ratio": round(ratio, 2),
            "inner": sub_confidences["inner"], "outer": sub_confidences["outer"],
            "cage": sub_confidences["cage"], "ball": sub_confidences["ball"]
        }
        
    elif fault_type == 'bolt_loose':
        baseline = 0.1
        features = {"rms_ratio": round(max_rms / baseline, 2) if baseline > 0 else 1.0}
        
    elif fault_type in ['wire_rope', 'rope_fault']:
        baseline = 0.05
        features = {"rms_baseline_ratio": round(max_rms / baseline, 2) if baseline > 0 else 1.0}
        
    elif fault_type == 'guide_rail':
        features = {"wear_percent": round(min(100.0, max_rms * 100), 2)}
        
    elif fault_type == 'car':
        # [完全适配 elevator_car.py 逻辑] 提取Z轴和X/Y轴各自的报警状态
        has_frame_vib = False
        has_smooth_issue = False
        
        directions_data = {}
        # 假设通道顺序即为 X, Y, Z
        for i, (ch_name, sig) in enumerate(data_dict.items()):
            axis = "X" if i == 0 else ("Y" if i == 1 else "Z")
            if len(sig) == 0: continue
            
            rms = np.sqrt(np.mean(sig**2)) + 1e-6
            peak, mean_abs = np.max(np.abs(sig)), np.mean(np.abs(sig)) + 1e-6
            smr = (np.mean(np.sqrt(np.abs(sig))))**2 + 1e-6
            
            directions_data[axis] = {
                "pf": peak / rms,
                "imp": peak / mean_abs,
                "mar": peak / smr,
                "rms": rms
            }
            
        # Z轴逻辑 (轿架振动) - 这里简化模拟了你的新逻辑
        z_data = directions_data.get("Z", {})
        if z_data:
            exceed = sum([1 for k in ["pf", "imp", "mar"] if z_data.get(k, 0) > 5.0]) # 阈值假设
            if z_data.get("rms", 0) > 4.0 and exceed >= 2:
                has_frame_vib = True
                
        # X/Y轴逻辑 (平稳度异常)
        for axis in ["X", "Y"]:
            a_data = directions_data.get(axis, {})
            if a_data:
                exceed = sum([1 for k in ["pf", "imp", "mar"] if a_data.get(k, 0) > 5.0])
                if a_data.get("rms", 0) > 4.0 and exceed >= 2:
                    has_smooth_issue = True
                    break # 只要一个轴超标即认为平稳度异常
                    
        features = {
            "has_frame_vibration": has_frame_vib,
            "has_smoothness_issue": has_smooth_issue
        }
        
    return features