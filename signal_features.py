import numpy as np

def extract_features(fault_type: str, data_dict: dict, fs: float = 10000.0) -> dict:
    """
    接收多通道数据字典，根据故障类型计算对应的特征指标。
    已包含基于物理原理的轴承包络解调谱计算。
    """
    all_signals = list(data_dict.values())
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
        
        # --- 真实的轴承包络谱分析 ---
        # 1. 默认轴承物理参数
        rpm = 1500.0          
        fr = rpm / 60.0       
        n_rollers = 8         
        d_roller = 10.0       
        D_pitch = 50.0        
        beta = 0.0            
        
        # 2. 计算特征频率
        cos_beta = np.cos(beta)
        ratio_d_D = d_roller / D_pitch
        f_inner = 0.5 * n_rollers * fr * (1 + ratio_d_D * cos_beta)
        f_outer = 0.5 * n_rollers * fr * (1 - ratio_d_D * cos_beta)
        f_cage = 0.5 * fr * (1 - ratio_d_D * cos_beta)
        f_ball = (D_pitch / (2 * d_roller)) * fr * (1 - (ratio_d_D * cos_beta) ** 2)
        location_dict = {"inner": f_inner, "outer": f_outer, "cage": f_cage, "ball": f_ball}
        
        # 3. 对振动最大的通道进行包络解调
        best_sig = all_signals[np.argmax(rms_values)] if all_signals else np.array([0])
        envelope = np.abs(best_sig)  
        spectrum = np.abs(np.fft.rfft(envelope))
        freqs = np.fft.rfftfreq(len(envelope), 1/fs)
        
        # 4. 计算各部位特征频率处的信噪比(SNR)并映射为置信度
        sub_confidences = {}
        window = 2
        for key, target_f in location_dict.items():
            if len(freqs) < 2:
                sub_confidences[key] = 0.0
                continue
            idx = np.argmin(np.abs(freqs - target_f))
            amp = spectrum[idx]
            
            start = max(0, idx - window)
            end = min(len(spectrum), idx + window + 1)
            local_region = np.concatenate((spectrum[start:idx], spectrum[idx+1:end])) if end > start else []
            local_mean = np.mean(local_region) if len(local_region) > 0 else 1e-6
            
            snr = amp / local_mean
            # SNR <= 3 置信度为0；SNR >= 8 置信度为1
            conf = min(1.0, max(0.0, (snr - 3.0) / 5.0))
            sub_confidences[key] = round(conf, 2)
            
        features = {
            "rms_ratio": round(ratio, 2),
            "inner": sub_confidences["inner"],
            "outer": sub_confidences["outer"],
            "cage": sub_confidences["cage"],
            "ball": sub_confidences["ball"]
        }
        
    elif fault_type == 'bolt_loose':
        baseline = 0.1
        features = {"rms_ratio": round(max_rms / baseline, 2) if baseline > 0 else 1.0}
        
    elif fault_type in ['wire_rope', 'rope_fault']:
        baseline = 0.05
        features = {"rms_baseline_ratio": round(max_rms / baseline, 2) if baseline > 0 else 1.0}
        
    elif fault_type == 'guide_rail':
        wear_percent = min(100.0, max_rms * 100) 
        features = {"wear_percent": round(wear_percent, 2)}
        
    elif fault_type == 'car':
        directions = []
        for ch_name, sig in data_dict.items():
            rms = np.sqrt(np.mean(sig**2)) + 1e-6
            peak = np.max(np.abs(sig))
            mean_abs = np.mean(np.abs(sig)) + 1e-6
            smr = (np.mean(np.sqrt(np.abs(sig))))**2 + 1e-6
            directions.append({
                "crest_factor_ratio": round((peak / rms) / 2.0, 2),
                "impulse_factor_ratio": round((peak / mean_abs) / 2.0, 2),
                "margin_factor_ratio": round((peak / smr) / 2.0, 2)
            })
        features = {"directions": directions}
        
    return features