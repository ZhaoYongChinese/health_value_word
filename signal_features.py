import numpy as np

def extract_features(fault_type: str, data_dict: dict) -> dict:
    """
    接收多通道数据字典，根据故障类型计算对应的特征指标。
    返回的字典结构必须严格适配 health_evaluator.py 的输入。
    """
    # 提取所有通道的数据，计算全局或单通道最大值
    all_signals = list(data_dict.values())
    rms_values = [np.sqrt(np.mean(sig**2)) for sig in all_signals]
    max_rms = max(rms_values) if rms_values else 0.0

    features = {}
    
    # 根据具体的故障类型，返回对应的指标字典
    if fault_type in ['motor_fault', 'bearing_fault']:
        # 假设基线 RMS 为 0.1
        baseline = 0.1
        ratio = max_rms / baseline if baseline > 0 else 1.0
        # 简单模拟：RMS越高，置信度越高
        confidence = min(1.0, max(0.0, (ratio - 1) * 0.2))
        features = {
            "rms_ratio": round(ratio, 2),
            "confidence": round(confidence, 2)
        }
        
    elif fault_type == 'bolt_loose':
        baseline = 0.1
        features = {
            "rms_ratio": round(max_rms / baseline, 2) if baseline > 0 else 1.0
        }
        
    elif fault_type in ['wire_rope', 'rope_fault']:
        baseline = 0.05
        features = {
            "rms_baseline_ratio": round(max_rms / baseline, 2) if baseline > 0 else 1.0
        }
        
    elif fault_type == 'guide_rail':
        # 简单模拟导轨磨损率
        wear_percent = min(100.0, max_rms * 100) 
        features = {
            "wear_percent": round(wear_percent, 2)
        }
        
    elif fault_type == 'car':
        # 轿厢需要多通道指标
        directions = []
        for ch_name, sig in data_dict.items():
            rms = np.sqrt(np.mean(sig**2)) + 1e-6
            peak = np.max(np.abs(sig))
            mean_abs = np.mean(np.abs(sig)) + 1e-6
            smr = (np.mean(np.sqrt(np.abs(sig))))**2 + 1e-6
            
            # 这里简单用算出的指标除以某个健康基线(例如2.0)模拟ratio
            directions.append({
                "crest_factor_ratio": round((peak / rms) / 2.0, 2),
                "impulse_factor_ratio": round((peak / mean_abs) / 2.0, 2),
                "margin_factor_ratio": round((peak / smr) / 2.0, 2)
            })
        features = {"directions": directions}
        
    return features