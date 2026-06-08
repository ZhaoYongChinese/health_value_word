import numpy as np
from scipy.signal import hilbert, find_peaks

def extract_features(fault_type: str, data_dict: dict, fs: float = 10000.0, bearing_params: dict = None, baselines: dict = None, motor_params: dict = None, rms_params: dict = None) -> dict:
    """
    接收多通道数据字典，根据配置执行滑窗过滤与特征指标提取。
    加入连续异常校验机制，有效滤除偶发冲击。
    """
    all_signals = list(data_dict.values())
    baselines = baselines or {}
    rms_params = rms_params or {}
    
    # 防崩溃保护
    if not all_signals or len(all_signals) == 0 or len(all_signals[0]) == 0:
        return {}

    # 读取 RMS 滑窗参数
    duration_sec = rms_params.get("duration_sec", 2.0)
    trigger_count = rms_params.get("trigger_count", 3)
    trigger_ratio = rms_params.get("trigger_ratio", 1.0)
    window_size = int(duration_sec * fs)

    # ---------------- 内部函数：计算连续超标的最大次数 ----------------
    def get_max_consecutive_outliers(rms_list, base_val, ratio_thresh):
        max_consec = 0
        curr_consec = 0
        for r in rms_list:
            if base_val > 0 and (r / base_val) > ratio_thresh:
                curr_consec += 1
                if curr_consec > max_consec:
                    max_consec = curr_consec
            else:
                curr_consec = 0  
        return max_consec

    # 根据部件明确 Baseline
    if fault_type == 'car':
        baseline = baselines.get('car', 1.0)
    elif fault_type in ['wire_rope', 'rope_fault']:
        baseline = baselines.get('wire_rope', 1)
    else:
        baseline = baselines.get(fault_type, 1)

    features = {}

    # ---------------- 轿厢特性处理逻辑 ----------------
    if fault_type == 'car':
        directions = []
        for i, (ch_name, sig) in enumerate(data_dict.items()):
            axis = "X" if i == 0 else ("Y" if i == 1 else "Z")
            if len(sig) == 0:
                continue

            # 切分为 duration_sec 秒的滑窗
            if window_size > 0 and len(sig) >= window_size:
                num_chunks = len(sig) // window_size
                chunks = [sig[j*window_size : (j+1)*window_size] for j in range(num_chunks)]
            else:
                chunks = [sig]

            # 存储每个窗口的指标：
            window_metrics = []
            chunk_rms_list = []
            for chunk in chunks:
                r = np.sqrt(np.mean(chunk**2))
                p = np.max(np.abs(chunk))
                ma = np.mean(np.abs(chunk)) + 1e-6
                sm = (np.mean(np.sqrt(np.abs(chunk))))**2 + 1e-6
                window_metrics.append((r, p, ma, sm))
                chunk_rms_list.append(r)

            ch_max_rms = max(chunk_rms_list) if chunk_rms_list else 0.0
            max_consecutive = get_max_consecutive_outliers(chunk_rms_list, baseline, trigger_ratio)
            triggered = (max_consecutive >= trigger_count)

            if triggered:
                used_rms = ch_max_rms
                # 找出所有 RMS 超过基线的窗口
                outlier_indices = [idx for idx, r in enumerate(chunk_rms_list)
                                if baseline > 0 and (r / baseline) > trigger_ratio]
                if outlier_indices:
                    # 从异常窗口中取 CF、IF、MF 的最大值
                    cf_list = [window_metrics[i][1] / (window_metrics[i][0] + 1e-6) for i in outlier_indices]
                    if_list = [window_metrics[i][1] / (window_metrics[i][2] + 1e-6) for i in outlier_indices]
                    mf_list = [window_metrics[i][1] / (window_metrics[i][3] + 1e-6) for i in outlier_indices]
                    crest_factor_ratio = max(cf_list)
                    impulse_factor_ratio = max(if_list)
                    margin_factor_ratio = max(mf_list)
                else:
                    peak = np.max(np.abs(sig))
                    mean_abs = np.mean(np.abs(sig)) + 1e-6
                    smr = (np.mean(np.sqrt(np.abs(sig))))**2 + 1e-6
                    crest_factor_ratio = peak / (used_rms + 1e-6)
                    impulse_factor_ratio = peak / mean_abs
                    margin_factor_ratio = peak / smr
            else:
                # 未触发：强制使用基线值，所有因子设为 1.0（表示健康）
                used_rms = baseline if baseline > 0 else 1.0
                crest_factor_ratio = 1.0
                impulse_factor_ratio = 1.0
                margin_factor_ratio = 1.0

            rms_ratio = used_rms / baseline if baseline > 0 else 1.0

            directions.append({
                "name": axis,
                "rms_ratio": rms_ratio,
                "crest_factor_ratio": crest_factor_ratio,
                "impulse_factor_ratio": impulse_factor_ratio,
                "margin_factor_ratio": margin_factor_ratio
            })
        features = {"directions": directions}
        return features

    # ---------------- 曳引机及常规部件特性处理逻辑 ----------------
    max_rms = 0.0
    best_sig = None
    is_triggered = False

    for sig in all_signals:
        if len(sig) == 0: continue
        
        if window_size > 0 and len(sig) >= window_size:
            num_chunks = len(sig) // window_size
            chunks = [sig[i*window_size : (i+1)*window_size] for i in range(num_chunks)]
        else:
            chunks = [sig]
            
        chunk_rms_list = [np.sqrt(np.mean(c**2)) for c in chunks]
        ch_max_rms = max(chunk_rms_list) if chunk_rms_list else 0.0
        
        # 判断当前通道是否有足够的【连续】滑窗触发异常
        max_consecutive = get_max_consecutive_outliers(chunk_rms_list, baseline, trigger_ratio)
        
        # 寻找能量最大的通道
        if ch_max_rms >= max_rms:
            max_rms = ch_max_rms
            best_sig = sig
            # 以最恶劣通道的统计结果为准
            is_triggered = (max_consecutive >= trigger_count)

    # 触发保护：只有连续滑窗触发故障时才分析 max_rms / baseline，否则直接为 1.0
    ratio = max_rms / baseline if (is_triggered and baseline > 0) else 1.0

    if fault_type == 'motor_fault':
        base_confidence = min(1.0, max(0.0, (ratio - 1) * 0.2))
        features = {"rms_ratio": round(ratio, 2), "confidence": round(base_confidence, 2)}
        
        # 若判定为已触发 (连续>=3个异常窗口)，则执行 FFT 细分判断
        if is_triggered and best_sig is not None:
            best_sig_centered = best_sig - np.mean(best_sig)
            N = len(best_sig_centered)
            
            spectrum = 2.0 / N * np.abs(np.fft.rfft(best_sig_centered))
            freqs = np.fft.rfftfreq(N, 1/fs)
            
            peaks, _ = find_peaks(spectrum, height=0.1 * np.max(spectrum))
            if len(peaks) > 0:
                sorted_idx = sorted(peaks, key=lambda x: spectrum[x], reverse=True)
                f1 = freqs[sorted_idx[0]]
                amp1 = spectrum[sorted_idx[0]]
                
                def get_amp(target_freq, tolerance=1.0):
                    idx = np.argmin(np.abs(freqs - target_freq))
                    if abs(freqs[idx] - target_freq) <= tolerance:
                        return spectrum[idx]
                    return 0.0
                    
                amp2 = get_amp(2 * f1)
                amp3 = get_amp(3 * f1)
                
                r2 = amp2 / amp1 if amp1 > 0 else 0
                r3 = amp3 / amp1 if amp1 > 0 else 0
                
                mp = motor_params or {}
                r2_m = mp.get("ratio2_misalign", 0.6)
                r3_m = mp.get("ratio3_misalign", 0.3)
                r2_e = mp.get("ratio2_eccentric", 0.3)
                r3_e = mp.get("ratio3_eccentric", 0.15)
                
                if r2 >= r2_m and r3 >= r3_m:
                    sub_fault = "rotor_misalignment"
                    sub_conf = min(1.0, (r2 + r3) / 1.5)
                elif r2 >= r2_e and r3 >= r3_e:
                    sub_fault = "stator_eccentricity"
                    sub_conf = min(1.0, (r2 + r3) / 1.0)
                else:
                    if r2 >= r3:
                        sub_fault = "rotor_misalignment"
                        sub_conf = min(0.5, r2)
                    else:
                        sub_fault = "stator_eccentricity"
                        sub_conf = min(0.5, r3)
                        
                final_conf = max(base_confidence, sub_conf)
                features[sub_fault] = round(final_conf, 2)
                features["confidence"] = round(final_conf, 2)
                
    elif fault_type == 'bearing_fault':
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
        
        sub_confidences = {"inner": 0.0, "outer": 0.0, "cage": 0.0, "ball": 0.0}
        
        # 仅在连续触发前提下，才提取包络谱细分故障
        if is_triggered and best_sig is not None:
            best_sig_centered = best_sig - np.mean(best_sig) 
            envelope = np.abs(hilbert(best_sig_centered))
            envelope -= np.mean(envelope)
            
            spectrum = np.abs(np.fft.rfft(envelope))
            freqs = np.fft.rfftfreq(len(envelope), 1/fs)
            
            window = 2
            for key, target_f in location_dict.items():
                if len(freqs) < 2:
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
        features = {"rms_ratio": round(ratio, 2)}
        
    elif fault_type in ['wire_rope', 'rope_fault']:
        features = {"rms_baseline_ratio": round(ratio, 2)}
        
    elif fault_type == 'guide_rail':
        if is_triggered:
            features = {"wear_percent": round(min(100.0, max_rms * 100), 2)}
        else:
            features = {"wear_percent": 0.0}

    return features