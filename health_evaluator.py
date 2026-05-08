# health_evaluator.py
"""
电梯健康度评估模块 (V2.1.1)
- 采用 "本体加权 + 环境一票否决(短板效应)" 架构
- 配置分层：YAML负责[权重分配与功能开关]，Python全局设定负责[阈值与等级防呆]
- 2026.05.08 整合位移监测功能 (来自V3.0)
"""

import json
import os
import datetime
from typing import Dict, List, Optional, Union, Tuple

try:
    import yaml
except ImportError:
    print("错误: 缺少 pyyaml 库。请在命令行执行 'pip install pyyaml' 后再运行程序。")
    exit(1)

# ==============================================================================
# [全局管控区] GLOBAL_SETTINGS: 专家设定区，集中管理所有核心阈值、分数和等级映射
# ==============================================================================
GLOBAL_SETTINGS = {
    # 1. 评分等级划分范围 (区间左闭右开)
    "grade_ranges": {
        "default": {"H1": [90, 101], "H2": [75, 90], "H3": [60, 75], "H4": [0, 60]},
        
        # 曳引机子故障特定等级划分
        "motor_fault": {"H1": [90, 101], "H2": [80, 90], "H3": [70, 80], "H4": [0, 70]},
        "bearing_fault": {"H1": [90, 101], "H2": [75, 90], "H3": [60, 75], "H4": [0, 60]},
        "bearing_inner": {"H1": [90, 101], "H2": [75, 90], "H3": [60, 75], "H4": [0, 60]},
        "bearing_outer": {"H1": [90, 101], "H2": [75, 90], "H3": [60, 75], "H4": [0, 60]},
        "bearing_cage": {"H1": [90, 101], "H2": [80, 90], "H3": [50, 80], "H4": [0, 50]}, # 保持架高危
        "bearing_ball": {"H1": [90, 101], "H2": [75, 90], "H3": [60, 75], "H4": [0, 60]},
        "bolt_loose": {"H1": [90, 101], "H2": [70, 90], "H3": [40, 70], "H4": [0, 40]},
    },
    
    # 2. 机械本体评分细节参数 (注意：权重已移至外部 YAML)
    "scoring_params": {
        "traction": {
            "safety_baseline": 60.0,
            "rms_penalty_per_unit": 40.0,
            "bolt_penalty_per_unit": 20.0,
            "confidence_factor": 1.0
        },
        "wire_rope": {
            "segments": [
                {"range": [0, 4], "score": [100, 100]},
                {"range": [4, 6], "score": [100, 70]},
                {"range": [6, 8], "score": [70, 30]},
                {"range": [8, 9999], "score": [30, 0]}
            ]
        },
        "guide_rail": {"max_wear": 100},
        "car": {
            "vibration_threshold": 4.0,
            "smoothness_threshold": 2.0,
            "vibration_penalty_score": 50.0,
            "smoothness_penalty_score": 75.0,
            "base_penalty_slope": 5.0
        }
    },
    
    # 3. 环境与电气检测阈值参数
    "env_sensors": {
        "temp_t1": 20.0,        # 温度一级阈值: <= 20° 为满分
        "temp_t2": 60.0,        # 温度二级阈值: >= 60° 降为 0 分 (触发H4)
        "disp_t1": 5.0,         # 位移一级阈值: <= 5cm 为满分
        "disp_t2": 10.0,        # 位移二级阈值: >= 10cm 降为 0 分 (触发H4)
        "noise_threshold": 1.0, # 噪声比值阈值: > 1.0 开始扣分
        "noise_slope": 15.0     # 噪声每超 1.0 单位扣 15 分
    },
    
    "system_safety_baseline": 60.0
}
# ==============================================================================


# ------------------------------- 工具函数 ---------------------------------
def linear_score(value: float, in_min: float, in_max: float,
                 out_min: float = 100.0, out_max: float = 0.0) -> float:
    if in_max == in_min:
        return out_min
    ratio = (value - in_min) / (in_max - in_min)
    score = out_min + ratio * (out_max - out_min)
    return max(0.0, min(100.0, score))

def segmented_score(value: float, segments: List[Dict]) -> float:
    for seg in segments:
        low, high = seg["range"]
        if low <= value < high:
            s_low, s_high = seg["score"]
            return linear_score(value, low, high, s_low, s_high)
    return 0.0

# --------------------------- 环境与电气评估函数 ---------------------------
def evaluate_env_temperature(temp: float) -> float:
    t1 = GLOBAL_SETTINGS["env_sensors"]["temp_t1"]
    t2 = GLOBAL_SETTINGS["env_sensors"]["temp_t2"]
    if temp <= t1: return 100.0
    elif temp >= t2: return 0.0
    else: return linear_score(temp, t1, t2, 100.0, 0.0)

def evaluate_env_water(water: int) -> float:
    return 0.0 if water == 1 else 100.0

def evaluate_env_current(current: int) -> float:
    return 0.0 if current == 1 else 100.0

def evaluate_env_noise(ratio: float) -> float:
    th = GLOBAL_SETTINGS["env_sensors"]["noise_threshold"]
    slope = GLOBAL_SETTINGS["env_sensors"]["noise_slope"]
    return 100.0 if ratio <= th else max(0.0, 100.0 - (ratio - th) * slope)

def evaluate_env_displacement(disp: float) -> float:
    """位移检测评估（V3.0新增）"""
    t1 = GLOBAL_SETTINGS["env_sensors"]["disp_t1"]
    t2 = GLOBAL_SETTINGS["env_sensors"]["disp_t2"]
    if disp <= t1:
        return 100.0
    elif disp >= t2:
        return 0.0
    else:
        return linear_score(disp, t1, t2, 100.0, 0.0)


# --------------------------- 机械单设备评估函数 ---------------------------
def evaluate_traction(data: Dict, config: Dict) -> Tuple[float, Dict[str, float]]:
    details = {}
    
    # 1. 电机
    motor = data.get("motor_fault", {})
    rms = motor.get("rms_ratio", 1.0)
    conf = motor.get("confidence", 0.0)
    if rms > 1.0 and conf > 0:
        penalty = (rms - 1.0) * config.get("rms_penalty_per_unit", 40) * conf * config.get("confidence_factor", 1.0)
        details["motor_fault"] = max(0.0, 100.0 - penalty)
    else:
        details["motor_fault"] = 100.0

    # 2. 轴承(细分结构)
    bearing = data.get("bearing_fault", {})
    bearing_rms = bearing.get("rms_ratio", 1.0)
    sub_weights = config.get("bearing_sub_weights", {"inner": 0.25, "outer": 0.25, "cage": 0.25, "ball": 0.25})
    
    bearing_total = 0.0
    for sub, w in sub_weights.items():
        sub_conf = bearing.get(sub, 0.0)
        if bearing_rms > 1.0 and sub_conf > 0:
            penalty = (bearing_rms - 1.0) * config.get("rms_penalty_per_unit", 40) * sub_conf * config.get("confidence_factor", 1.0)
            sub_score = max(0.0, 100.0 - penalty)
        else:
            sub_score = 100.0
        details[f"bearing_{sub}"] = sub_score
        bearing_total += sub_score * w

    details["bearing_fault"] = bearing_total

    # 3. 螺栓松动
    bolt = data.get("bolt_loose", {})
    bolt_rms = bolt.get("rms_ratio", 1.0)
    if bolt_rms > 1.0:
        penalty = (bolt_rms - 1.0) * config.get("bolt_penalty_per_unit", 20)
        details["bolt_loose"] = max(0.0, 100.0 - penalty)
    else:
        details["bolt_loose"] = 100.0

    # 4. 综合曳引机成绩 (根据 YAML 注入的权重)
    weights = config.get("fault_weights", {})
    total_score = sum(details.get(ft, 100.0) * weights.get(ft, 1.0) for ft in weights)
    weighted_score = total_score / sum(weights.values()) if sum(weights.values()) > 0 else 100.0
    
    min_fault = min(details.values()) if details else 100.0
    baseline = config.get("safety_baseline", 60.0)

    final_score = min(min_fault, weighted_score) if min_fault < baseline else weighted_score
    return final_score, details

def evaluate_wire_rope(data: Dict, config: Dict) -> Tuple[float, Dict[str, float]]:
    score = segmented_score(data.get("rms_baseline_ratio", 1.0), config["segments"])
    return score, {"rms_baseline_ratio": score} 

def evaluate_guide_rail(data: Dict, config: Dict) -> Tuple[float, Dict[str, float]]:
    score = linear_score(data.get("wear_percent", 0.0), 0, config["max_wear"], 100.0, 0.0)
    return score, {"wear_percent": score}

def evaluate_car(data: Dict, config: Dict) -> Tuple[float, Dict[str, float]]:
    directions = data.get("directions", [])
    if not directions: return 100.0, {}

    vib_th = config["vibration_threshold"]
    smooth_th = config["smoothness_threshold"]
    slope = config["base_penalty_slope"]

    vib_cnt, smooth_cnt, all_max = 0, 0, []

    for d in directions:
        inds = [d.get("crest_factor_ratio", 1.0), d.get("impulse_factor_ratio", 1.0), d.get("margin_factor_ratio", 1.0)]
        if sum(1 for v in inds if v >= vib_th) >= 2: vib_cnt += 1
        elif sum(1 for v in inds if v >= smooth_th) >= 2: smooth_cnt += 1
        all_max.append(max(inds))

    if vib_cnt >= 2: return config["vibration_penalty_score"], {"vibration_issue": config["vibration_penalty_score"]}
    if smooth_cnt >= 2: return config["smoothness_penalty_score"], {"smoothness_issue": config["smoothness_penalty_score"]}

    score = max(0.0, 100.0 - (sum(all_max) / len(all_max) - 1.0) * slope)
    return score, {"avg_vibration": score}


# ------------------------------- 核心类 ---------------------------------
class HealthEvaluator:
    # 1. 类型提示改为支持 str 或 Dict
    def __init__(self, config_input: Union[str, Dict] = "health_config.yml"):
        self.config = self._load_config(config_input)

    def _load_config(self, config_input: Union[str, Dict]) -> Dict:
        # 设置基础默认权重及开关
        base = {
            "device_weights": {"曳引机": 0.4, "钢丝绳": 0.2, "导轨": 0.15, "轿厢": 0.25},
            "fault_weights": {"motor_fault": 0.3, "bearing_fault": 0.5, "bolt_loose": 0.2},
            "bearing_sub_weights": {"inner": 0.25, "outer": 0.25, "cage": 0.25, "ball": 0.25},
            "env_checks": {"enable_temperature": False, "enable_water": False, "enable_current": False, "enable_noise": False, "enable_displacement": False}
        }
        
        user_cfg = {}
        
        # 2. 动态判断传入的是文件路径还是现成的字典
        if isinstance(config_input, str):
            if os.path.exists(config_input):
                with open(config_input, 'r', encoding='utf-8') as f:
                    user_cfg = yaml.safe_load(f) or {}
        elif isinstance(config_input, dict):
            user_cfg = config_input

        # 从字典中动态加载各项权重和设置并覆盖默认值
        for key in base:
            if key in user_cfg:
                base[key].update(user_cfg[key])
                
        return base

    def _get_grade(self, score: float, unit_name: str = "default") -> str:
        ranges = GLOBAL_SETTINGS["grade_ranges"].get(unit_name, GLOBAL_SETTINGS["grade_ranges"]["default"])
        for grade, (low, high) in ranges.items():
            if low <= score < high: return grade
        return "H4"

    def evaluate(self, data: Dict) -> Dict:
        # ---- 动态组装参数 (合并全局设定与YAML外部权重) ----
        traction_params = GLOBAL_SETTINGS["scoring_params"]["traction"].copy()
        traction_params["fault_weights"] = self.config["fault_weights"]
        traction_params["bearing_sub_weights"] = self.config["bearing_sub_weights"]

        evaluators = {
            "曳引机": (evaluate_traction, traction_params),
            "钢丝绳": (evaluate_wire_rope, GLOBAL_SETTINGS["scoring_params"]["wire_rope"]),
            "钢带":   (evaluate_wire_rope, GLOBAL_SETTINGS["scoring_params"]["wire_rope"]),
            "导轨":   (evaluate_guide_rail, GLOBAL_SETTINGS["scoring_params"]["guide_rail"]),
            "轿厢":   (evaluate_car, GLOBAL_SETTINGS["scoring_params"]["car"])
        }

        # ---- 第一部分: 计算机械本体分数 ----
        device_scores = {}
        mechanical_total = 0.0
        weight_total = 0.0

        for device, weight in self.config["device_weights"].items():
            if device not in data: continue
            
            func, params = evaluators.get(device, (lambda d, c: (100.0, {}), {}))
            score, details = func(data[device], params)
            
            formatted_details = {}
            for sub_unit, sub_score in details.items():
                fmt = {"score": round(sub_score, 2)}
                if sub_unit in GLOBAL_SETTINGS["grade_ranges"]:
                    fmt["grade"] = self._get_grade(sub_score, sub_unit)
                formatted_details[sub_unit] = fmt
                
            device_scores[device] = {
                "score": round(score, 2),
                "grade": self._get_grade(score, device),
                "details": formatted_details
            }
            mechanical_total += score * weight
            weight_total += weight

        weighted_mech_score = (mechanical_total / weight_total) if weight_total > 0 else 100.0
        mech_scores_list = [v["score"] for v in device_scores.values()]
        min_mech = min(mech_scores_list) if mech_scores_list else 100.0
        mech_sys_score = min(min_mech, weighted_mech_score) if min_mech < GLOBAL_SETTINGS["system_safety_baseline"] else weighted_mech_score

        # ---- 第二部分: 计算环境与电气附加分数 (一票否决) ----
        env_scores = {}
        env_data = data.get("环境与电气", {})
        switches = self.config["env_checks"]
        
        if switches.get("enable_temperature"):
            s = evaluate_env_temperature(env_data.get("temperature", 20.0))
            env_scores["temperature"] = {"score": round(s, 2), "grade": self._get_grade(s)}
            
        if switches.get("enable_water"):
            s = evaluate_env_water(env_data.get("water", 0))
            env_scores["water"] = {"score": round(s, 2), "grade": self._get_grade(s)}
            
        if switches.get("enable_current"):
            s = evaluate_env_current(env_data.get("motor_current", 0))
            env_scores["motor_current"] = {"score": round(s, 2), "grade": self._get_grade(s)}
            
        if switches.get("enable_noise"):
            s = evaluate_env_noise(env_data.get("noise_ratio", 1.0))
            env_scores["noise"] = {"score": round(s, 2), "grade": self._get_grade(s)}

        # ---- 新增：位移检测 (V3.0 功能整合) ----
        if switches.get("enable_displacement"):
            s = evaluate_env_displacement(env_data.get("displacement", 0.0))
            env_scores["displacement"] = {"score": round(s, 2), "grade": self._get_grade(s)}

        min_env_score = min([info["score"] for info in env_scores.values()]) if env_scores else 100.0

        # ---- 第三部分: 最终裁决 ----
        final_system_score = min(mech_sys_score, min_env_score)

        return {
            "score": round(final_system_score, 2),
            "grade": self._get_grade(final_system_score),
            "device_scores": device_scores,
            "env_scores": env_scores
        }

# ------------------------------- 示例执行 ---------------------------------
if __name__ == "__main__":
    import shutil
    
    if os.path.exists("input_data.json"):
        with open("input_data.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            input_data = raw_data.get("input_data", raw_data)
        
        evaluator = HealthEvaluator("health_config.yml")
        result = evaluator.evaluate(input_data)
        
        h4_items = []
        for device, info in result['device_scores'].items():
            if info.get('grade') == 'H4': h4_items.append(f"[{device}]主设备异常")
            for sub_unit, sub_info in info.get('details', {}).items():
                if sub_info.get('grade') == 'H4': h4_items.append(f"[{device}-{sub_unit}]报警")
        for env_item, info in result.get('env_scores', {}).items():
            if info.get('grade') == 'H4': h4_items.append(f"[环境-{env_item}]危机报警")

        print("\n" + "="*45)
        print("         电梯健康度评估摘要 (控制台)")
        print("="*45)
        print(f"  > 整体健康度得分: \t{result['score']}")
        print(f"  > 整体健康度等级: \t{result['grade']}")
        
        if h4_items:
            print(f"  > ⚠️ 触发 H4 短板项: \t{', '.join(h4_items)}")
        else:
            print("  > 触发 H4 短板项: \t无")
        print("="*45)
        
        now = datetime.datetime.now()
        time_str = now.strftime("%Y_%m_%d_%H%M")
        health_dir = "health"
        if not os.path.exists(health_dir): os.makedirs(health_dir)
            
        file_path = os.path.join(health_dir, f"{time_str}_{result['score']}.txt")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write("             电梯健康度评估详细报告\n")
            f.write("="*50 + "\n")
            f.write(f"评估时间: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"系统最终健康度: {result['score']} (等级: {result['grade']})\n")
            f.write("-" * 50 + "\n")
            
            f.write("【一、 环境与电气状态 (一票否决项)】\n")
            if not result['env_scores']:
                f.write("  - 未开启相关检测或无数据接入\n")
            for env_name, info in result['env_scores'].items():
                f.write(f"  * {env_name:<15} | 得分: {info['score']:<6} | 等级: {info['grade']}\n")
            
            f.write("\n【二、 机械本体状态】\n")
            for device, info in result['device_scores'].items():
                f.write(f"\n  [{device}] 综合得分: {info['score']} (等级: {info.get('grade', 'N/A')})\n")
                if info.get('details'):
                    for sub_unit, sub_info in info['details'].items():
                        grade_str = f"| 等级: {sub_info['grade']}" if 'grade' in sub_info else ""
                        f.write(f"    - {sub_unit:<15} | 得分: {sub_info['score']:<6} {grade_str}\n")
            
            f.write("\n" + "="*50 + "\n")
            
        print(f"\n详细诊断报告已生成并保存至: {file_path}\n")