# health_evaluator.py
"""
电梯健康度评估模块
"""

import json
import os
import datetime
from typing import Dict, List, Union, Tuple

try:
    import yaml
except ImportError:
    print("错误: 缺少 pyyaml 库。请在命令行执行 'pip install pyyaml' 后再运行程序。")
    exit(1)

GLOBAL_SETTINGS = {
    "grade_ranges": {
        "default": {"H1": [90, 101], "H2": [75, 90], "H3": [60, 75], "H4": [0, 60]},
        "motor_fault": {"H1": [90, 101], "H2": [80, 90], "H3": [70, 80], "H4": [0, 70]},
        "bearing_fault": {"H1": [90, 101], "H2": [75, 90], "H3": [60, 75], "H4": [0, 60]},
        "bearing_inner": {"H1": [90, 101], "H2": [75, 90], "H3": [60, 75], "H4": [0, 60]},
        "bearing_outer": {"H1": [90, 101], "H2": [75, 90], "H3": [60, 75], "H4": [0, 60]},
        "bearing_cage": {"H1": [90, 101], "H2": [80, 90], "H3": [50, 80], "H4": [0, 50]},
        "bearing_ball": {"H1": [90, 101], "H2": [75, 90], "H3": [60, 75], "H4": [0, 60]},
        "bolt_loose": {"H1": [90, 101], "H2": [70, 90], "H3": [40, 70], "H4": [0, 40]},
        "轿架振动": {"H1": [90, 101], "H2": [80, 90], "H3": [50, 80], "H4": [0, 50]},
        "平稳度异常": {"H1": [90, 101], "H2": [80, 90], "H3": [60, 80], "H4": [0, 60]},
    },
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
            "frame_vibration_penalty": 40.0,  # Z轴严重扣分
            "smoothness_penalty": 55.0        # X/Y轴中等扣分
        }
    },
    "system_safety_baseline": 60.0
}

# ------------------------------- 工具函数 ---------------------------------
def linear_score(value, in_min, in_max, out_min=100.0, out_max=0.0):
    if in_max == in_min: return out_min
    ratio = (value - in_min) / (in_max - in_min)
    return max(0.0, min(100.0, out_min + ratio * (out_max - out_min)))

def segmented_score(value, segments):
    for seg in segments:
        low, high = seg["range"]
        if low <= value < high:
            return linear_score(value, low, high, seg["score"][0], seg["score"][1])
    return 0.0

def apply_smooth_penalty(avg_score: float, min_score: float, baseline: float = 60.0, k: float = 0.7) -> float:
    if min_score >= baseline: return avg_score
    severity = (baseline - min_score) / baseline
    penalty_weight = severity ** k
    return avg_score * (1 - penalty_weight) + min_score * penalty_weight

def score_to_fuzzy_distribution(score: float, unit_name: str = "default", transition_radius: float = 2.5) -> Dict[str, float]:
    ranges = GLOBAL_SETTINGS["grade_ranges"].get(unit_name, GLOBAL_SETTINGS["grade_ranges"]["default"])
    sorted_grades = sorted(ranges.items(), key=lambda x: x[1][0])
    grade_names = [g[0] for g in sorted_grades]
    min_grade, max_grade = grade_names[0], grade_names[-1]

    memberships = {}
    r = transition_radius
    
    for grade, (low, high) in ranges.items():
        is_min, is_max = (grade == min_grade), (grade == max_grade)
        if is_min: left_val = 1.0
        elif score <= low - r: left_val = 0.0
        elif score >= low + r: left_val = 1.0
        else: left_val = (score - (low - r)) / (2 * r)
        
        if is_max: right_val = 1.0
        elif score >= high + r: right_val = 0.0
        elif score <= high - r: right_val = 1.0
        else: right_val = (high + r - score) / (2 * r)

        memberships[grade] = max(0.0, min(left_val, right_val))

    total_mem = sum(memberships.values())
    probs = {g: round((m / total_mem) * 100, 1) for g, m in memberships.items() if m > 0} if total_mem > 0 else {min_grade: 100.0}
    return dict(sorted(probs.items(), key=lambda item: item[1], reverse=True))

# --------------------------- 环境与电气评估函数 (全新精简) ---------------------------
def evaluate_env_status(status_flag: int) -> float:
    """
    环境数据已在边缘端完成预处理和阈值判断。
    0: 正常 -> 100分
    1: 报警 -> 45分 (落入 H4 危险区间)
    """
    return 45.0 if status_flag == 1 else 100.0

# --------------------------- 机械单设备评估函数 ---------------------------
def evaluate_traction(data: Dict, config: Dict) -> Tuple[float, Dict[str, Union[float, Dict]]]:
    details = {}
    
    # 1. 电机故障
    motor = data.get("motor_fault", {})
    rms = motor.get("rms_ratio", 1.0)
    conf = motor.get("confidence", 0.0)
    if rms > 1.0 and conf > 0:
        penalty = (rms - 1.0) * config.get("rms_penalty_per_unit", 40) * conf * config.get("confidence_factor", 1.0)
        details["motor_fault"] = max(0.0, 100.0 - penalty)
    else:
        details["motor_fault"] = 100.0

    # 2. 轴承组（内部嵌套）
    bearing = data.get("bearing_fault", {})
    bearing_rms = bearing.get("rms_ratio", 1.0)
    sub_weights = config.get("bearing_sub_weights", {"inner": 0.25, "outer": 0.25, "cage": 0.25, "ball": 0.25})
    
    bearing_subs = {}
    min_bearing_sub = 100.0
    for sub, w in sub_weights.items():
        sub_conf = bearing.get(sub, 0.0)
        if bearing_rms > 1.0 and sub_conf > 0:
            penalty = (bearing_rms - 1.0) * config.get("rms_penalty_per_unit", 40) * sub_conf * config.get("confidence_factor", 1.0)
            sub_score = max(0.0, 100.0 - penalty)
        else:
            sub_score = 100.0
        bearing_subs[f"bearing_{sub}"] = sub_score
        if sub_score < min_bearing_sub:
            min_bearing_sub = sub_score
    
    # 计算轴承综合分（加权+平滑惩罚）
    bearing_total = sum(sub_score * sub_weights[sub] for sub, sub_score in 
                        zip(sub_weights.keys(), [bearing_subs[f"bearing_{s}"] for s in sub_weights.keys()]))
    bearing_total = apply_smooth_penalty(bearing_total, min_bearing_sub, config.get("safety_baseline", 60.0))
    
    details["bearing_fault"] = {
        "score": bearing_total,
        "subs": bearing_subs
    }

    # 3. 螺栓松动
    bolt = data.get("bolt_loose", {})
    bolt_rms = bolt.get("rms_ratio", 1.0)
    if bolt_rms > 1.0:
        penalty = (bolt_rms - 1.0) * config.get("bolt_penalty_per_unit", 20)
        details["bolt_loose"] = max(0.0, 100.0 - penalty)
    else:
        details["bolt_loose"] = 100.0

    # 综合曳引机分数
    fault_scores = {
        "motor_fault": details["motor_fault"],
        "bearing_fault": details["bearing_fault"]["score"], 
        "bolt_loose": details["bolt_loose"]
    }
    weights = config.get("fault_weights", {})
    total_score = sum(fault_scores[ft] * weights.get(ft, 1.0) for ft in weights)
    weighted_score = total_score / sum(weights.values()) if sum(weights.values()) > 0 else 100.0
    
    min_fault = min(fault_scores.values())
    final_score = apply_smooth_penalty(weighted_score, min_fault, config.get("safety_baseline", 60.0))
    
    return final_score, details

def evaluate_wire_rope(data: Dict, config: Dict):
    score = segmented_score(data.get("rms_baseline_ratio", 1.0), config["segments"])
    return score, {"rms_baseline_ratio": score} 

def evaluate_guide_rail(data: Dict, config: Dict):
    score = linear_score(data.get("wear_percent", 0.0), 0, config["max_wear"], 100.0, 0.0)
    return score, {"wear_percent": score}

def evaluate_car(data: Dict, config: Dict):
    score_details = {}
    if data.get("has_frame_vibration", False):
        score_details["轿架振动"] = config.get("frame_vibration_penalty", 40.0)
    if data.get("has_smoothness_issue", False):
        score_details["平稳度异常"] = config.get("smoothness_penalty", 55.0)
    if not score_details:
        score_details["轿厢综合运行"] = 100.0

    final_score = min(score_details.values())
    return final_score, score_details

class HealthEvaluator:
    def __init__(self, config_input: Union[str, Dict] = "health_config.yml"):
        self.config = self._load_config(config_input)

    def _load_config(self, config_input: Union[str, Dict]) -> Dict:
        base = {
            "device_weights": {"曳引机": 0.4, "钢丝绳": 0.2, "导轨": 0.15, "轿厢": 0.25},
            "fault_weights": {"motor_fault": 0.3, "bearing_fault": 0.5, "bolt_loose": 0.2},
            "bearing_sub_weights": {"inner": 0.25, "outer": 0.25, "cage": 0.25, "ball": 0.25},
            "env_checks": {"enable_temperature": False, "enable_water": False, "enable_current": False, "enable_noise": False, "enable_displacement": False}
        }
        user_cfg = {}
        if isinstance(config_input, str) and os.path.exists(config_input):
            with open(config_input, 'r', encoding='utf-8') as f: user_cfg = yaml.safe_load(f) or {}
        elif isinstance(config_input, dict):
            user_cfg = config_input

        for key in base:
            if key in user_cfg: base[key].update(user_cfg[key])
        return base

    def _get_crisp_grade(self, score: float, unit_name: str = "default") -> str:
        ranges = GLOBAL_SETTINGS["grade_ranges"].get(unit_name, GLOBAL_SETTINGS["grade_ranges"]["default"])
        for grade, (low, high) in ranges.items():
            if low <= score < high: return grade
        return "H4"

    def _format_score_node(self, score: float, unit_name: str = "default") -> Dict:
        return {
            "score": round(score, 2),
            "crisp_grade": self._get_crisp_grade(score, unit_name),
            "fuzzy_distribution": score_to_fuzzy_distribution(score, unit_name)
        }

    def evaluate(self, data: Dict) -> Dict:
        traction_params = GLOBAL_SETTINGS["scoring_params"]["traction"].copy()
        traction_params["fault_weights"] = self.config["fault_weights"]
        traction_params["bearing_sub_weights"] = self.config["bearing_sub_weights"]

        evaluators = {
            "曳引机": (evaluate_traction, traction_params),
            "钢丝绳": (evaluate_wire_rope, GLOBAL_SETTINGS["scoring_params"]["wire_rope"]),
            "轿厢":   (evaluate_car, GLOBAL_SETTINGS["scoring_params"]["car"]),
            "导轨":   (evaluate_guide_rail, GLOBAL_SETTINGS["scoring_params"]["guide_rail"])
        }

        device_scores = {}
        mechanical_total = 0.0
        weight_total = 0.0

        for device, weight in self.config["device_weights"].items():
            if device not in data: continue
            
            func, params = evaluators.get(device, (lambda d, c: (100.0, {}), {}))
            score, details = func(data[device], params)
            
            formatted_details = {}
            for sub_unit, sub_content in details.items():
                if isinstance(sub_content, dict) and "subs" in sub_content:
                    parent_node = self._format_score_node(sub_content["score"], sub_unit)
                    sub_nodes = {}
                    for sub_sub, sub_sub_score in sub_content["subs"].items():
                        sub_nodes[sub_sub] = self._format_score_node(sub_sub_score, sub_sub)
                    parent_node["sub_details"] = sub_nodes
                    formatted_details[sub_unit] = parent_node
                else:
                    formatted_details[sub_unit] = self._format_score_node(sub_content, sub_unit)
                
            device_scores[device] = {
                **self._format_score_node(score, device),
                "details": formatted_details
            }
            mechanical_total += score * weight
            weight_total += weight

        weighted_mech_score = (mechanical_total / weight_total) if weight_total > 0 else 100.0
        
        all_mech_scores = []
        for info in device_scores.values():
            all_mech_scores.append(info["score"])
            for det in info["details"].values():
                all_mech_scores.append(det["score"])
                if "sub_details" in det:
                    all_mech_scores.extend(s["score"] for s in det["sub_details"].values())
        
        global_min_mech = min(all_mech_scores) if all_mech_scores else 100.0
        mech_sys_score = apply_smooth_penalty(weighted_mech_score, global_min_mech, GLOBAL_SETTINGS["system_safety_baseline"])

        # ========================================================
        # 环境与电气 (边缘端传入 0/1, 一票否决)
        # ========================================================
        env_scores = {}
        env_data = data.get("环境与电气", {})
        switches = self.config["env_checks"]
        
        env_fields_mapping = {
            "enable_temperature": "temperature",
            "enable_water": "water",
            "enable_current": "motor_current",
            "enable_noise": "noise_ratio",
            "enable_displacement": "displacement"
        }
        
        for switch_key, data_field in env_fields_mapping.items():
            if switches.get(switch_key):
                # 获取数据，默认 0 为正常
                status_flag = env_data.get(data_field, 0)
                # 计算得分 (0->100, 1->45)
                s = evaluate_env_status(status_flag)
                env_scores[data_field] = self._format_score_node(s)

        # 只要有一项是 45 (即环境告警), min_env_score 就会是 45
        min_env_score = min([info["score"] for info in env_scores.values()]) if env_scores else 100.0
        
        # 整体健康度取机械与环境的最小值
        final_system_score = min(mech_sys_score, min_env_score)

        return {
            "score": round(final_system_score, 2),
            **self._format_score_node(final_system_score),
            "device_scores": device_scores,
            "env_scores": env_scores
        }

# ------------------------------- 测试脚本 (支持嵌套输出) ---------------------------------
if __name__ == "__main__":
    if os.path.exists("input_data.json"):
        with open("input_data.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            input_data = raw_data.get("input_data", raw_data)
        
        evaluator = HealthEvaluator("health_config.yml")
        result = evaluator.evaluate(input_data)
        
        h4_items = []
        for device, info in result['device_scores'].items():
            if info.get('crisp_grade') == 'H4':
                h4_items.append(f"[{device}]主设备异常")
            for unit_name, unit_info in info.get('details', {}).items():
                if unit_info.get('crisp_grade') == 'H4':
                    h4_items.append(f"[{device}-{unit_name}]报警")
                if 'sub_details' in unit_info:
                    for sub_name, sub_info in unit_info['sub_details'].items():
                        if sub_info.get('crisp_grade') == 'H4':
                            h4_items.append(f"[{device}-{unit_name}-{sub_name}]报警")
        for env_item, info in result.get('env_scores', {}).items():
            if info.get('crisp_grade') == 'H4':
                h4_items.append(f"[环境-{env_item}]危机报警")

        print("\n" + "="*45)
        print("         电梯健康度评估摘要 (控制台)")
        print("="*45)
        print(f"  > 整体健康度得分: \t{result['score']}")
        print(f"  > 整体健康度等级: \t{result['crisp_grade']}")
        if h4_items:
            print(f"  > ⚠️ 触发 H4 短板项: \t{', '.join(h4_items)}")
        else:
            print("  > 触发 H4 短板项: \t无")
        print("="*45)

        now = datetime.datetime.now()
        time_str = now.strftime("%Y_%m_%d_%H%M")
        health_dir = "health"
        os.makedirs(health_dir, exist_ok=True)
        file_path = os.path.join(health_dir, f"{time_str}_{result['score']}.txt")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write("             电梯健康度评估详细报告\n")
            f.write("="*50 + "\n")
            f.write(f"评估时间: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"系统最终健康度: {result['score']} (等级: {result['crisp_grade']})\n")
            f.write(f"模糊分布: {result.get('fuzzy_distribution', {})}\n")
            f.write("-" * 50 + "\n")
            
            f.write("\n【一、 环境与电气状态 (一票否决项)】\n")
            if not result['env_scores']:
                f.write("  - 未开启相关检测或无数据接入\n")
            for env_name, info in result['env_scores'].items():
                f.write(f"  * {env_name:<15} | 得分: {info['score']:<6} | 等级: {info['crisp_grade']}")
                f.write(f" | 模糊: {info.get('fuzzy_distribution', {})}\n")
            
            f.write("\n【二、 机械本体状态】\n")
            for device, info in result['device_scores'].items():
                f.write(f"\n  [{device}] 综合得分: {info['score']} (等级: {info['crisp_grade']})")
                f.write(f" | 模糊: {info.get('fuzzy_distribution', {})}\n")
                if info.get('details'):
                    for unit_name, unit_info in info['details'].items():
                        f.write(f"    - {unit_name:<15} | 得分: {unit_info['score']:<6} | 等级: {unit_info['crisp_grade']}")
                        f.write(f" | 模糊: {unit_info.get('fuzzy_distribution', {})}\n")
                        if 'sub_details' in unit_info:
                            for sub_name, sub_info in unit_info['sub_details'].items():
                                f.write(f"      · {sub_name:<13} | 得分: {sub_info['score']:<6} | 等级: {sub_info['crisp_grade']}")
                                f.write(f" | 模糊: {sub_info.get('fuzzy_distribution', {})}\n")
            f.write("\n" + "="*50 + "\n")
        
        print(f"\n详细诊断报告已生成并保存至: {file_path}\n")
    else:
        print("❌ 错误: 找不到输入文件 'input_data.json'，请将其与该脚本放在同一目录。")