import json
import os
import datetime
from typing import Dict, Union, Tuple

try:
    import yaml
except ImportError:
    exit(1)

GLOBAL_SETTINGS = {
    "grade_ranges": {
        "default": {"H1": [90, 101], "H2": [75, 90], "H3": [60, 75], "H4": [0, 60]},
        "轿厢运行": {"H1": [90, 101], "H2": [80, 90], "H3": [60, 80], "H4": [0, 60]},
    },
    "system_safety_baseline": 60.0
}

def linear_score(value, in_min, in_max, out_min=100.0, out_max=0.0):
    if in_max == in_min: return out_min
    ratio = (value - in_min) / (in_max - in_min)
    return max(0.0, min(100.0, out_min + ratio * (out_max - out_min)))

def apply_smooth_penalty(avg_score: float, min_score: float, baseline: float = 60.0, k: float = 0.7) -> float:
    if min_score >= baseline: return avg_score
    severity = (baseline - min_score) / baseline
    penalty_weight = severity ** k
    return avg_score * (1 - penalty_weight) + min_score * penalty_weight

def score_to_fuzzy_distribution(score: float, unit_name: str = "default") -> Dict[str, float]:
    ranges = GLOBAL_SETTINGS["grade_ranges"].get(unit_name, GLOBAL_SETTINGS["grade_ranges"]["default"])
    sorted_grades = sorted(ranges.items(), key=lambda x: x[1][0])
    grade_names = [g[0] for g in sorted_grades]
    min_grade, max_grade = grade_names[0], grade_names[-1]

    memberships = {}
    r = 2.5
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

def evaluate_env_status(status_flag: int) -> float:
    return 45.0 if status_flag == 1 else 100.0

def evaluate_traction(data: Dict, config: Dict):
    details = {}
    
    # 电机
    motor = data.get("motor_fault", {})
    rms = motor.get("rms_ratio", 1.0)
    conf = motor.get("confidence", 0.0)
    if rms > 1.0 and conf > 0:
        penalty = (rms - 1.0) * 40.0 * conf
        details["motor_fault"] = {"score": max(0.0, 100.0 - penalty), "sub_fault": motor.get("sub_fault", "未知")}
    else:
        details["motor_fault"] = {"score": 100.0, "sub_fault": "正常"}

    # 轴承
    bearing = data.get("bearing_fault", {})
    bearing_rms = bearing.get("rms_ratio", 1.0)
    sub_weights = config.get("bearing_sub_weights", {"inner": 0.25, "outer": 0.25, "cage": 0.25, "ball": 0.25})
    bearing_subs = {}
    min_b = 100.0
    for sub, w in sub_weights.items():
        sub_conf = bearing.get(sub, 0.0)
        sub_score = max(0.0, 100.0 - (bearing_rms - 1.0) * 40.0 * sub_conf) if (bearing_rms > 1.0 and sub_conf > 0) else 100.0
        bearing_subs[f"bearing_{sub}"] = sub_score
        min_b = min(min_b, sub_score)
        
    b_total = apply_smooth_penalty(sum(bearing_subs[f"bearing_{s}"] * sub_weights[s] for s in sub_weights), min_b)
    details["bearing_fault"] = {"score": b_total, "subs": bearing_subs}

    # 螺栓
    bolt = data.get("bolt_loose", {})
    bolt_rms = bolt.get("rms_ratio", 1.0)
    details["bolt_loose"] = {"score": max(0.0, 100.0 - (bolt_rms - 1.0) * 20.0) if bolt_rms > 1.0 else 100.0}

    fault_scores = {k: v["score"] for k, v in details.items()}
    weights = config.get("fault_weights", {})
    weighted_score = sum(fault_scores[ft] * weights.get(ft, 1.0) for ft in weights) / sum(weights.values())
    final_score = apply_smooth_penalty(weighted_score, min(fault_scores.values()))
    
    return final_score, details

def evaluate_wire_rope(data: Dict, config: Dict):
    ratio = data.get("rms_baseline_ratio", 1.0)
    # 基于分段打分
    if ratio < 4.0: score = 100.0
    elif ratio < 6.0: score = linear_score(ratio, 4.0, 6.0, 100.0, 70.0)
    elif ratio < 8.0: score = linear_score(ratio, 6.0, 8.0, 70.0, 30.0)
    else: score = linear_score(ratio, 8.0, 12.0, 30.0, 0.0)
    
    return score, {"wire_rope": {"score": score, "sub_fault": data.get("sub_fault", "正常")}}

def evaluate_guide_rail(data: Dict, config: Dict):
    score = linear_score(data.get("wear_percent", 0.0), 0, 100, 100.0, 0.0)
    return score, {"guide_rail": {"score": score}}

def evaluate_car(data: Dict, config: Dict):
    # [修改] 使用因子比值进行连续扣分
    factors = []
    for axis in ["X", "Y", "Z"]:
        factors.extend([data.get(f"{axis}_pf", 0), data.get(f"{axis}_imp", 0), data.get(f"{axis}_mar", 0)])
    
    max_factor = max(factors) if factors else 0
    # 假设因子正常在3左右，超过5开始明显扣分，达到15为0分
    score = linear_score(max_factor, 3.0, 15.0, 100.0, 0.0)
    
    return score, {"轿厢运行": {"score": score}}

class HealthEvaluator:
    def __init__(self, config_input: Union[str, Dict]):
        self.config = config_input

    def _get_crisp_grade(self, score: float, unit_name: str = "default") -> str:
        ranges = GLOBAL_SETTINGS["grade_ranges"].get(unit_name, GLOBAL_SETTINGS["grade_ranges"]["default"])
        for grade, (low, high) in ranges.items():
            if low <= score < high: return grade
        return "H4"

    def _format_node(self, score: float, unit_name: str = "default", extra: dict = None) -> Dict:
        res = {
            "score": round(score, 2),
            "crisp_grade": self._get_crisp_grade(score, unit_name),
            "fuzzy_distribution": score_to_fuzzy_distribution(score, unit_name)
        }
        if extra: res.update(extra)
        return res

    def evaluate(self, data: Dict) -> Dict:
        evaluators = {
            "曳引机": evaluate_traction, "钢丝绳": evaluate_wire_rope,
            "轿厢": evaluate_car, "导轨": evaluate_guide_rail
        }

        device_scores = {}
        mechanical_total, weight_total = 0.0, 0.0

        for device, weight in self.config["device_weights"].items():
            if device not in data: continue
            score, details = evaluators[device](data[device], self.config)
            
            formatted_details = {}
            for unit, content in details.items():
                extra_data = {"sub_fault": content.get("sub_fault")} if "sub_fault" in content else {}
                parent_node = self._format_node(content["score"], unit, extra_data)
                if "subs" in content:
                    parent_node["sub_details"] = {sub: self._format_node(s_score, sub) for sub, s_score in content["subs"].items()}
                formatted_details[unit] = parent_node
                
            device_scores[device] = {**self._format_node(score, device), "details": formatted_details}
            mechanical_total += score * weight; weight_total += weight

        weighted_mech_score = (mechanical_total / weight_total) if weight_total > 0 else 100.0
        all_mech_scores = [d["score"] for d in device_scores.values()]
        mech_sys_score = apply_smooth_penalty(weighted_mech_score, min(all_mech_scores) if all_mech_scores else 100.0, 60.0)

        env_scores = {}
        env_data = data.get("环境与电气", {})
        for field in ["temperature", "water", "motor_current", "noise_ratio", "displacement"]:
            if self.config.get("env_checks", {}).get(f"enable_{field.split('_')[0]}", True):
                env_scores[field] = self._format_node(evaluate_env_status(env_data.get(field, 0)))

        min_env_score = min([info["score"] for info in env_scores.values()]) if env_scores else 100.0
        final_system_score = min(mech_sys_score, min_env_score)

        return {"score": round(final_system_score, 2), **self._format_node(final_system_score), "device_scores": device_scores, "env_scores": env_scores}