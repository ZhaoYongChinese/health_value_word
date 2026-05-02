# health_evaluator.py
"""
健康度评估模块
- 曳引机: motor_fault, bearing_fault, bolt_loose (带专属 H 等级)
- 钢丝绳: 张力不均, 打滑, 磨损、断裂 (仅有分数)
- 导轨: 磨损情况 (设备级带 H 等级)
- 轿厢: 轿架振动, 运行状态异常, 平稳度异常 (仅有分数)
- 支持外部 YAML 配置文件, 为不同故障和设备实现独立的 H 等级评价区间
- 支持日志文件输出分离
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

# ------------------------------- 工具函数 ---------------------------------
def linear_score(value: float, in_min: float, in_max: float,
                 out_min: float = 100.0, out_max: float = 0.0) -> float:
    """线性映射并限幅到 [0, 100]"""
    if in_max == in_min:
        return out_min
    ratio = (value - in_min) / (in_max - in_min)
    score = out_min + ratio * (out_max - out_min)
    return max(0.0, min(100.0, score))

def segmented_score(value: float, segments: List[Dict]) -> float:
    """分段线性插值计算得分，区间左闭右开"""
    for seg in segments:
        low, high = seg["range"]
        if low <= value < high:
            s_low, s_high = seg["score"]
            return linear_score(value, low, high, s_low, s_high)
    return 0.0

# --------------------------- 单设备评估函数 -------------------------------
def evaluate_traction(data: Dict, config: Dict) -> Tuple[float, Dict[str, float]]:
    scores = {}

    motor = data.get("motor_fault", {})
    rms = motor.get("rms_ratio", 1.0)
    conf = motor.get("confidence", 0.0)
    if rms > 1.0 and conf > 0:
        penalty = (rms - 1.0) * config.get("rms_penalty_per_unit", 40) * conf * config.get("confidence_factor", 1.0)
        scores["motor_fault"] = max(0.0, 100.0 - penalty)
    else:
        scores["motor_fault"] = 100.0

    bearing = data.get("bearing_fault", {})
    rms = bearing.get("rms_ratio", 1.0)
    conf = bearing.get("confidence", 0.0)
    if rms > 1.0 and conf > 0:
        penalty = (rms - 1.0) * config.get("rms_penalty_per_unit", 40) * conf * config.get("confidence_factor", 1.0)
        scores["bearing_fault"] = max(0.0, 100.0 - penalty)
    else:
        scores["bearing_fault"] = 100.0

    bolt = data.get("bolt_loose", {})
    bolt_rms = bolt.get("rms_ratio", 1.0)
    if bolt_rms > 1.0:
        penalty = (bolt_rms - 1.0) * config.get("bolt_penalty_per_unit", 20)
        scores["bolt_loose"] = max(0.0, 100.0 - penalty)
    else:
        scores["bolt_loose"] = 100.0

    weights = config.get("fault_weights", {})
    safety_baseline = config.get("safety_baseline", 60.0)

    total_score = 0.0
    total_weight = 0.0
    for fault_type, score in scores.items():
        weight = weights.get(fault_type, 1.0)
        total_score += score * weight
        total_weight += weight

    weighted_score = (total_score / total_weight) if total_weight > 0 else 100.0
    min_fault_score = min(scores.values()) if scores else 100.0

    if min_fault_score < safety_baseline:
        final_score = min(min_fault_score, weighted_score)
    else:
        final_score = weighted_score
        
    return final_score, scores

def evaluate_wire_rope(data: Dict, config: Dict) -> Tuple[float, Dict[str, float]]:
    ratio = data.get("rms_baseline_ratio", 1.0)
    score = segmented_score(ratio, config["segments"])
    # 如果将来有打滑、张力不均的数据，可以在这里生成字典输出
    # 只要 YAML 不配对应等级，就只显示分数
    return score, {"rms_baseline_ratio": score} 

def evaluate_guide_rail(data: Dict, config: Dict) -> Tuple[float, Dict[str, float]]:
    wear = data.get("wear_percent", 0.0)
    score = linear_score(wear, 0, config["max_wear"], 100.0, 0.0)
    return score, {"wear_percent": score}

def evaluate_car(data: Dict, config: Dict) -> Tuple[float, Dict[str, float]]:
    directions = data.get("directions", [])
    if not directions:
        return 100.0, {}

    vibration_th = config["vibration_threshold"]
    smoothness_th = config["smoothness_threshold"]
    slope = config["base_penalty_slope"]

    vibration_count = 0
    smoothness_count = 0
    all_max_ratios = []

    for d in directions:
        indicators = [
            d.get("crest_factor_ratio", 1.0),
            d.get("impulse_factor_ratio", 1.0),
            d.get("margin_factor_ratio", 1.0)
        ]
        if sum(1 for v in indicators if v >= vibration_th) >= 2:
            vibration_count += 1
        elif sum(1 for v in indicators if v >= smoothness_th) >= 2:
            smoothness_count += 1
        all_max_ratios.append(max(indicators))

    details = {}
    if vibration_count >= 2:
        return config["vibration_penalty_score"], {"vibration_issue": config["vibration_penalty_score"]}
    if smoothness_count >= 2:
        return config["smoothness_penalty_score"], {"smoothness_issue": config["smoothness_penalty_score"]}

    avg_max = sum(all_max_ratios) / len(all_max_ratios)
    score = max(0.0, 100.0 - (avg_max - 1.0) * slope)
    return score, {"avg_vibration": score}


# ------------------------------- 核心类 ---------------------------------
class HealthEvaluator:
    def __init__(self, config: Optional[Union[str, Dict]] = None):
        self.default_config = {
            "device_weights": {"曳引机": 0.4, "钢丝绳": 0.2, "导轨": 0.15, "轿厢": 0.25},
            "default_grade_ranges": {"H1": [90, 101], "H2": [75, 90], "H3": [60, 75], "H4": [0, 60]},
            "system_safety_baseline": 60.0,
            "scoring_params": {}
        }
        self.config = self._load_config(config)

    def _load_config(self, config: Union[str, Dict, None]) -> Dict:
        base = self.default_config.copy()
        if config is None:
            return base
            
        user_config = {}
        if isinstance(config, str) and os.path.exists(config):
            if config.endswith(('.yml', '.yaml')):
                with open(config, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f) or {}
            elif config.endswith('.json'):
                with open(config, 'r', encoding='utf-8') as f:
                    user_config = json.load(f) or {}
        elif isinstance(config, dict):
            user_config = config
        else:
            return base
            
        # 深度合并字典，确保配置无缝接入
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    deep_update(d[k], v)
                else:
                    d[k] = v
                    
        deep_update(base, user_config)
        return base

    def _get_grade(self, score: float, custom_ranges: Optional[Dict] = None) -> str:
        """根据给定的配置范围（或全局范围）判定等级"""
        ranges = custom_ranges if custom_ranges else self.config.get("default_grade_ranges", {})
        for grade, (low, high) in ranges.items():
            if low <= score < high:
                return grade
        return "H4"  # 默认兜底极差情况

    def evaluate(self, data: Dict) -> Dict:
        device_scores = {}
        weights = self.config.get("device_weights", {})
        scoring = self.config.get("scoring_params", {})

        evaluators = {
            "曳引机": (evaluate_traction, scoring.get("traction", {})),
            "钢丝绳": (evaluate_wire_rope, scoring.get("wire_rope", {})),
            "钢带":   (evaluate_wire_rope, scoring.get("wire_rope", {})),
            "导轨":   (evaluate_guide_rail, scoring.get("guide_rail", {})),
            "轿厢":   (evaluate_car, scoring.get("car", {}))
        }

        total_score = 0.0
        total_weight = 0.0

        # 1. 计算所有设备级得分及最小单元细节
        for device, weight in weights.items():
            if device not in data:
                continue
            
            if device in evaluators:
                func, params = evaluators[device]
                score, details = func(data[device], params)
            else:
                score, details = 100.0, {}
                params = {}
                
            # 计算设备级等级 (优先使用 params 中的专属范围)
            device_grade = self._get_grade(score, params.get("grade_ranges"))
            
            # 格式化子单元详情
            formatted_details = {}
            fault_grade_ranges = params.get("fault_grade_ranges", {})
            
            for sub_unit, sub_score in details.items():
                detail_info = {"score": round(sub_score, 2)}
                # 若配置中存在该子项的独立等级划分范围，则计算其H等级；否则（如钢丝绳/轿厢等）仅保留得分
                if sub_unit in fault_grade_ranges:
                    detail_info["grade"] = self._get_grade(sub_score, fault_grade_ranges[sub_unit])
                    
                formatted_details[sub_unit] = detail_info
                
            device_scores[device] = {
                "score": round(score, 2),
                "grade": device_grade,
                "details": formatted_details
            }
            
            total_score += score * weight
            total_weight += weight

        # 2. 系统级总体得分处理
        weighted_sys_score = (total_score / total_weight) if total_weight > 0 else 100.0
        sys_safety_baseline = self.config.get("system_safety_baseline", 60.0)
        
        scores_list = [info["score"] for info in device_scores.values()]
        min_device_score = min(scores_list) if scores_list else 100.0

        if min_device_score < sys_safety_baseline:
            final_score = min(min_device_score, weighted_sys_score)
        else:
            final_score = weighted_sys_score

        # 3. 获取系统总等级
        sys_grade = self._get_grade(final_score, self.config.get("default_grade_ranges"))

        return {
            "score": round(final_score, 2),
            "grade": sys_grade,
            "device_scores": device_scores
        }

# ------------------------------- 示例执行 ---------------------------------
if __name__ == "__main__":
    import shutil
    
    # 支持加载外部输入数据
    if os.path.exists("input_data.json"):
        with open("input_data.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            input_data = raw_data.get("input_data", raw_data)
        
        # 实例化并评估 (使用新的 yml 配置文件)
        evaluator = HealthEvaluator("health_config.yml")
        result = evaluator.evaluate(input_data)
        
        # ---------------- 改进 3: 控制台精简输出与 TXT 生成 ----------------
        
        # 收集达到 H4 的设备和子故障
        h4_items = []
        for device, info in result['device_scores'].items():
            if info.get('grade') == 'H4':
                h4_items.append(f"[{device}]设备异常")
            for sub_unit, sub_info in info.get('details', {}).items():
                if sub_info.get('grade') == 'H4':
                    h4_items.append(f"[{device}-{sub_unit}]故障报警")

        print("\n" + "="*35)
        print("    电梯健康度评估摘要 (控制台)")
        print("="*35)
        print(f"  > 整体健康度: \t{result['score']}")
        print(f"  > 整体健康度等级: \t{result['grade']}")
        
        if h4_items:
            print(f"  > 达到 H4 的风险项: \t{', '.join(h4_items)}")
        else:
            print("  > 达到 H4 的风险项: \t无")
        print("="*35)
        
        # 准备生成 txt 详单
        now = datetime.datetime.now()
        # 格式：年_月_日_时分_分数 (比如：2026_04_30_1525_82.0.txt)
        time_str = now.strftime("%Y_%m_%d_%H%M")
        file_name = f"{time_str}_{result['score']}.txt"
        
        health_dir = "health"
        if not os.path.exists(health_dir):
            os.makedirs(health_dir)
            
        file_path = os.path.join(health_dir, file_name)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("="*40 + "\n")
            f.write("          电梯健康度评估详细报告\n")
            f.write("="*40 + "\n")
            f.write(f"评估时间: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"整体健康度: {result['score']}\n")
            f.write(f"整体健康等级: {result['grade']}\n")
            f.write("-"*40 + "\n")
            f.write("各部件评分详情:\n")
            for device, info in result['device_scores'].items():
                f.write(f"\n【{device}】\n")
                f.write(f"  - 设备综合得分: {info['score']} \t| 设备等级: {info.get('grade', 'N/A')}\n")
                
                if info.get('details'):
                    for sub_unit, sub_info in info['details'].items():
                        # 若有等级则格式化拼接，若无则只打印分数
                        grade_str = f" \t| 评估等级: {sub_info['grade']}" if 'grade' in sub_info else ""
                        f.write(f"    * 故障指标: {sub_unit:<15} | 指标得分: {sub_info['score']}{grade_str}\n")
            f.write("\n" + "="*40 + "\n")
            
        print(f"\n详细报告已生成并保存至目录: {file_path}\n")