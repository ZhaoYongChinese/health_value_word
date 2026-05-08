import os
import glob
import pandas as pd
import numpy as np
import yaml
import json
from loguru import logger

from signal_features import extract_features
from health_evaluator import HealthEvaluator
from visualizer import SignalVisualizer
from report_generator import FinalReportGenerator

class DiagnosticPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.evaluator = HealthEvaluator(self.config.get('health_evaluation', {}))
        self.visualizer = SignalVisualizer()
        self.raw_data_cache = {} 

    def _parse_csv(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            sensor_id = f.readline().strip().split(',')[0]
            fault_type = f.readline().strip().split(',')[0]
            fs = float(f.readline().strip().split(',')[0])
            channels_line = f.readline().strip().split(',')
            channels = [ch.strip() for ch in channels_line if ch.strip()]
            
        df = pd.read_csv(file_path, skiprows=4, header=None)
        
        data_dict = {}
        for i, ch in enumerate(channels):
            col_data = df.iloc[:, i].values
            col_data = col_data - np.mean(col_data)
            if ch.lower() != 'none':
                data_dict[ch] = col_data
            else:
                data_dict[f'Channel_{i+1}'] = col_data
                
        return sensor_id, fault_type, fs, data_dict

    def run_batch(self, data_folder: str, output_report_path: str):
        logger.info(f"开始批量诊断，扫描文件夹: {data_folder}")
        
        aggregated_data = {"曳引机": {}, "钢丝绳": {}, "导轨": {}, "轿厢": {}}
        csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
        
        if not csv_files:
            logger.error(f"在 {data_folder} 中未找到 CSV 文件！")
            return

        # ================= 1. 遍历计算波形数据 =================
        for file in csv_files:
            try:
                sensor_id, fault_type, fs, signal_data = self._parse_csv(file)
                mapping = self.config.get('sensor_mapping', {}).get(sensor_id, {})
                device_category = mapping.get('component', '未知设备')
                
                features = extract_features(fault_type, signal_data, fs)
                
                if device_category == "曳引机":
                    aggregated_data[device_category][fault_type] = features
                elif device_category in aggregated_data:
                    aggregated_data[device_category].update(features)
                
                cache_key = f"{device_category}_{fault_type}"
                self.raw_data_cache[cache_key] = {
                    "signal_data": signal_data, "fs": fs,
                    "sensor_id": sensor_id, "location": mapping.get('location', '未知'),
                    "component": device_category, "features": features
                }
                logger.info(f"成功处理文件: {os.path.basename(file)} -> 映射为: {device_category} - {fault_type}")
            except Exception as e:
                logger.error(f"处理文件 {file} 失败: {e}")

        # --- 注入环境与电气数据 (演示逻辑：在实际场景中可能来自于API或其他传感器) ---
        env_test_file = os.path.join(data_folder, "env_data.json")
        if os.path.exists(env_test_file):
            with open(env_test_file, 'r', encoding='utf-8') as f:
                aggregated_data["环境与电气"] = json.load(f)
        else:
            # 没有独立文件则模拟环境恶化 (供测试使用)
            aggregated_data["环境与电气"] = {"temperature": 35.0, "water": 0, "displacement": 8.0}

        # ================= 2. 综合打分 =================
        evaluation_result = self.evaluator.evaluate(aggregated_data)
        logger.info(f"整体健康度评估完成，总得分: {evaluation_result['score']} ({evaluation_result['grade']})")

        # ================= 3. 寻找最差项 (兼顾机械本体与环境) =================
        worst_score = 101.0  
        worst_device, worst_f_name = "未知", "未知"
        worst_detail = {}
        is_environment_issue = False
        
        # 3.1 遍历机械项
        for device, info in evaluation_result.get('device_scores', {}).items():
            for f_name, detail in info.get('details', {}).items():
                if detail['score'] < worst_score:
                    worst_score = detail['score']
                    worst_device = device
                    worst_f_name = f_name
                    worst_detail = detail
        
        # 3.2 遍历环境项比对短板
        for env_name, info in evaluation_result.get('env_scores', {}).items():
            if info['score'] < worst_score:
                worst_score = info['score']
                worst_device = "环境与电气"
                worst_f_name = env_name
                worst_detail = info
                is_environment_issue = True
        
        img_paths = {}
        worst_data_info = {}
        
        # 只有机械问题且在缓存中有波形时才画图
        if not is_environment_issue:
            fault_type_mapping = {"曳引机": worst_f_name, "钢丝绳": "wire_rope", "轿厢": "car", "导轨": "guide_rail"}
            worst_fault_type = fault_type_mapping.get(worst_device, worst_f_name)
            worst_fault_key = f"{worst_device}_{worst_fault_type}"
            
            if worst_fault_key in self.raw_data_cache:
                worst_data_info = self.raw_data_cache[worst_fault_key]
                logger.info(f"锁定得分最低项: {worst_fault_key} ({worst_score}分)，生成自适应波形图...")
                
                display_cfg = self.config.get('visual_config', {}).get(worst_fault_type, {})
                metrics_to_display = display_cfg.get('display_metrics', [])
                
                combined_metrics = {**worst_data_info['features'], **worst_detail}
                img_paths = self.visualizer.plot_worst_case(worst_data_info, metrics_to_display, combined_metrics)
        else:
            logger.warning(f"最差项为环境指标 [{worst_f_name}]，跳过波形图生成。")
            worst_data_info = {"location": "机房/底坑/井道", "component": "环境与电气传感系统", "sensor_id": "N/A", "fs": "N/A", "signal_data": []}

        # ================= 4. 组装 Word 报告 =================
        reporter = FinalReportGenerator(self.config, evaluation_result)
        reporter.generate(worst_device, worst_f_name, worst_detail, img_paths, worst_data_info, output_report_path)

if __name__ == "__main__":
    pipeline = DiagnosticPipeline("master_config.yml")
    pipeline.run_batch("./data", "./直梯深度诊断报告.docx")