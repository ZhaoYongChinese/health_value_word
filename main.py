import os
import glob
import pandas as pd
import numpy as np
import yaml
from loguru import logger

# 引入我们刚才写的模块
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

        # ================= 1. 遍历计算 =================
        for file in csv_files:
            try:
                sensor_id, fault_type, fs, signal_data = self._parse_csv(file)
                mapping = self.config.get('sensor_mapping', {}).get(sensor_id, {})
                device_category = mapping.get('component', '未知设备')
                
                features = extract_features(fault_type, signal_data)
                
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

        # ================= 2. 综合打分 =================
        evaluation_result = self.evaluator.evaluate(aggregated_data)
        logger.info(f"整体健康度评估完成，总得分: {evaluation_result['score']} ({evaluation_result['grade']})")

        # ================= 3. 寻找最差项画图 =================
        worst_score = 101.0  
        worst_device, worst_f_name = "未知", "未知"
        
        for device, info in evaluation_result['device_scores'].items():
            for f_name, detail in info.get('details', {}).items():
                if detail['score'] < worst_score:
                    worst_score = detail['score']
                    worst_device = device
                    worst_f_name = f_name
        
        # 提前把最差的分数详情取出来，直接传给生成器，免去各种KeyError
        worst_detail = evaluation_result['device_scores'].get(worst_device, {}).get('details', {}).get(worst_f_name, {})
        
        fault_type_mapping = {
            "曳引机": worst_f_name, 
            "钢丝绳": "wire_rope",
            "轿厢": "car",
            "导轨": "guide_rail"
        }
        worst_fault_type = fault_type_mapping.get(worst_device, worst_f_name)
        worst_fault_key = f"{worst_device}_{worst_fault_type}"
        
        img_paths = {}
        worst_data_info = {}
        if worst_fault_key in self.raw_data_cache:
            worst_data_info = self.raw_data_cache[worst_fault_key]
            logger.info(f"锁定得分最低项: {worst_fault_key} ({worst_score}分)，生成自适应波形图...")
            
            display_cfg = self.config.get('visual_config', {}).get(worst_fault_type, {})
            metrics_to_display = display_cfg.get('display_metrics', [])
            
            combined_metrics = {**worst_data_info['features'], **worst_detail}
            img_paths = self.visualizer.plot_worst_case(worst_data_info, metrics_to_display, combined_metrics)

        # ================= 4. 组装 Word 报告 =================
        reporter = FinalReportGenerator(self.config, evaluation_result)
        # 将参数拆分开，直接传入明确的 worst_device, worst_fault_type 和 worst_detail
        reporter.generate(worst_device, worst_fault_type, worst_detail, img_paths, worst_data_info, output_report_path)

if __name__ == "__main__":
    pipeline = DiagnosticPipeline("master_config.yml")
    pipeline.run_batch("./data", "./直梯深度诊断报告.docx")