import os
import sys
import glob
import pandas as pd
import numpy as np
import yaml
import json
import tkinter as tk
from tkinter import filedialog
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
        self.visualizer = None 
        self.raw_data_cache = {} 

    def _parse_csv(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            sensor_id, fault_type, fs = [f.readline().strip().split(',')[0] for _ in range(3)]
            channels = [ch.strip() for ch in f.readline().strip().split(',') if ch.strip()]
            
        df = pd.read_csv(file_path, skiprows=4, header=None)
        
        data_dict = {}
        for i, ch in enumerate(channels):
            col_data = df.iloc[:, i].values
            col_data = col_data - np.mean(col_data)
            data_dict[ch if ch.lower() != 'none' else f'Channel_{i+1}'] = col_data
        return sensor_id, fault_type, float(fs), data_dict

    def _get_risk_score(self, detail: dict) -> tuple:
        fuzzy = detail.get('fuzzy_distribution', {})
        p_h4, p_h3 = fuzzy.get('H4', 0.0), fuzzy.get('H3', 0.0)
        return (p_h4, p_h3, -detail.get('score', 100))

    def run_batch(self, default_data_folder="./data/test"):
        if os.path.exists(default_data_folder):
            data_folder = default_data_folder
            logger.info(f"检测到默认数据批次文件夹: {data_folder}")
        else:
            logger.info("未检测到默认文件夹，启动GUI选择向导...")
            root = tk.Tk()
            root.withdraw()
            data_folder = filedialog.askdirectory(title="请选择检测批次文件夹")
            if not data_folder:
                logger.error("操作取消：未选择任何文件夹。程序退出。")
                sys.exit(0)
                
        folder_name = os.path.basename(os.path.normpath(data_folder))
        output_dir = os.path.join("results", f"{folder_name}_输出报告")
        if not os.path.exists(output_dir): os.makedirs(output_dir)
            
        output_report_path = os.path.join(output_dir, f"{folder_name}_智能诊断报告.docx")
        self.visualizer = SignalVisualizer(output_dir=output_dir)

        logger.info(f"开始批量智能诊断，扫描文件夹: {data_folder}")
        aggregated_data = {"曳引机": {}, "钢丝绳": {}, "导轨": {}, "轿厢": {}}
        csv_files = glob.glob(os.path.join(data_folder, "*.csv"))

        for file in csv_files:
            try:
                sensor_id, fault_type, fs, signal_data = self._parse_csv(file)
                mapping = self.config.get('sensor_mapping', {}).get(sensor_id, {})
                device_category = mapping.get('component', '未知设备')
                
                bearing_params = mapping.get('bearing_params')
                features = extract_features(fault_type, signal_data, fs, bearing_params)
                
                # [修复致命漏洞A]：解开数据嵌套，让导轨、钢丝绳等一层级设备不再被套成两层
                if device_category == "曳引机":
                    aggregated_data[device_category][fault_type] = features
                elif device_category in aggregated_data:
                    aggregated_data[device_category].update(features)
                
                self.raw_data_cache[f"{device_category}_{fault_type}"] = {
                    "signal_data": signal_data, "fs": fs,
                    "sensor_id": sensor_id, "location": mapping.get('location', '未知'),
                    "component": device_category, "features": features
                }
            except Exception as e:
                logger.error(f"处理文件 {file} 失败: {e}")

        env_file = os.path.join(data_folder, "env_data.json")
        if os.path.exists(env_file):
            with open(env_file, 'r', encoding='utf-8') as f: aggregated_data["环境与电气"] = json.load(f)
        else:
            logger.warning(f"未找到 env_data.json，将使用系统模拟环境数据。")
            aggregated_data["环境与电气"] = {"temperature": 35.0, "water": 0, "displacement": 8.0}

        eval_res = self.evaluator.evaluate(aggregated_data)
        logger.info(f"整体健康评估完毕 | 总分: {eval_res['score']} | 分布: {eval_res['fuzzy_distribution']}")

        worst_risk = (-1.0, -1.0, 0.0)
        worst_device, worst_f_name, worst_detail = "未知", "未知", {}
        is_environment_issue = False
        
        # [修复漏洞2]：先排查机械的最差报警，作为默认主角（为了有图有真相）
        for device, info in eval_res.get('device_scores', {}).items():
            for f_name, detail in info.get('details', {}).items():
                risk = self._get_risk_score(detail)
                if risk > worst_risk:
                    worst_risk, worst_device, worst_f_name, worst_detail = risk, device, f_name, detail
        
        # [修复漏洞2]：再排查环境风险。只有环境风险绝对超越(>)机械最差风险时，才将主角让位给环境。
        # 即使环境是H4，如果机械也是H4，优先报机械以便给维保看波形。
        for env_name, info in eval_res.get('env_scores', {}).items():
            risk = self._get_risk_score(info)
            if risk > worst_risk:
                worst_risk, worst_device, worst_f_name, worst_detail = risk, "环境与电气", env_name, info
                is_environment_issue = True

        img_paths, worst_data_info = {}, {}
        if not is_environment_issue:
            # [修复隐蔽漏洞C]：废除写死的字典映射，动态遍历 cache 寻找相符的原始波形钥匙
            actual_fault_name = "bearing_fault" if worst_f_name.startswith("bearing_") else worst_f_name
            cache_key = ""
            for key in self.raw_data_cache:
                if worst_device in key: # 只要设备名字匹配上，模糊抓取波形，防止重命名丢失
                    cache_key = key
                    break

            if cache_key and cache_key in self.raw_data_cache:
                worst_data_info = self.raw_data_cache[cache_key]
                logger.info(f"定位核心风险点: [{cache_key}], 正在生成带寻峰标记的频谱图...")
                
                # [适配轿厢新逻辑]：如果是轿厢，直接取 car 的绘图配置
                vis_key = 'car' if worst_device == '轿厢' else actual_fault_name
                display_cfg = self.config.get('visual_config', {}).get(vis_key, {})
                combined_metrics = {**worst_data_info['features'], **worst_detail}
                img_paths = self.visualizer.plot_worst_case(worst_data_info, display_cfg.get('display_metrics', []), combined_metrics)
            else:
                logger.warning(f"未能从波形缓存中定位到相关数据 (设备:{worst_device})")
        else:
            logger.warning(f"最大风险源于环境/灾变 [{worst_f_name}]，跳过波形生成。")
            # [修复漏洞1]：在报告为环境报警时，为其手工塞入一个虚拟的数据源，以防位置信息渲染为'未知'
            env_map = self.config.get('sensor_mapping', {}).get('ENV_001', {})
            worst_data_info = {
                'sensor_id': 'ENV_001 (环境探头)',
                'location': env_map.get('location', '全局机房底坑环境'),
                'component': env_map.get('component', '环境与电气'),
                'fs': 'N/A',
                'signal_data': []
            }

        reporter = FinalReportGenerator(self.config, eval_res)
        reporter.generate(worst_device, worst_f_name, worst_detail, img_paths, worst_data_info, output_report_path)

if __name__ == "__main__":
    pipeline = DiagnosticPipeline("master_config.yml")
    pipeline.run_batch("./data/test")