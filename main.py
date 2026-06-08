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
from llm_advisor import LLMAdvisor

class DiagnosticPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.evaluator = HealthEvaluator(self.config.get('health_evaluation', {}))
        self.visualizer = None
        self.raw_data_cache = {}
        self.llm_advisor = LLMAdvisor(self.config)

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
        
        # 提取基准线、电机参数与RMS滑窗过滤配置
        signal_baselines = self.config.get('signal_baselines', {})
        motor_params = self.config.get('motor_params', {})
        rms_params = self.config.get('rms_window', {})

        for file in csv_files:
            try:
                sensor_id, fault_type, fs, signal_data = self._parse_csv(file)
                mapping = self.config.get('sensor_mapping', {}).get(sensor_id, {})
                device_category = mapping.get('component', '未知设备')
                
                bearing_params = mapping.get('bearing_params')
                # 传入配置的阈值及滑窗过滤参数
                features = extract_features(
                    fault_type, signal_data, fs, bearing_params, 
                    baselines=signal_baselines, 
                    motor_params=motor_params,
                    rms_params=rms_params
                )
                
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
            with open(env_file, 'r', encoding='utf-8') as f:
                env_raw = json.load(f)
                if "环境与电气" in env_raw:
                    aggregated_data["环境与电气"] = env_raw["环境与电气"]
                elif "input_data" in env_raw and "环境与电气" in env_raw["input_data"]:
                    aggregated_data["环境与电气"] = env_raw["input_data"]["环境与电气"]
                else:
                    aggregated_data["环境与电气"] = env_raw
        else:
            logger.warning(f"未找到 env_data.json，将使用系统无故障模拟环境数据 (全0正常状态)。")
            aggregated_data["环境与电气"] = {
                "temperature": 0, 
                "water": 0, 
                "motor_current": 0,
                "noise_ratio": 0,
                "displacement": 0
            }

        eval_res = self.evaluator.evaluate(aggregated_data)
        logger.info(f"整体健康评估完毕 | 总分: {eval_res['score']} | 分布: {eval_res['fuzzy_distribution']}")

        # -------------------------------------------------------------------
        # [修改点1]：改为列表收集机制，支持多故障源并发收集
        # -------------------------------------------------------------------
        report_targets = []
        lowest_score = 90.0
        lowest_item = None
        
        # 1. 先排查机械类故障
        for device, info in eval_res.get('device_scores', {}).items():
            for f_name, detail in info.get('details', {}).items():
                if not isinstance(detail, dict) or 'score' not in detail:
                    continue
                score = detail.get('score', 100)
                fuzzy = detail.get('fuzzy_distribution', {})
                
                # 若为高危项（H4），则直接进入分析列表
                if 'H4' in fuzzy or detail.get('crisp_grade') == 'H4':
                    report_targets.append({"device": device, "fault_name": f_name, "detail": detail})
                
                # 追踪当前全场最低分，作为备选方案
                if score < lowest_score:
                    lowest_score = score
                    lowest_item = {"device": device, "fault_name": f_name, "detail": detail}
        
        # 2. 再排查环境风险
        for env_name, info in eval_res.get('env_scores', {}).items():
            score = info.get('score', 100)
            fuzzy = info.get('fuzzy_distribution', {})
            if 'H4' in fuzzy or info.get('crisp_grade') == 'H4':
                report_targets.append({"device": "环境与电气", "fault_name": env_name, "detail": info})
            if score < lowest_score:
                lowest_score = score
                lowest_item = {"device": "环境与电气", "fault_name": env_name, "detail": info}

        # 3. 汇总判定：如果没有H4，但有低于90分的异常，则抓取最低分作为分析对象
        if not report_targets and lowest_item:
            report_targets.append(lowest_item)

        # -------------------------------------------------------------------
        # [修改点2]：生成所有报警项的详细数据结构与可视化图像
        # -------------------------------------------------------------------
        fault_cases = []
        if not report_targets:
            logger.info("系统整体运行优良，未发现明显风险隐患，无需生成异常波形图。")
        else:
            for target in report_targets:
                device = target["device"]
                f_name = target["fault_name"]
                detail = target["detail"]
                is_env = (device == "环境与电气")

                img_paths, data_info = {}, {}
                if not is_env:
                    actual_fault_name = "bearing_fault" if f_name.startswith("bearing_") else f_name
                    cache_key = ""
                    # 防止同设备多故障(如曳引机同时发生motor和bearing故障)匹配混乱
                    for key in self.raw_data_cache:
                        if device in key and actual_fault_name in key:
                            cache_key = key
                            break
                    if not cache_key:
                        for key in self.raw_data_cache:
                            if device in key:
                                cache_key = key; break

                    if cache_key and cache_key in self.raw_data_cache:
                        data_info = self.raw_data_cache[cache_key]
                        logger.info(f"提取预警项波形: [{device}-{f_name}]，正在生成验证视图...")
                        
                        vis_key = 'car' if device == '轿厢' else actual_fault_name
                        display_cfg = self.config.get('visual_config', {}).get(vis_key, {})
                        combined_metrics = {**data_info['features'], **detail}
                        
                        # 传入前缀，确保多故障生成的图片文件名不会互相覆盖
                        prefix_name = f"{device}_{f_name}"
                        img_paths = self.visualizer.plot_worst_case(
                            data_info, display_cfg.get('display_metrics', []), combined_metrics, prefix=prefix_name
                        )
                    else:
                        logger.warning(f"未能从波形缓存中定位到相关数据 (设备:{device})")
                else:
                    logger.warning(f"分析环境灾变风险 [{f_name}]，跳过波形生成。")
                    env_map = self.config.get('sensor_mapping', {}).get('ENV_001', {})
                    data_info = {
                        'sensor_id': 'ENV_001 (环境探头)',
                        'location': env_map.get('location', '全局机房底坑环境'),
                        'component': env_map.get('component', '环境与电气'),
                        'fs': 'N/A',
                        'signal_data': []
                    }
                
                fault_cases.append({
                    "device": device,
                    "fault_name": f_name,
                    "detail": detail,
                    "data_info": data_info,
                    "img_paths": img_paths
                })

        # -------------------------------------------------------------------
        # [修改点3]：将收集好的列表整体传入报表生成器
        # -------------------------------------------------------------------
        reporter = FinalReportGenerator(self.config, eval_res, llm_advisor=self.llm_advisor)
        reporter.generate(fault_cases, output_report_path)

if __name__ == "__main__":
    pipeline = DiagnosticPipeline("master_config.yml")
    pipeline.run_batch("./data/test")