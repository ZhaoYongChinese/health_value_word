import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

class FinalReportGenerator:
    def __init__(self, config: dict, eval_result: dict):
        self.doc = Document()
        self.config = config
        self.eval_result = eval_result

    def set_font_style(self, run, size_pt=None, bold=False, color=None):
        run.font.name = 'Times New Roman'
        run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
        if size_pt: run.font.size = Pt(size_pt)
        if bold: run.bold = bold
        if color: run.font.color.rgb = color

    def generate(self, worst_device: str, worst_fault_type: str, worst_detail: dict, img_paths: dict, worst_data_info: dict, output_path: str):
        # --- 全局字体 ---
        style = self.doc.styles['Normal']
        style.font.name = 'Times New Roman'
        style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
        style.font.size = Pt(10.5)

        # 标题
        header = self.doc.add_heading('', 0)
        title_text = self.config.get('report_info', {}).get('title', '直梯运行状态智能化诊断报告')
        run = header.add_run(title_text)
        self.set_font_style(run, size_pt=22, bold=True)
        header.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # --- 标题1：直梯整体运行健康度 ---
        h1 = self.doc.add_heading('', level=1)
        self.set_font_style(h1.add_run('一、直梯整体运行健康度'), size_pt=14, bold=True)
        
        sys_score = self.eval_result.get('score', 100)
        sys_grade = self.eval_result.get('grade', 'H1')
        p1 = self.doc.add_paragraph()
        self.set_font_style(p1.add_run(f"系统整体健康得分: {sys_score} 分 | 整体健康等级: "), bold=True)
        color = RGBColor(255, 0, 0) if sys_grade in ['H3', 'H4'] else RGBColor(0, 150, 0)
        self.set_font_style(p1.add_run(sys_grade), bold=True, color=color)

        table = self.doc.add_table(rows=1, cols=4); table.style = 'Table Grid'
        for cell, text in zip(table.rows[0].cells, ['检测部件', '部件得分', '具体故障项', '故障项得分/等级']):
            self.set_font_style(cell.paragraphs[0].add_run(text), bold=True)

        for device, info in self.eval_result.get('device_scores', {}).items():
            for fault_name, detail in info.get('details', {}).items():
                row = table.add_row().cells
                row[0].text = device
                row[1].text = f"{info['score']} ({info.get('grade', '-')})"
                row[2].text = fault_name
                grade_str = f" / {detail['grade']}" if 'grade' in detail else ""
                row[3].text = f"{detail['score']}{grade_str}"

        # --- 标题2：设备环境与采样信息 ---
        h2 = self.doc.add_heading('', level=1)
        self.set_font_style(h2.add_run('二、设备环境与采样信息'), size_pt=14, bold=True)
        
        p2 = self.doc.add_paragraph()
        loc = worst_data_info.get('location', '未知')
        comp = worst_data_info.get('component', '未知')
        p2.add_run(f"本次报告聚焦于得分最低的预警项。该信号采集于位置【{loc}】的【{comp}】部件。")
        
        info_table = self.doc.add_table(rows=3, cols=2); info_table.style = 'Table Grid'
        details = [
            ("触发报警传感器编号", worst_data_info.get('sensor_id', 'N/A')),
            ("数据采样率 (Hz)", str(worst_data_info.get('fs', 'N/A'))),
            ("采集通道数量", str(len(worst_data_info.get('signal_data', []))))
        ]
        for i, (k, v) in enumerate(details):
            info_table.rows[i].cells[0].text = k
            info_table.rows[i].cells[1].text = v

        # --- 标题3：深度诊断结论 ---
        h3 = self.doc.add_heading('', level=1)
        self.set_font_style(h3.add_run('三、深度诊断结论'), size_pt=14, bold=True)
        
        p3 = self.doc.add_paragraph()
        self.set_font_style(p3.add_run("判定结果："), bold=True)
        run_res = p3.add_run(f"【{worst_device} - {worst_fault_type}】健康度异常")
        self.set_font_style(run_res, size_pt=14, bold=True, color=RGBColor(255, 0, 0))
        
        p3_desc = self.doc.add_paragraph()
        p3_desc.add_run(f"经系统计算，该指标当前得分为 {worst_detail.get('score', 'N/A')} 分")
        if 'grade' in worst_detail:
            p3_desc.add_run(f" (风险等级: {worst_detail['grade']})")
        p3_desc.add_run("，为本次巡检中情况最差环节。")

        # --- 标题4：可视化分析验证 ---
        h4 = self.doc.add_heading('', level=1)
        self.set_font_style(h4.add_run('四、可视化分析验证'), size_pt=14, bold=True)
        
        img_titles = [('time', '图1：原始信号时域波形'), ('spectrum', '图2：原始信号频谱图 (FFT)'), ('envelope', '图3：包络解调谱')]
        for key, title in img_titles:
            if key in img_paths and os.path.exists(img_paths[key]):
                self.doc.add_picture(img_paths[key], width=Inches(6.0))
                self.doc.add_paragraph(title).alignment = WD_ALIGN_PARAGRAPH.CENTER

        # --- 标题5：专家评估与维修建议 ---
        h5 = self.doc.add_heading('', level=1)
        self.set_font_style(h5.add_run('五、专家评估与维修建议'), size_pt=14, bold=True)
        
        advice_dict = self.config.get('expert_advice', {})
        target_advice = advice_dict.get(worst_fault_type, advice_dict.get('default', {}))
        grade = worst_detail.get('grade', 'H4') 
        advice_text = target_advice.get(grade, "暂无特定级别建议，请结合现场实际情况排查。")
        
        p5 = self.doc.add_paragraph()
        self.set_font_style(p5.add_run(advice_text))

        # --- 标题6：报告说明 ---
        h6 = self.doc.add_heading('', level=1)
        self.set_font_style(h6.add_run('六、报告说明'), size_pt=14, bold=True)
        p6 = self.doc.add_paragraph()
        p6.add_run("1. 本报告由人工智能系统自动生成，诊断结论基于采集数据的数字模型推导。\n"
                   "2. 实际维护和配件更换作业，请严格结合现场物理检测和专业工程师二次确认。\n"
                   "3. 请妥善保管此报告，作为设备全生命周期管理的依据。")

        self.doc.save(output_path)
        print(f"\n[+] 诊断报告生成完毕，保存至: {output_path}")