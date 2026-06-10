"""
LLM-based expert advice generator for elevator diagnostic reports.
Uses Qwen2.5-0.5B-Instruct via HuggingFace transformers for CPU inference.
Falls back to static YAML advice on any failure.
"""

import os
import random

from loguru import logger

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "transformers/torch not installed. LLM advice disabled, using static YAML fallback."
    )


class LLMAdvisor:
    # Core requirements that MUST appear verbatim in generated advice.
    # (fault_type, risk_grade) -> mandatory core action text
    # H4 = emergency action, H3 = scheduled repair, H2 = monitoring, H1 = routine
    # Keep in sync with master_config.yml expert_advice section.
    # For H4 validation: key terms that must appear (not the full phrase order)
    H4_KEY_TERMS = {
        'bearing_fault': ["更换轴承"],
        'motor_fault':   [["停机", "停止"], "激光对中"],
        'bearing_cage':  [["停机", "停止"], "大修"],
        'bolt_loose':    [["禁行", "停止"], "锁紧"],
        'wire_rope':     [["停梯", "停止"], "探伤"],
        'guide_rail':    ["更换", "导靴"],
        'frame_vibration':  [["停梯", "停止"], "排查轿架"],
        'smoothness':       [["停机", "停止"], "导靴", "导轨"],
        'rotor_misalignment': [["停机", "停止"], "对中"],
        'stator_eccentricity': [["停机", "停止"], "大修"],
        'motor_current':  [["停机", "停止"], "控制柜"],
        'noise_ratio':    [["停梯", "停止"]],
        'water':         ["拉闸", "断电"],
        'temperature':   ["空调", ["排风", "通风"]],
        'displacement':  [["停梯", "停止"], "结构"],
        'default':       ["停梯", "全面排查"],
    }

    # Patterns that are semantically wrong and must be rejected
    FORBIDDEN_PATTERNS = [
        "禁止禁行",   # double negation = allow operation (wrong)
        "禁止停止",   # double negation
        "禁止停机",   # double negation
        "禁止停梯",   # double negation
        "不建议停",   # should be imperative for H4
    ]

    CORE_REQUIREMENTS = {
        # --- 曳引机电机 ---
        ('motor_fault',   'H1'): "保持常规巡检",
        ('motor_fault',   'H2'): "缩短巡检周期，检查润滑状态",
        ('motor_fault',   'H3'): "停梯检查固定底座及同心度",
        ('motor_fault',   'H4'): "立即停机进行激光对中校准",
        # --- 转子不对中 ---
        ('rotor_misalignment', 'H1'): "电机定转子同心度良好",
        ('rotor_misalignment', 'H2'): "检查联轴器或底座",
        ('rotor_misalignment', 'H3'): "安排对中校准",
        ('rotor_misalignment', 'H4'): "立即停机使用激光对中仪重新校准",
        # --- 定子偏心 ---
        ('stator_eccentricity', 'H1'): "电机气隙均匀",
        ('stator_eccentricity', 'H2'): "保持观察",
        ('stator_eccentricity', 'H3'): "检查定子紧固状态",
        ('stator_eccentricity', 'H4'): "立即断电停机大修",
        # --- 轴承（内外圈/滚动体） ---
        ('bearing_fault', 'H1'): "保持常规巡检",
        ('bearing_fault', 'H2'): "补充润滑脂",
        ('bearing_fault', 'H3'): "准备备件并密切监控",
        ('bearing_fault', 'H4'): "立即开盖更换轴承",
        # --- 轴承保持架 ---
        ('bearing_cage',  'H1'): "保持常规监测",
        ('bearing_cage',  'H2'): "维持观察",
        ('bearing_cage',  'H3'): "密切关注保持架状态",
        ('bearing_cage',  'H4'): "立即停机大修",
        # --- 底座螺栓松动 ---
        ('bolt_loose',    'H1'): "保持常规巡检",
        ('bolt_loose',    'H2'): "常规巡检观察",
        ('bolt_loose',    'H3'): "安排力矩扳手复紧作业",
        ('bolt_loose',    'H4'): "必须立即禁行并重新锁紧",
        # --- 钢丝绳 ---
        ('wire_rope',     'H1'): "保持常规巡检",
        ('wire_rope',     'H2'): "维持常规润滑",
        ('wire_rope',     'H3'): "做测力平衡",
        ('wire_rope',     'H4'): "必须立刻停梯进行探伤检测",
        # --- 导轨 ---
        ('guide_rail',    'H1'): "导轨表面平整磨损率极低",
        ('guide_rail',    'H2'): "定期润滑",
        ('guide_rail',    'H3'): "加强润滑列入观察计划",
        ('guide_rail',    'H4'): "校轨更换导靴",
        # --- 轿架振动 ---
        ('frame_vibration', 'H1'): "轿架运行平稳无明显冲击",
        ('frame_vibration', 'H2'): "保持常规维护",
        ('frame_vibration', 'H3'): "检查导靴间隙及曳引钢丝绳张力平衡",
        ('frame_vibration', 'H4'): "立刻停梯全面排查轿架机械结构",
        # --- 平稳度异常 ---
        ('smoothness',     'H1'): "轿厢水平运行平稳",
        ('smoothness',     'H2'): "属于正常范围",
        ('smoothness',     'H3'): "安排校轨",
        ('smoothness',     'H4'): "立即停机检查导靴与导轨磨损",
        # --- 环境 ---
        ('water',         'H1'): "环境干燥无积水隐患",
        ('water',         'H2'): "检查通风或除湿设备",
        ('water',         'H3'): "排查漏水点",
        ('water',         'H4'): "立即拉闸断电",
        ('temperature',   'H1'): "温度在理想工作范围",
        ('temperature',   'H2'): "保持观察",
        ('temperature',   'H3'): "开启空调增强排风散热",
        ('temperature',   'H4'): "立即检查机房空调与排风设备",
        ('displacement',  'H1'): "结构稳定无异常位移",
        ('displacement',  'H2'): "定期校准",
        ('displacement',  'H3'): "安排工程复测",
        ('displacement',  'H4'): "必须立刻全面停梯测量结构沉降",
        # --- 电流异常 ---
        ('motor_current', 'H1'): "电流输出平稳无异常波动",
        ('motor_current', 'H2'): "属正常现象",
        ('motor_current', 'H3'): "排查抱闸及电气接触",
        ('motor_current', 'H4'): "立即停机检查控制柜与电气回路",
        # --- 异常噪声 ---
        ('noise_ratio',   'H1'): "声学环境正常无异常噪音",
        ('noise_ratio',   'H2'): "下次保养时关注",
        ('noise_ratio',   'H3'): "安排维保人员到场确认",
        ('noise_ratio',   'H4'): "立刻停梯",
        # --- 兜底 ---
        ('default',       'H1'): "按计划维保",
        ('default',       'H2'): "保持观察",
        ('default',       'H3'): "现场复核",
        ('default',       'H4'): "立刻停梯安排人工全面排查",
    }

    FAULT_TYPE_NAMES = {
        'motor_fault':   '曳引机电机故障',
        'rotor_misalignment': '电机转子不对中',
        'stator_eccentricity': '电机定子偏心',
        'bearing_fault': '轴承故障（内外圈/滚动体）',
        'bearing_cage':  '轴承保持架故障',
        'bolt_loose':    '曳引机底座螺栓松动',
        'wire_rope':     '钢丝绳故障',
        'guide_rail':    '导轨磨损故障',
        'frame_vibration': '轿架Z轴振动异常',
        'smoothness':    '轿厢X/Y轴平稳度异常',
        'water':         '底坑/机房水浸',
        'temperature':   '机房温度异常',
        'displacement':  '设备结构位移',
        'motor_current': '电机电流异常',
        'noise_ratio':   '机房异常噪声',
        'default':       '设备异常',
    }

    RISK_GRADE_DESC = {
        'H1': '良好，正常运行状态',
        'H2': '轻度异常，需加强关注',
        'H3': '预警，需安排检修计划',
        'H4': '高危，需立即停机处理',
    }

    STYLE_HINTS_H4 = [
        "以资深维保工程师的鉴定口吻撰写，语气严谨客观。",
        "使用简洁精准的工程术语，直接陈述危害结论和处置要求。",
        "用一段连贯文字撰写，先分析风险后果再给出强制处置措施。",
    ]

    STYLE_HINTS_H3 = [
        "以资深维保工程师的口吻撰写预防性维护建议，语气专业肯定。",
        "使用简洁精准的工程术语，强调提前干预的必要性和紧迫性。",
        "用一段连贯文字撰写，从设备可靠性角度论述为什么需要安排检修。",
    ]

    STYLE_HINTS_GENERAL = [
        "以资深维保工程师的口吻撰写诊断意见，语气专业务实。",
        "使用简洁精准的工程术语，像真人专家手写的评语。",
        "用一段连贯文字撰写，从设备全生命周期管理角度给出维护建议。",
    ]

    STYLE_HINTS_LOW = [
        "以资深维保工程师的口吻撰写巡检评语，语气积极肯定。",
        "使用简洁精准的工程术语，突出设备状态良好、运行稳定。",
        "用一段连贯文字撰写，体现预防性维护理念和对设备长期可靠性的信心。",
    ]

    DOMAIN_CONTEXT = (
        "【电梯维保领域知识】\n"
        "- 校轨：指导轨的校准校正，是导轨维护的专业术语，不可改为「校正轨道」\n"
        "- 导靴：轿厢与导轨之间的导向滑动装置，磨损后必须更换\n"
        "- 包络谱：轴承故障诊断的频域分析方法\n"
        "- 对中校准：电机与曳引机之间的轴对中调整\n"
        "- 反绳轮：曳引钢丝绳的导向轮\n"
        "- 门刀：轿门上的开锁装置\n"
        "- 保持架：滚动轴承中分隔滚动体的部件\n"
        "- 力矩扳手：用于螺栓定扭矩紧固的工具\n"
        "- 探伤检测：钢丝绳无损检测方法\n"
    )

    SYSTEM_PROMPT = (
        "你是电梯维保专家。" + DOMAIN_CONTEXT +
        "将用户给出的维保建议用不同措辞重新表述，保持专业含义完全一致。"
        "直接输出改写结果，禁止加任何前缀说明。"
    )

    def __init__(self, config: dict):
        llm_cfg = config.get('llm_config', {})
        self.enabled = llm_cfg.get('enable_llm', True) and TRANSFORMERS_AVAILABLE
        self.model_id = llm_cfg.get('model_id', 'Qwen/Qwen2.5-3B-Instruct')
        self.temperature = llm_cfg.get('temperature', 0.8)
        self.max_tokens = llm_cfg.get('max_tokens', 200)
        self.fallback_advice = config.get('expert_advice', {})
        self._model = None
        self._tokenizer = None
        if self.enabled:
            logger.info(f"LLMAdvisor 初始化完成 | model_id: {self.model_id}")
        else:
            logger.warning(
                f"LLMAdvisor 已禁用 "
                f"(enable_llm={llm_cfg.get('enable_llm', True)}, "
                f"TRANSFORMERS_AVAILABLE={TRANSFORMERS_AVAILABLE})"
            )

    def _load_model(self):
        try:
            print(f"[LLM] 正在加载模型: {self.model_id} (首次运行会自动下载，约1GB)")
            logger.info(f"正在加载LLM模型: {self.model_id}")

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=torch.float32,
            )

            print(f"[LLM] 模型加载成功")
            logger.info(f"LLM模型加载成功")
        except Exception as e:
            print(f"[LLM] 模型加载失败: {e}")
            logger.error(f"模型加载失败: {e}，回退到静态YAML建议")
            self.enabled = False

    @property
    def model_ready(self) -> bool:
        if not self.enabled:
            return False
        if self._model is None:
            self._load_model()
        return self._model is not None

    def _get_style_hint(self, risk_grade: str) -> str:
        if risk_grade == 'H4':
            pool = self.STYLE_HINTS_H4
        elif risk_grade == 'H3':
            pool = self.STYLE_HINTS_H3
        elif risk_grade == 'H1':
            pool = self.STYLE_HINTS_LOW
        else:
            pool = self.STYLE_HINTS_GENERAL  # H2 falls here
        return random.choice(pool)

    def _build_prompt(self, fault_type: str, risk_grade: str,
                      score=None, fuzzy_dist=None) -> str:
        seed_text = self._get_fallback(fault_type, risk_grade)
        style_hint = self._get_style_hint(risk_grade)

        task = "请用不同措辞重新表述以上维保建议，专业含义完全不变。"

        parts = [
            f"原文：{seed_text}",
            "",
            task,
            style_hint,
        ]

        parts.append("禁止输出称呼、问候语、开场白。")
        parts.append("禁止使用编号列表（1. 2. 3.）或项目符号（- ·）。")
        parts.append("禁止输出【高危】【预警】【轻度】【良好】等带括号的等级格式标签。")
        parts.append("必须输出一段连续完整的文字段落，不少于40字，内容充实具体。")

        if risk_grade == 'H4':
            raw_terms = self.H4_KEY_TERMS.get(fault_type, self.H4_KEY_TERMS.get('default', []))
            flat_terms = []
            for t in raw_terms:
                if isinstance(t, list):
                    flat_terms.append("或".join(t))
                else:
                    flat_terms.append(t)
            parts.append(f"必须包含以下关键词：{'、'.join(flat_terms)}。")

        parts.append("\n改写：")

        return "\n".join(parts)

    def _validate_output(self, text: str, fault_type: str, risk_grade: str) -> bool:
        text = text.strip()
        if len(text) < 15:
            return False

        # Check forbidden patterns (P0: double negation etc.)
        for pattern in self.FORBIDDEN_PATTERNS:
            if pattern in text:
                logger.warning(
                    f"LLM output contains forbidden pattern '{pattern}'. "
                    f"Output: {text[:80]}..."
                )
                return False

        if risk_grade == 'H4':
            terms = self.H4_KEY_TERMS.get(fault_type, self.H4_KEY_TERMS.get('default', []))
            for term in terms:
                if isinstance(term, list):
                    if not any(t in text for t in term):
                        logger.warning(
                            f"LLM output missing all H4 alternatives {term}. "
                            f"Output: {text[:80]}..."
                        )
                        return False
                elif term not in text:
                    logger.warning(
                        f"LLM output missing H4 key term '{term}'. "
                        f"Output: {text[:80]}..."
                    )
                    return False
        else:
            core_req = self.CORE_REQUIREMENTS.get((fault_type, risk_grade))
            if core_req:
                match_chars = sum(1 for c in core_req if c in text)
                if match_chars < len(core_req) * 0.4:
                    logger.warning(
                        f"LLM output too divergent from core '{core_req}'. "
                        f"Output: {text[:80]}..."
                    )
                    return False
        return True

    def _get_fallback(self, fault_type: str, risk_grade: str) -> str:
        """Replica of original report_generator.py static lookup logic."""
        advice_key = (
            'bearing_fault'
            if fault_type.startswith('bearing_') and fault_type != 'bearing_cage'
            else fault_type
        )
        target = self.fallback_advice.get(advice_key,
                                          self.fallback_advice.get('default', {}))
        return target.get(risk_grade, "暂无特定级别建议，请结合现场物理排查确认。")

    def generate_advice(self, fault_type: str, risk_grade: str,
                        score: float = None, fuzzy_dist: dict = None) -> str:
        if not self.model_ready:
            print(f"[LLM] 不可用，使用静态建议")
            logger.info(f"LLM不可用，使用静态建议 | fault={fault_type}, grade={risk_grade}")
            return self._get_fallback(fault_type, risk_grade)

        try:
            user_prompt = self._build_prompt(fault_type, risk_grade, score, fuzzy_dist)

            for attempt in range(2):
                temp = self.temperature if attempt == 0 else 0.3
                print(f"[LLM] 生成建议 attempt={attempt+1}, temp={temp}")
                logger.info(
                    f"LLM生成建议 | fault={fault_type}, grade={risk_grade}, "
                    f"attempt={attempt+1}, temp={temp}"
                )

                messages = [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
                text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self._tokenizer(text, return_tensors="pt")

                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=self.max_tokens,
                        temperature=temp,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )

                generated = outputs[0][len(inputs.input_ids[0]):]
                advice = self._tokenizer.decode(generated, skip_special_tokens=True).strip()

                if self._validate_output(advice, fault_type, risk_grade):
                    print(f"[LLM] 生成成功: {advice[:60]}...")
                    logger.info(f"LLM生成成功 | 输出: {advice[:50]}...")
                    return advice

                logger.warning(
                    f"验证失败 attempt={attempt+1}, "
                    f"{'用temperature=0.3重试' if attempt == 0 else '回退到静态建议'}"
                )

            return self._get_fallback(fault_type, risk_grade)

        except Exception as e:
            print(f"[LLM] 推理异常: {e}")
            logger.error(f"LLM推理异常: {e}，回退到静态建议")
            return self._get_fallback(fault_type, risk_grade)
