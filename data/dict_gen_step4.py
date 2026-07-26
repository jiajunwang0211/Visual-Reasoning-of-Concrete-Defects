import os
import json
import openai
import re
import time

# ================= 配置区域 =================
API_KEY = ""
BASE_URL = ""
MODEL_NAME = "gpt-4o"  # 强烈建议双智能体对话使用 gpt-4o 以获得最佳的 JSON 结构化和响应速度

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 目录配置 =================
INPUT_ONTOLOGY_PATH = "step3_defect_refined_output/step3_defect_refined_ontology.json"
KFOLD_CONFIG_PATH = "defect_k_fold_config.json"
OUTPUT_DIR = "step4_defect_adversarial_output_D"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 全局背景 =================
GLOBAL_OBJECTIVE = """
<Role>
你是一位拥有双重顶级学术背景的交叉学科专家：
1. 资深土木工程师与病理学家：深耕建筑结构健康监测（SHM），对混凝土表面病害（如裂缝、剥落、渗水、泛碱、碳化等）的视觉表象、几何形态、纹理特征和色彩分布了如指掌。
2. 计算机视觉本体学泰斗：精通开放世界的建筑物混凝土表面目标检测、视觉多模态大模型以及知识图谱构建。
你善于观察建筑混凝土表面病害，将复杂的物理病害现象，严谨地解构为计算机视觉模型能够完美理解、学习和对齐的、高度正交的底层“视觉原语”。
</Role>

<Objective>
构建一套“混凝土表面病害原语本体（Concrete Defect Primitive Ontology）”。这套属性维度本体必须满足：
1.  双空间解耦：严格区分 `S_int` (物体中心空间) 和 `S_ctx` (环境关联空间)。希望两个语义空间的属性维度都可以为最后的开放世界-开放词汇目标检测任务提供支持。
2.  视觉导向分析：在归纳两个语义空间的属性维度时，针对静态视觉图像，仅关注视觉可感知的视觉属性维度特征，忽略纯功能性或抽象的非视觉属性维度特征。
3.  正交完备性：维度之间无冗余元素（No Superfluous Elements），且能描述任何可见物体。
4.  零样本泛化力：必须能通过特定“未知物体集合”的对抗性压力测试。
5.  原子性：必须将复杂的病害拆解为不可分割的原子视觉原语（如：形状、颜色、边缘特征、纹理），杜绝使用包含主观判断的高级语义（如：严重程度）。
</Objective>
"""

# ================= 蓝方智能体：防御者 (Defender) =================
DEFENDER_PROMPT = """
<Current Task: Step_4_Blue_Team_Defense>
<Reference_Standard>
{GLOBAL_OBJECTIVE}
</Reference_Standard>

<Role>
你是对混凝土表面病害（如裂缝、剥落、渗水、泛碱、碳化等）的视觉表象、几何形态、纹理特征和色彩分布了如指掌的【一线巡检工程师 (Frontline Inspection Engineer)】。
你的任务是：在现场遇到了一种从未见过的“未知病害”，但你手里只有一份现有的“标准视觉特征排查表（即当前的特征维度本体）”。
你必须尽最大努力，遵守规范，仅使用排查表上现有的特征维度，像拼积木一样把这个新病害的视觉表现尽可能准确地描述出来。
</Role>

<Input Data>
1. 现有的本体 Schema:
{refined_ontology_json}

2. 需要你描述的未知病害:
【 {target_defect} 】
</Input Data>

<Execution_Protocol>
1. **Compositional Attempt (组合重构尝试)**: 
   - 核心任务：尝试使用 Step 3 已有的多个纯视觉属性维度进行“乐高式”拼接。
   - 判定逻辑：如果“维度A（如：颜色对比度） + 维度B（如：坑洞阴影） + 维度C（如：边缘锐度）...”的组合足以勾勒出该病害的核心视觉特征并能与其他病害区分，则视为【描述成功】。
   - 严禁：严禁因为没有专属维度（如没有“钢筋纹理”这个维度）或者“单一匹配词”而判定失败。

2. **Semantic Boundary Check (语义边界校准)**: 
   - 核心任务：区分“属性值缺失（Value Missing）”与“维度缺失（Dimension Missing）”。
   - 判定逻辑：如果发现缺失的只是一个具体的描述词（例如：遇到“青苔附着”，发现缺少“绿色”或者“绒毛感”），但它明明可以归入已有的【Color_Hue (色相)】或【Surface_Texture (表面纹理)】维度下，这属于“取值空间待扩充”，必须判定为 **PASS**。   
</Execution_Protocol>

<Output_Format>
请返回 JSON 格式：
{{
    "selected_dimensions": ["Dim_A", "Dim_B", "Dim_C"],
    "attempted_description": "使用了维度A(取值为...) + 维度B(取值为...) + 维度C(取值为...) 来重构该病害。因为..."
}}
</Output_Format>
</Current Task: Step_4_Blue_Team_Defense>
"""

# ================= 红方智能体：挑刺官 (Attacker) =================
ATTACKER_PROMPT = """
<Current Task: Step_4_Red_Team_Attack>
<Reference_Standard>
{GLOBAL_OBJECTIVE}
</Reference_Standard>

<Role>
你是对混凝土表面病害（如裂缝、剥落、渗水、泛碱、碳化等）的视觉表象、几何形态、纹理特征和色彩分布了如指掌的【资深病害鉴定专家 (Senior Defect Identification Expert)】。
你的任务是：极其严苛地审查一线工程师提交的“病害重构报告”。
你深知该病害在真实物理世界中的表现，你需要排查：工程师是不是在“生搬硬套”？目前的排查表是不是遗漏了某种根本无法描述的【核心视觉盲点】？
</Role>

<Input Data>
1. 现有的本体 Schema:
{refined_ontology_json}

2. 测试的未知病害:
【 {target_defect} 】

3. 提交的重构方案 (Attempted Description):
{defender_response}
</Input Data>

<Execution_Protocol>
1. **True Blind Spot Discovery (真盲点判定)**: 
   - 核心任务：寻找核心的“视觉感知坐标轴”缺失。
   - 判定逻辑：只有当病害的核心视觉属性在现有的属性维度体系中完全找不到任何可以容纳它的维度时（例如：遇到了“半透明的树脂修复胶”，而本体中完全没有描述“透明度/透光性”的维度），才判定为 **FAIL**。

2. **Verdict & Documentation (定论与转译)**: 
   - 如果 PASS：简述使用了哪些维度进行重构逻辑。
   - 如果 FAIL：记录“Missing Blind Spot (缺失的视觉盲点)”，并将其转译为一个高水平的“能力问题（CQ, Competency Question）”。
</Execution_Protocol>

<Output_Format>
请返回 JSON 格式：
{{
    "critique_logic": "你的内心批判逻辑：巡检工程师忽略了什么？现有的维度组合能完全刻画该病害吗？",
    "visual_blind_spots": "None / 具体缺失的视觉原语维度（例如：缺乏描述半透明光学深度的维度）",
    "verdict": "PASS/FAIL",
    "supplemental_cq": {{
        "question": "如果是 FAIL，请提出针对排查表研发团队的 Competency Question（英文），例如 'How can the ontology represent semi-transparent optical depth?'。如果 PASS，填 'None'。",
        "strategic_goal": "如果是 FAIL，请解释提出该问题的战略目标（英文），例如 'This question addresses the challenge of representing material properties that affect light transmission.'。如果 PASS，填 'None'。"
    }}
}}
</Output_Format>
</Current Task: Step_4_Red_Team_Attack>
"""


def parse_json_response(raw_content):
    clean_json = re.sub(r"```json\s*|```", "", raw_content).strip()
    return json.loads(clean_json)


def call_llm(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"}
            )
            return parse_json_response(response.choices[0].message.content)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"❌ LLM 请求失败: {e}")
                return None
            time.sleep(2)


def run_step4_adversarial_test():
    # 1. 加载本体数据
    try:
        with open(INPUT_ONTOLOGY_PATH, "r", encoding='utf-8') as f:
            ontology_data = json.load(f)
        ontology_json_str = json.dumps(ontology_data, ensure_ascii=False)
        print(f"✅ 成功读取病害本体架构 ({INPUT_ONTOLOGY_PATH})")
    except FileNotFoundError:
        print(f"❌ 错误：未找到输入文件 {INPUT_ONTOLOGY_PATH}。")
        return

    # 2. 加载 K-Fold 配置
    try:
        with open(KFOLD_CONFIG_PATH, "r", encoding='utf-8') as f:
            kfold_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 未找到配置文件 {KFOLD_CONFIG_PATH}。")
        return

    rounds_to_test = ["round_1", "round_2", "round_3"]

    # 3. 循环遍历每一个 Round
    for current_round in rounds_to_test:
        adversarial_classes = kfold_data.get(current_round, {}).get("adversarial_challenge", [])
        if not adversarial_classes:
            continue

        print(f"\n" + "=" * 60)
        print(f"🚀 开始执行 [{current_round}] 双智能体对抗测试 (测试病害类共 {len(adversarial_classes)} 个)")
        print("=" * 60)

        round_results = {
            "round_info": current_round,
            "test_summary": {"total_classes": len(adversarial_classes), "pass_count": 0, "fail_count": 0},
            "adversarial_test_results": {},
            "failure_driven_cqs": []
        }

        for defect_idx, target_defect in enumerate(adversarial_classes):
            print(f"\n🔍 测试样本 [{defect_idx + 1}/{len(adversarial_classes)}]: {target_defect}")

            # --- 回合 1 重构
            print("   🛡️ 工程师(Defender) 正在思考重构方案...")
            defender_msg = [{"role": "user", "content": DEFENDER_PROMPT.format(
                GLOBAL_OBJECTIVE=GLOBAL_OBJECTIVE,
                refined_ontology_json=ontology_json_str,
                target_defect=target_defect
            )}]
            defender_output = call_llm(defender_msg)

            if not defender_output:
                continue

            attempted_desc = defender_output.get("attempted_description", "N/A")
            print(f"   💬 工程师提交: 选用了维度 {defender_output.get('selected_dimensions', [])}")

            # --- 回合 2 审计
            print("   🗡️  鉴定专家正在进行极限审计...")
            attacker_msg = [{"role": "user", "content": ATTACKER_PROMPT.format(
                GLOBAL_OBJECTIVE=GLOBAL_OBJECTIVE,
                refined_ontology_json=ontology_json_str,
                target_defect=target_defect,
                defender_response=json.dumps(defender_output, ensure_ascii=False)
            )}]
            attacker_output = call_llm(attacker_msg)

            if not attacker_output:
                continue

            verdict = attacker_output.get("verdict", "PASS")
            print(f"   ⚖️ 鉴定专家裁决: {verdict}")
            if verdict == "FAIL":
                print(f"   🔥 发现盲点: {attacker_output.get('visual_blind_spots')}")

            # --- 记录数据 ---
            if verdict == "PASS":
                round_results["test_summary"]["pass_count"] += 1
            else:
                round_results["test_summary"]["fail_count"] += 1
                round_results["failure_driven_cqs"].append({
                    "source_class": target_defect,
                    "missing_logic": attacker_output.get("visual_blind_spots", ""),
                    "new_cq": attacker_output.get("supplemental_cq", "")
                })

            round_results["adversarial_test_results"][target_defect] = {
                "attempted_description": attempted_desc,
                "attacker_critique": attacker_output.get("critique_logic", ""),
                "visual_blind_spots": attacker_output.get("visual_blind_spots", "None"),
                "verdict": verdict,
                "supplemental_cq": attacker_output.get("supplemental_cq", "None")
            }

        # 4. 保存当前 Round 的测试报告
        report_path = os.path.join(OUTPUT_DIR, f"step4_{current_round}_dual_report.json")
        with open(report_path, "w", encoding='utf-8') as f:
            json.dump(round_results, f, indent=4, ensure_ascii=False)
        step1_compatible_cqs = {"cqs": []}
        for idx, failure_item in enumerate(round_results["failure_driven_cqs"]):
            cq_data = failure_item.get("new_cq", {})
            if isinstance(cq_data, dict):
                question = cq_data.get("question", "N/A")
                strategic_goal = cq_data.get("strategic_goal",
                                             f"Addresses the missing dimension: {failure_item.get('missing_logic')}")
            else:
                question = str(cq_data)
                strategic_goal = f"Addresses the missing dimension: {failure_item.get('missing_logic')}"

            if question and str(question).strip().lower() != "none":
                step1_compatible_cqs["cqs"].append({
                    "id": f"CQ_{current_round.upper()}_{idx + 1}",
                    "question": question,
                    "strategic_goal": strategic_goal
                })

        cqs_path = os.path.join(OUTPUT_DIR, f"step4_{current_round}_failure_cqs.json")
        with open(cqs_path, "w", encoding='utf-8') as f:
            json.dump({"round": current_round, "supplemental_cqs": round_results["failure_driven_cqs"]}, f, indent=4,
                      ensure_ascii=False)

        print(
            f"\n🎯 [{current_round}] 汇总: 通过 {round_results['test_summary']['pass_count']} 个, 失败(盲点) {round_results['test_summary']['fail_count']} 个")
        print(f"📂 报告已保存至: {report_path}")

    print(f"\n🎉 所有 Round 对抗测试执行完毕！输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_step4_adversarial_test()