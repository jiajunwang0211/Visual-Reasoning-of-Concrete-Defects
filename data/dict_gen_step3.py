import os
import json
import openai
import re

# ================= 配置区域 =================
API_KEY = ""
BASE_URL = ""
MODEL_NAME = "o1"  # 您也可以根据需要使用 o1-preview 或 gpt-4o

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 输出目录配置 =================
OUTPUT_DIR = "step3_defect_refined_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 全局背景与协议提示词 =================
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

# ================= 1. 执行者 (Refiner/Generator) 提示词 =================
REFINER_PROMPT = """
<Current Task: Step_3_Refinement_Execution>
<Reference_Standard>
{GLOBAL_OBJECTIVE}
</Reference_Standard>

<Role_Definition>
你现在的具体身份是【视觉本体架构师（Visual Ontology Architect）】。
你的思维模式是“建设性、收敛性与极度严谨”。
你不仅拥有丰富的数据清洗和结构化抽象经验，还极度擅长将散乱的候选维度重塑为一套犹如数学公理般精确、稳固的 Schema，在剔除冗余的情况下，尽可能地修改还有价值地视觉线索。
</Role_Definition>

<Context>
你的任务是基于输入的候选属性维度，以及【审计员（Auditor）的上一轮反馈意见】，严格执行精炼协议，输出优化后的 Schema。
</Context>

<Input Data>
1. 当前的 Ontology Schema:
{current_ontology_json}

2. 来自审计员的上一轮反馈 (Auditor Feedback):
{auditor_feedback}
</Input Data>

<Refinement Protocols>
请针对输入中的 S_int 和 S_ctx，逐一执行以下五项协议，严禁跳过：

1. **冗余合并协议**:
   - 逻辑：扫描所有维度，寻找语义重叠或视觉特征高度一致的项（如 "Color" 与 "Hue", "Form" 与 "Shape"）。
   - 动作：将它们合并为一个最具概括性的名称，并重新整合定义与视觉锚点。

2. **抽象修正协议**: 
   - 逻辑：检查条目是否误将“属性值”（Value，如 "Circular"）当成了“属性维度”（Key，如 "Geometric Form"）。
   - 动作：如果发现此类错误，必须将其重命名为其更高层级的维度名称。

3. **脏数据清洗协议**:
   - 逻辑：检查是否有类似"AppleWebKit"、网络代码等大模型幻觉字符串，或误把“属性的具体取值”当成“维度本身名称”的错误 
   - 动作：如果发现此类错误，直接将其删除。

4. **视觉锚定协议**: 
   - 逻辑 A（剔除）：发现无法通过单帧像素直接观察的属性（如：成因 Damage_Cause、严重程度 Severity_Level、修复紧急度 Repair_Urgency），必须直接剔除。
   - 逻辑 B（转译）：若描述中带有物理状态/物质名词，请将其分解或者转译为客观的视觉证据描述。
     - 示例1：发现 "Efflorescence (泛碱)" -> 重命名为 "Surface_Powder_Distribution (表面粉末分布)"。
     - 示例2：发现 "Volume_Loss (体积流失)" -> 既然不能测体积，但可以看到坑洞阴影，可重命名转译为 "Cavity_Shadow_Pattern (坑洞阴影模式)"。
     - 示例3：如果该物理现象已被 Color_Hue, Texture 等完美覆盖，则在 log 中注明“已合并至 Color/Texture”，方可删除。

5. **本质性审计协议 (Essence & Invariance Audit) **:
   - 逻辑：剔除受“相机参数”或“随机构图”影响的瞬态特征。
   - 审查对象：
     - **尺寸与尺度 (Size/Scale)**: 剔除一切描述物体在图中“大小”、“占比”的维度。因为这些由拍摄距离决定，不是物体的本质属性。
     - **画面位置 (Image Position)**: 剔除描述物体在图中“左上角、中心、边缘”的维度。因为这些由拍摄角度决定，对类别识别无意义。
   - 动作：直接剔除（Discard）此类维度。保留的维度必须是“无论相机怎么拍，物体本身都具备，其语义都保持稳定”的视觉原语。

6. **维度正交化协议 (Orthogonal Axis Check)**:
   - 逻辑 A（空间归属）：确保 S_int 仅含病害自身视觉表象特征，S_ctx 仅含病害与环境或其他物体的交互。
   - 动作 A：检查空间归属是否错误。例如，“Downward_Streak_Orientation”（向下条纹方向）本质描述的是污渍的自身几何形态，应当归属于 S_int，而不是 S_ctx。若发现类似错误，必须将其移入正确的空间类别。
   - 逻辑 B（正交解耦）：确保同空间内的各个维度描述的是物体的一个“独立物理视觉上的切面”，保证绝对独立，互不干涉。
   - 动作 B：如果发现两个维度高度相关或存在强逻辑绑定，必须强制拆分或重新定义。确保改变一个维度的取值时，理论上绝对不会强制改变另一个维度的取值。
</Refinement Protocols>

<Output_Format>
必须返回 JSON 格式：
{{
    "refinement_log": [
        "Action 1: Removed 'Size_Scale' based on auditor feedback (violates scale invariance).",
        "Action 2: Merged 'Hue' into 'Color_Hue' to eliminate redundancy.",
        "Action 3: Moved 'Downward_Streak_Orientation' to S_int.",
        ...
    ],
    "proposed_ontology": {{
        "S_int": {{ ... }},
        "S_ctx": {{ ... }}
    }}
}}
</Output_Format>
</Current Task: Step_3_Refinement_Execution>
"""

# ================= 2. 审查者 (Auditor/Critic) 提示词 =================
AUDITOR_PROMPT = """
<Current Task: Step_3_Strict_Auditing>
<Reference_Standard>
{GLOBAL_OBJECTIVE}
</Reference_Standard>

<Role_Definition>
你现在的具体身份是【视觉本体严酷审计员（Visual Ontology Auditor）】。
你的思维模式是“极其挑剔的批判性与防御性”。
你是一个没有任何情感、只讲究绝对规则的逻辑判官。你的任务绝对不是去帮忙修改，而是拿着放大镜和红线规则，逐字逐句地去判断是否触碰到了“红线”。
只要 Refiner（架构师）提交的 Schema 中哪怕藏着一个非视觉词汇、一个受相机距离影响的特征、或者一丝逻辑重叠，你都要毫不留情地将其打回。
</Role_Definition>

<Context>
审视架构师刚刚提交的 proposed_ontology_json，对照所有红线协议，寻找任何违规之处。只有当 Schema 完美无瑕时，才允许放行。
</Context>

<Input Data>
架构师提交的拟定 Schema:
{proposed_ontology_json}
</Input Data>

<Audit Checklist>
请逐一排查以下红线：
1. 是否还残留着非视觉特征？(如：原因分析、深度评估、严重等级、修复建议、力学特征)
2. 是否还残留着受相机参数/构图影响的特征？(极其严格：绝对不允许存在 Size_Scale, Area_Percentage, Image_Position 等描述大小和位置的词汇)
3. 空间归属是否错乱？(S_ctx 中是否混入了仅仅描述病害自身颜色/形状/纹理的维度？S_int 中是否混入了描述病害与周围背景/环境交互的维度？)
4. 是否存在明显的同义词冗余未合并？(如 Color_Hue 与 Color，或者 Texture 与 Surface_Roughness 指向完全相同的视觉特征)
5. 视觉锚点(visual_anchors)中是否还残留毫无意义的幻觉乱码？(如 AppleWebKit)
6. 维度是否正交且原子化？(确保没有维度包含了主观判断，且各个维度互不干涉。)
7. 检查是否出现了物理名字污染，比如"Rust(锈迹)", "Moisture/Wet(水渍/潮湿)", "Biological/Algae(生物/藻类)" 等物质或物理状态名词？如果有，要求用视觉原语来进行转译或者解构。
</Audit Checklist>

<Requirement>
- 如果上述任何一条被触发，或者有其他违背 Objective 的缺陷，请将 `is_stable` 设为 false，并在 `audit_feedback` 中给出严厉、具体的修改指令（必须指出具体的维度名称）。
- 如果经历了多轮修改，该 Schema 已经完全无懈可击，没有任何毛病，请将 `is_stable` 设为 true，反馈可填 "Approved."。
</Requirement>

<Output_Format>
必须返回 JSON 格式：
{{
    "is_stable": true/false,
    "audit_feedback": "Your explicit critique here. E.g., 'REJECTED. You failed to remove Size_Scale. Size is determined by camera distance. Delete it immediately in the next round.' or 'Approved.'"
}}
</Output_Format>
</Current Task: Step_3_Strict_Auditing>
"""


def parse_json_response(raw_content):
    # 清理 markdown 标记提取 json
    clean_json = re.sub(r"```json\s*|```", "", raw_content).strip()
    return json.loads(clean_json)


def run_step3_refinement():
    # 1. 加载 Step 2 的最终结果文件
    step2_file_path = "step2_defect_final_output/step2_defect_final_schema.json"
    try:
        with open(step2_file_path, "r", encoding='utf-8') as f:
            step2_data = json.load(f)
        raw_schema = step2_data.get("final_schema", step2_data)
        current_ontology_input = json.dumps(raw_schema, indent=2, ensure_ascii=False)
    except FileNotFoundError:
        print(f"错误：未找到输入文件 {step2_file_path}")
        print("请确保先完成 Step 2，并将结果正确放置在对应目录。")
        return

    iteration_count = 0
    max_iters = 6  # 设定最大博弈轮次
    is_stable = False
    full_audit_history = []

    # 初始的空白反馈
    auditor_feedback = "None. This is the initial iteration. Please perform your first pass of comprehensive refinement based on the protocols."
    final_ontology = {}

    print(f"--- Step 3: 开始执行 [架构师-审计员] 双智能体高强度对抗循环 ---")

    while not is_stable and iteration_count < max_iters:
        iteration_count += 1
        print(f"\n[第 {iteration_count} 轮对抗] =======================================")

        try:
            # ================= 阶段 A: Refiner (架构师) 执行修改 =================
            print(" ⏳ [架构师] 正在分析反馈，重构属性维度...")
            refiner_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": REFINER_PROMPT.format(
                        GLOBAL_OBJECTIVE=GLOBAL_OBJECTIVE,
                        current_ontology_json=current_ontology_input,
                        auditor_feedback=auditor_feedback
                    )}
                ],
                response_format={"type": "json_object"}
            )

            refiner_data = parse_json_response(refiner_response.choices[0].message.content)
            refinement_log = refiner_data.get("refinement_log", [])
            proposed_ontology = refiner_data.get("proposed_ontology", {})
            proposed_ontology_json = json.dumps(proposed_ontology, indent=2, ensure_ascii=False)

            print(f"  🏗️ [架构师] 提交了新版本，执行了 {len(refinement_log)} 项大手术:")
            for idx, action in enumerate(refinement_log, 1):
                print(f"     - {action}")

            # ================= 阶段 B: Auditor (审计员) 严格审查 =================
            print("\n ⏳ [审计员] 正在拿着放大镜审查新版本 Schema...")
            auditor_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": AUDITOR_PROMPT.format(
                        GLOBAL_OBJECTIVE=GLOBAL_OBJECTIVE,
                        proposed_ontology_json=proposed_ontology_json
                    )}
                ],
                response_format={"type": "json_object"}
            )

            auditor_data = parse_json_response(auditor_response.choices[0].message.content)
            is_stable = auditor_data.get("is_stable", False)
            auditor_feedback = auditor_data.get("audit_feedback", "")

            print(f"  ⚖️ [审计员] 最终判决: {'✅ 放行 (Stable)' if is_stable else '❌ 驳回重做 (Unstable)'}")
            print(f"  🗣️ [审计员] 严酷评语: {auditor_feedback}")

            # 记录本轮博弈历史
            full_audit_history.append({
                "iteration": iteration_count,
                "architect_actions": refinement_log,
                "auditor_critique": auditor_feedback,
                "is_stable": is_stable,
                "ontology_snapshot": proposed_ontology
            })

            # 为下一轮准备（或作为最终结果结束）
            current_ontology_input = proposed_ontology_json
            final_ontology = proposed_ontology

            if is_stable:
                print(f"\n🎯 恭喜！审计员找不到任何破绽，本体结构已完全收敛并定型！")
            elif iteration_count == max_iters:
                print(f"\n⚠️ 达到最大对抗轮数 ({max_iters})，强制结束循环。输出当前可用版本。")

        except Exception as e:
            print(f"\n❌ 第 {iteration_count} 轮执行或 JSON 解析异常: {e}")
            break

    # ================= 结果分类保存 =================
    final_output_path = os.path.join(OUTPUT_DIR, "step3_defect_refined_ontology.json")
    with open(final_output_path, "w", encoding='utf-8') as f:
        json.dump(final_ontology, f, indent=4, ensure_ascii=False)

    evolution_history = {
        "metadata": {
            "step": "Step 3: Dual-Agent Defect Ontology Refinement",
            "total_iterations": iteration_count,
            "convergence_reached": is_stable,
            "model_used": MODEL_NAME
        },
        "evolution_logs": full_audit_history
    }
    history_output_path = os.path.join(OUTPUT_DIR, "step3_defect_evolution_history.json")
    with open(history_output_path, "w", encoding='utf-8') as f:
        json.dump(evolution_history, f, indent=4, ensure_ascii=False)

    print(f"\n--- Step 3 双智能体本体精炼对抗流程结束 ---")
    print(f"✅ 最终精炼结果已保存至: {final_output_path}")
    print(f"📝 完整博弈审查历史已保存至: {history_output_path}")


if __name__ == "__main__":
    run_step3_refinement()