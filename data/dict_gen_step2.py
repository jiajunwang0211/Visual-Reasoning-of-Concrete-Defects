import os
import json
import openai
import time

# ================= 配置区域 =================
API_KEY = ""
BASE_URL = ""
MODEL_NAME = "o1"

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 已知病害类别 (手动配置) =================
# 您可以在此处手动填入需要的病害类别
DEFECT_CLASSES = [
    "定向裂缝","网状裂缝","钢筋裸露","剥落","起皮","崩解","蜂窝","水渍","铁锈","生物生长污渍","泛碱","结壳","钟乳石状析出","气孔","冷缝"
]

# ================= 输出目录配置 =================
# 定义两个文件夹名称
HISTORY_DIR = "step2_defect_history_logs"  # 存放每一轮的历史记录
FINAL_DIR = "step2_defect_final_output"  # 存放最终结果

# 确保文件夹存在
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# ================= 全局背景 (针对混凝土病害定制) =================
GLOBAL_OBJECTIVE = """
<Role>
你是一位专精于“建筑混凝土表面病害检测与分析”的资深土木工程本体学家。你善于通过“起草-批判-修正”的元认知循环的迭代来构建严密的病害属性维度体系。
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

# ================= Step 2 发散生成提示词 (中文) =================
STEP2_GENERATION_PROMPT = """

<Current Task: Step_2_Dimension_Generation>
<Reference_Standard>
{GLOBAL_OBJECTIVE}
</Reference_Standard>

<Context>
你是一位善于思考的“开放世界建筑混凝土表面病害”的资深计算机视觉本体学家。

你现在正在基于 Step 1 产出的能力问题（CQs），针对已知的混凝土表面病害类别提取初始属性维度。
你的核心信条是：任何复杂的视觉物体都是由一系列基础的、可观察的、且能跨类别共享的**“视觉原语属性维度”**组合而成的。
根据流程要求，这一步是“尽情发散阶段”，核心目标是**尽可能发掘和补充**所有可能的视觉特征维度以满足回答所有的能力问题（CQs）。
这一步是“发散建模阶段”。你的目标是**完备性**。如果发现当前的维度体系存在缺失，无法确保所有能力问题都能在属性维度体系中找到对应的“视觉答案”，你需要**在现有基础上进行补充**。
</Context>

<Input Data>
1. 已知病害类别列表: {class_list_str}
2. 核心能力问题 (CQs):
{cqs_content}

3. 当前已累积的属性维度 (Current Accumulated Schema):
{current_schema_context}
</Input Data>

<Execution_Protocol>
请独立起草一套**新增**的属性维度方案：
1. 找出“核心能力问题”中提及但尚未包含在“当前已累积的属性维度”中的特征。
2. 直接起草并输出这些新的属性维度。
3. 确保区分 S_int (仅描述物体自身的视觉特征) 和 S_ctx (仅描述物体与环境或其它物体的视觉交互)。
4. 尽情发散，直到你认为现有维度已经完全覆盖了所有的 CQs。如果没有任何新维度可以新增，请输出空的 JSON 结构。
5. 严禁提取不可见的属性维度，必须确保提取的每个属性维度都能在单帧静态像素中找到直接视觉证据。例如Color、Edge_Sharpness、Proximity_to_Rebar是正确的视觉属性维度，Severity_Level、Damage_Cause是错误的。
</Execution_Protocol>

<Output_Format>
请以 JSON 格式输出，定义部分请使用 **英文**。
格式如下：
{{
    "S_int": {{
        "Dimension_Name": {{
        "definition": "Definition of the defect dimension...",
        "visual_anchors":  ["value_example_1", "value_example_2"]
        }}
    }},
    "S_ctx": {{
        "Dimension_Name": {{
        "definition": "Contextual relationship definition...",
        "visual_anchors":  ["value_example_1", "value_example_2"]
        }}
    }}
}}
</Output_Format>
</Current Task: Step_2_Dimension_Generation>
"""


def get_dimension_keys(schema):
    """辅助函数：提取 schema 中所有的维度名称，用于比较差异"""
    keys = set()
    if not schema:
        return keys
    for key in schema.get("S_int", {}).keys():
        keys.add(f"S_int:{key}")
    for key in schema.get("S_ctx", {}).keys():
        keys.add(f"S_ctx:{key}")
    return keys


def run_step2_exploration():
    # 1. 加载输入数据
    try:
        # 读取 Step 1 生成的英文病害 CQs (请确保此文件存在)
        with open("step1_final_result/step1_final_cqs_only.json", "r", encoding='utf-8') as f:
            step1_data = json.load(f)

        # 处理可能的嵌套结构
        cqs_raw = step1_data.get("final_cqs", {})
        if isinstance(cqs_raw, dict) and "cqs" in cqs_raw:
            cqs_list = cqs_raw["cqs"]
        else:
            cqs_list = cqs_raw if isinstance(cqs_raw, list) else []

        cqs_content = "\n".join([f"- {cq.get('id', 'N/A')}: {cq.get('question', '')}" for cq in cqs_list])

        # 使用手动填入的病害分类配置
        class_list_str = ", ".join(DEFECT_CLASSES)

    except FileNotFoundError as e:
        print(f"错误：缺少必要的输入文件 {e}")
        print("请确保同目录下存在 'step1_final_result/step1_final_cqs_only.json'")
        return

    # 2. 初始化迭代变量
    current_schema = {"S_int": {}, "S_ctx": {}}  # 初始为空
    all_evolution_logs = []

    iteration = 0
    max_iterations = 8  # 安全上限，防止无限循环
    is_saturated = False  # 收敛标志

    print(f"--- Step 2: 正在执行针对混凝土表面病害的初始维度发散提取 ---")
    print(f"--- 历史记录将保存至: ./{HISTORY_DIR} ---")
    print(f"--- 最终结果将保存至: ./{FINAL_DIR} ---")

    while not is_saturated and iteration < max_iterations:
        iteration += 1
        print(f"\n[第 {iteration} 轮迭代思考中...]")

        # 准备上下文
        if iteration == 1:
            schema_context_str = "None (Initial Empty State)"
        else:
            schema_context_str = json.dumps(current_schema, indent=2, ensure_ascii=False)

        try:
            # ================= 阶段: 发散生成初始维度 =================
            print("  -> 正在发散生成新增维度...")
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": STEP2_GENERATION_PROMPT.format(
                        GLOBAL_OBJECTIVE=GLOBAL_OBJECTIVE,
                        class_list_str=class_list_str,
                        cqs_content=cqs_content,
                        current_schema_context=schema_context_str
                    )}
                ],
                response_format={"type": "json_object"}
            )
            new_schema = json.loads(response.choices[0].message.content)

            # ================= 智能合并与判断 =================
            # 【修复浅拷贝Bug】: 必须分别 copy 内部的字典，防止 current_schema 被连带修改
            merged_schema = {
                "S_int": current_schema.get("S_int", {}).copy(),
                "S_ctx": current_schema.get("S_ctx", {}).copy()
            }

            # 合并 S_int
            if "S_int" in new_schema:
                for k, v in new_schema["S_int"].items():
                    merged_schema["S_int"][k] = v

            # 合并 S_ctx
            if "S_ctx" in new_schema:
                for k, v in new_schema["S_ctx"].items():
                    merged_schema["S_ctx"][k] = v

            # 比较差异，判断维度是否不再新增
            prev_keys = get_dimension_keys(current_schema)
            new_keys = get_dimension_keys(merged_schema)

            added_keys = new_keys - prev_keys
            num_s_int = len(merged_schema.get('S_int', {}))
            num_s_ctx = len(merged_schema.get('S_ctx', {}))

            print(f"  -> 本轮合并后统计: S_int: {num_s_int}, S_ctx: {num_s_ctx}")

            # 构建单轮日志
            log_entry = {
                "iteration": iteration,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "generated_schema_snapshot": new_schema,
                "added_dimensions_count": len(added_keys),
                "added_dimensions_names": list(added_keys),
                "current_schema_snapshot": merged_schema
            }
            all_evolution_logs.append(log_entry)

            # 实时保存这一轮的 Log 到历史文件夹
            single_log_filename = os.path.join(HISTORY_DIR, f"iteration_{iteration}_log.json")
            with open(single_log_filename, "w", encoding='utf-8') as f:
                json.dump(log_entry, f, indent=4, ensure_ascii=False)
            print(f"  -> [备份] 第 {iteration} 轮日志已保存至 {single_log_filename}")

            # 判定是否不再新增 (流程图：判断维度是否不再新增 -> 是 -> 保存最终属性维度)
            if len(added_keys) == 0:
                print(f"  -> [收敛判定] 维度不再新增。结束本阶段迭代。")
                is_saturated = True
            else:
                print(f"  -> [继续发散] 本轮新增了 {len(added_keys)} 个病害维度，进入下一轮迭代。")
                # 更新当前状态
                current_schema = merged_schema

        except Exception as e:
            print(f"第 {iteration} 轮执行异常: {e}")
            break

    # 3. 最终结果分类保存

    # 3.1 保存最终的 Schema (提取干净的病害本体结果)
    final_result_payload = {
        "metadata": {
            "step": "Step 2: Final Concrete Defect Schema",
            "total_iterations": iteration,
            "converged": is_saturated
        },
        "final_schema": current_schema
    }
    final_filename = os.path.join(FINAL_DIR, "step2_defect_final_schema.json")
    with open(final_filename, "w", encoding='utf-8') as f:
        json.dump(final_result_payload, f, indent=4, ensure_ascii=False)

    # 3.2 保存完整的演变历史总集
    history_payload = {
        "metadata": {
            "step": "Step 2: Defect Ontology Evolution History",
            "total_iterations": iteration
        },
        "evolution_history": all_evolution_logs
    }
    history_filename = os.path.join(HISTORY_DIR, "step2_defect_full_history.json")
    with open(history_filename, "w", encoding='utf-8') as f:
        json.dump(history_payload, f, indent=4, ensure_ascii=False)

    print(f"\n--- 建筑混凝土病害 Step 2 提取流程结束 ---")
    print(f"最终病害 Schema 已保存至: {final_filename}")
    print(f"完整演化历史已保存至: {history_filename}")


if __name__ == "__main__":
    run_step2_exploration()