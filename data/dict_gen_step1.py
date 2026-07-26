import os
import json
from openai import OpenAI

# ================= 配置区域 =================
API_KEY = ""
BASE_URL = ""
MODEL_NAME = "o1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120.0)

# 📁 定义输出文件夹路径
FINAL_DIR = "step1_final_result"

# 自动创建文件夹（如果已存在则跳过）
os.makedirs(FINAL_DIR, exist_ok=True)

# 实时更新的日志路径
REPORT_PATH = os.path.join(FINAL_DIR, "step1_evolution_history.json")

# ================= 提示词模板 =================
GLOBAL_OBJECTIVE = """
<Role>
你是一位专精于“开放世界建筑混凝土表面病害检测”的资深计算机视觉本体学家。你善于通过“起草-批判-修正”的元认知循环（Metacognitive Loop）来构建严密、正交的通用属性维度体系。
</Role>

<Objective>
构建一套“通用视觉原语本体（Visual Primitive Ontology）”。这套视觉属性维度本体必须满足：
1.  双空间解耦：严格区分 `S_int` (物体中心空间) 和 `S_ctx` (环境关联空间)。希望两个语义空间的属性维度都可以为最后的开放世界-开放词汇目标检测任务提供支持。
2.  视觉导向分析：在归纳两个语义空间的属性维度时，针对静态视觉图像，仅关注视觉可感知的视觉属性维度特征，忽略纯功能性或抽象的非视觉属性维度特征。
3.  正交完备性：维度之间无冗余元素（No Superfluous Elements），且能描述任何可见物体。
4.  零样本泛化力：必须能通过特定“未知物体集合”的对抗性压力测试。
</Objective>
"""

CQ_GENERATION_PROMPT = """
<Current Task: Step_1_CQ_Generation>
<Reference_Standard>
{GLOBAL_OBJECTIVE}
</Reference_Standard>

<Context>
你是一位专精于“建筑混凝土表面病害检测”的资深计算机视觉与土木工程交叉领域的本体学家。
请针对已知病害类别（C_known），提出几个“能力问题（Competency Questions, CQs）”。

这些问题必须能够指导模型进行下一步的双语义空间属性维度生成任务。可以从这几方面进行思考：
1. 仅凭单帧图像，将这些物体从复杂的背景中准确扣取出来。
2. 在已知类别之间建立视觉区分度。
3. 引导了 S_int 和 S_ctx 的解耦。
4. 严格限制在视觉可感知维度。
</Context>

<Input Data>
C_known: {class_list_str}
</Input Data>

<Iteration History>
{history_context}
</Iteration History>

<Output_Format>
请结合全局目标和历史反馈，生成最后优化后的CQs。
请严格按以下标准 JSON 格式用英文输出 CQs，务必确保双引号不嵌套、格式绝对正确：
{{"cqs": [
    {{
        "id": "CQ1",
        "question": "A specific description of the problem.",
        "strategic_goal": "This problem aims to solve 'background interference'..."
    }}
]}}
</Output_Format>
</Current Task: Step_1_CQ_Generation>
"""

CQ_JUDGE_PROMPT = """
<Current Task: Step_1_CQ_Audit>
<Reference_Standard>
{GLOBAL_OBJECTIVE}
</Reference_Standard>

<Context>
你现在担任“首席病害审计本体学家”。你拥有资深的“开放世界建筑混凝土表面病害检测”的深刻洞察能力，同时具备了审计员极度苛刻的逻辑检查能力。
请用极度苛刻的逻辑检查能力，评估以下 CQs 是否满足<Global Objective>作为前提的情况下对于开放世界中的建筑混凝土表面病害的问题能否指导下一步的双语义空间属性维度生成任务。

</Context>

<Input Data>
待评估的 CQs：
{current_cqs}
</Input Data>

<Audit_Criteria>
1. 必须纯视觉，严禁物理/化学化验指标。
2. 必须能区分 S_int 与 S_ctx。
3. 必须满足“完备”且“正交”的要求。
4. 如果满足，判定 true；只要有一条不满足，判定 false，并给出具体修改建议（如要求拆分、补充特定干扰的题目等）。
</Audit_Criteria>

<Output_Format>
请务必运用“思维链”逻辑，先对逐条进行批判分析，最后再下定论。
请务必严格按以下 JSON 格式返回：
{{
    "is_perfect": true,
    "rationale": "详细审计依据和修正建议..."
}}
</Output_Format>
</Current Task: Step_1_CQ_Audit>
"""


# ================= 辅助保存函数 =================
def save_evolution_history(iteration_count, all_logs, is_finished=False, is_perfect=False):
    """实时将当前所有日志覆写到同一个 JSON 文件中"""
    if not is_finished:
        status = "Running..."
    else:
        status = "Perfect" if is_perfect else "Max_Limit_Reached"

    history_report = {
        "experiment_info": "Step 1 Competency Questions Evolution (Concrete Defects)",
        "total_iterations_run": iteration_count,
        "final_status": status,
        "evolution_logs": all_logs
    }

    with open(REPORT_PATH, "w", encoding='utf-8') as f:
        json.dump(history_report, f, indent=4, ensure_ascii=False)


# ================= 主循环逻辑 =================
def run_automated_loop():
    classes = "定向裂缝，网状裂缝，钢筋裸露，剥落，起皮，崩解，蜂窝，水渍，铁锈，生物生长污渍，泛碱，结壳，钟乳石状析出，气孔，冷缝"
    max_iterations = 5  # 设置最大循环次数，防止死循环
    iteration_count = 0
    all_iteration_logs = []

    print("🚀 开始全自动 CQ 生成与审计迭代闭环...")

    while iteration_count < max_iterations:
        iteration_count += 1
        print(f"\n" + "=" * 40)
        print(f"🔄 [开始第 {iteration_count} 轮迭代]")
        print("=" * 40)

        # 1. 动态构建 History Context（只传上一轮的错题，防止上下文过长）
        if all_iteration_logs:
            last_log = all_iteration_logs[-1]
            history_context = (
                "【上一轮打回记录】\n"
                f"1. 你生成的错误 CQs 是：\n{json.dumps(last_log['cqs'], ensure_ascii=False)}\n\n"
                f"2. 审计官的拒绝理由与修改建议是：\n{last_log['audit']}\n\n"
                "请务必在这一轮中吸收教训，严格按照审计意见修正上述 CQs！"
            )
        else:
            history_context = "这是第一轮尝试，暂无历史失败记录。请直接根据基准要求生成最优 CQs。"

        # ---------------------------------------------------------
        # A. 生成器阶段
        # ---------------------------------------------------------
        print("🧠 [生成器] 正在思考并生成 CQs...")
        current_cqs = {}
        try:
            gen_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": CQ_GENERATION_PROMPT.format(
                    GLOBAL_OBJECTIVE=GLOBAL_OBJECTIVE,
                    class_list_str=classes,
                    history_context=history_context
                )}],
                max_tokens=4096,
                temperature=0.7
            )
            text_gen = gen_response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            current_cqs = json.loads(text_gen)
            print(f"✅ [生成器] 成功！")

        except Exception as e:
            print(f"❌ [生成器] 发生错误 (如 JSON 崩溃/超时): {e}")
            print("⚠️ 强行记录错误并进入下一轮重试...")
            all_iteration_logs.append({
                "iter": iteration_count,
                "cqs": {"error": "生成失败"},
                "is_perfect": False,
                "audit": "系统提示: 上一轮大模型输出 JSON 崩溃，请重新生成规范的 JSON。"
            })
            save_evolution_history(iteration_count, all_iteration_logs)
            continue

        # ---------------------------------------------------------
        # B. 判别器阶段
        # ---------------------------------------------------------
        print("⚖️  [判别器] 正在对刚刚生成的 CQs 进行严苛审查...")
        is_perfect = False
        rationale = "未审查完成"

        try:
            judge_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": CQ_JUDGE_PROMPT.format(
                    GLOBAL_OBJECTIVE=GLOBAL_OBJECTIVE,
                    current_cqs=json.dumps(current_cqs, ensure_ascii=False)
                )}],
                max_tokens=2048,
                temperature=0.1
            )
            text_judge = judge_response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            judge_data = json.loads(text_judge)

            is_perfect = judge_data.get("is_perfect", False)
            rationale = judge_data.get("rationale", "未给出具体理由")

            print(f"✅ [判别器] 审查完毕！")
            print(f"📝 审计结论: {'🌟 完美通过 (True)' if is_perfect else '❌ 驳回重做 (False)'}")
            print(f"💬 审计评语: {rationale}")

        except Exception as e:
            print(f"❌ [判别器] 发生错误: {e}")
            break

        # ---------------------------------------------------------
        # C. 记录并实时写入总文件
        # ---------------------------------------------------------
        all_iteration_logs.append({
            "iter": iteration_count,
            "cqs": current_cqs,
            "is_perfect": is_perfect,
            "audit": rationale
        })

        # 每次迭代完，立刻覆写 JSON（起到实时存档作用）
        save_evolution_history(iteration_count, all_iteration_logs,
                               is_finished=is_perfect or iteration_count >= max_iterations, is_perfect=is_perfect)
        print(f"💾 第 {iteration_count} 轮日志已实时更新至 {REPORT_PATH}")

        # 终止判断
        if is_perfect:
            print("\n🎉🎉🎉 太棒了！生成器通过了判别器的全部考核！迭代提前圆满结束。")

            # 单独提取最终成功的 CQs
            final_cq_path = os.path.join(FINAL_DIR, "step1_final_cqs_only.json")
            with open(final_cq_path, "w", encoding='utf-8') as f:
                json.dump(current_cqs, f, indent=4, ensure_ascii=False)
            print(f"🎯 [结果提取] 最终生成的最佳 CQs 已单独提取至: {final_cq_path}")
            break

        elif iteration_count >= max_iterations:
            print(f"\n⚠️ 已经达到最大迭代次数 ({max_iterations}次)，强制结束循环。")

    print(f"\n📁 [全流程结束] 完整的演化日志位于: {REPORT_PATH}")


if __name__ == "__main__":
    run_automated_loop()