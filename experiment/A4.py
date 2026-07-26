# -*- coding: utf-8 -*-
import os
import json
import base64
import re
from openai import OpenAI
import csv

# ==========================================
# 1. 配置区域 (请填入你的真实信息)
# ==========================================
API_KEY = ""
BASE_URL = ""
MODEL_NAME = "gpt-4o"

# 🌟 验证集目录保持不变
VAL_ROOT_DIR = r"C:\Users\admin\Desktop\15\data_final_noisy"
# 🌟 修改了输出文件名，避免覆盖你带字典的结果！
OUTPUT_CSV_PATH = r"C:\Users\admin\Desktop\小论文文献\实验\实验2_CoT_NoDict_noisy.csv"

ALL_CATEGORIES = ["定向裂缝", "网状裂缝", "钢筋裸露", "剥落", "起皮", "崩解", "蜂窝", "水渍", "铁锈", "生物生长污渍",
                  "泛碱", "结壳", "钟乳石状析出", "气孔", "冷缝"]
TEST_FOLDERS = ALL_CATEGORIES

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ==========================================
# 2. 辅助函数
# ==========================================
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_json_from_text(text):
    try:
        cleaned_text = text.replace('```json', '').replace('```', '').strip()
        match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {}
    except:
        return {}


# ==========================================
# 3. 实验：纯 CoT (无字典约束)
# ==========================================
def run_baseline_cot_no_dict():
    print(f"\n🚀 初始化【消融基线：GPT-4o + CoT (纯思维链，无字典约束)】...")
    print("=" * 65)

    if not os.path.exists(VAL_ROOT_DIR):
        print(f"❌ 找不到验证集目录: {VAL_ROOT_DIR}")
        return

    experiment_results = []
    total_correct = 0
    total_images = 0

    # 遍历 15 个病害类别文件夹
    for category_folder in os.listdir(VAL_ROOT_DIR):
        category_path = os.path.join(VAL_ROOT_DIR, category_folder)
        if not os.path.isdir(category_path) or category_folder not in TEST_FOLDERS:
            continue

        true_category = category_folder

        # 过滤出所有图片文件
        image_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # 调试用：如果只想测少量数据，可以解除下面这行的注释
        # image_files = image_files[:5]

        print(f"📁 开始测试类别: 【{true_category}】 (共 {len(image_files)} 张图片)")

        # 测试每张图片
        for filename in image_files:
            total_images += 1
            image_path = os.path.join(category_path, filename)
            base64_image = encode_image(image_path)

            # --- 🌟 纯 CoT 提示词设计 ---
            prompt_cot = f"""
            你是一个土木工程诊断专家。请仔细观察这张混凝土结构病害图片。
            候选类别严格限定为以下 15 类: {ALL_CATEGORIES}

            要求：
            1. 请运用你的内部常识与专业知识，一步步思考（Let's think step by step）。仔细分析图片中的视觉特征、形状、材质剥落情况等，并推导出它是哪种病害。
            2. 绝对不允许输出上述 15 类之外的名称。
            3. 必须且只能输出一个 JSON 格式的结果，格式如下:
            {{
                "Reasoning": "请在这里写下你详细的逐步诊断与推理过程",
                "Defect_Category": "推理出的最终类别名称"
            }}
            """

            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt_cot},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]}
                    ],
                    temperature=0.0  # 保持 0.0 确保结果可复现
                )
                pred_json = extract_json_from_text(resp.choices[0].message.content)
                predicted_category = pred_json.get("Defect_Category", "格式错误")
                reasoning = pred_json.get("Reasoning", "无推理过程")
            except Exception as e:
                predicted_category = "推理崩溃"
                reasoning = str(e)

            # 记录并打印结果
            is_correct = (predicted_category == true_category)
            if is_correct: total_correct += 1

            mark = "✅" if is_correct else "❌"
            print(f"  {mark} {filename} | 预测: {predicted_category}")
            if not is_correct:
                print(f"      🔍 [大模型错误幻觉推理]: {reasoning[:100]}...")  # 打印前100个字的推理看看它为啥错

            experiment_results.append({
                "Image_Name": filename,
                "True_Category": true_category,
                "Predicted_Category": predicted_category,
                "Is_Correct": is_correct,
                "CoT_Reasoning": reasoning  # 将原本的字典属性列替换为模型的推理过程
            })

    # --- 结果保存 ---
    if experiment_results:
        with open(OUTPUT_CSV_PATH, mode='w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=experiment_results[0].keys())
            writer.writeheader()
            writer.writerows(experiment_results)
        print("=" * 65)
        print(f"💾 纯CoT消融结果已保存至: {OUTPUT_CSV_PATH}")
        print(f"📊 实验总体准确率: {(total_correct / total_images) * 100:.2f}%")


if __name__ == "__main__":
    run_baseline_cot_no_dict()