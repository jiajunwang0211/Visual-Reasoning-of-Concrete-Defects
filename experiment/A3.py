import os
import json
import base64
import re
import csv
from openai import OpenAI

# ==========================================
# 1. 配置区域
# ==========================================
API_KEY = ""  # 请填入你的真实 API Key
BASE_URL = ""  # 你的 API Base URL
MODEL_NAME = "gpt-4o"  # 想要测试消融不同模型，只需改这里！(例如 qwen-vl-max, claude-3-5-sonnet-20240620)


# 该文件夹下应该有 15 个子文件夹，每个子文件夹的名字就是病害的名称（如"结壳"、"泛碱"等）
VAL_ROOT_DIR = r"C:\Users\admin\Desktop\15\data_final_noisy"  # 请替换为你的真实 val 文件夹路径

# 总结果保存路径
OUTPUT_CSV_PATH = r"C:\Users\admin\Desktop\小论文文献\实验\实验1_noisy.csv"

# 所有的可选病害类别
ALL_CATEGORIES = ["定向裂缝", "网状裂缝", "钢筋裸露", "剥落", "起皮", "崩解", "蜂窝", "水渍", "铁锈", "生物生长污渍",
                  "泛碱", "结壳", "钟乳石状析出", "气孔", "冷缝","无法判断"]

# 初始化大模型客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ==========================================
# 2. 辅助函数
# ==========================================
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_json_from_text(text):
    """终极防弹版 JSON 解析器"""
    try:
        cleaned_text = text.replace('```json', '').replace('```', '').strip()
        match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {}
    except Exception as e:
        print(f"JSON 解析失败: {e} \n原始文本为: {text}")
        return {}


# ==========================================
# 3. 实验1 核心运行主函数 (全量遍历版)
# ==========================================
def run_full_experiment_1():
    print(f"\n🚀 初始化【实验一：全验证集 Vanilla VLM 零样本测试】...")
    print(f"🧠 当前测试模型: {MODEL_NAME}")
    print("=" * 65)

    if not os.path.exists(VAL_ROOT_DIR):
        print(f"❌ 找不到验证集目录: {VAL_ROOT_DIR}")
        return

    experiment_results = []
    total_correct = 0
    total_images = 0

    # 统计每个类别的对错情况，方便后续画混淆矩阵
    class_stats = {cat: {"total": 0, "correct": 0} for cat in ALL_CATEGORIES}

    prompt_vanilla = f"""
    你是一个土木工程视觉诊断专家。
    请观察这张混凝土结构病害图片，并从以下给定的候选类别中，挑选出最准确的 1 个类别。

    【候选类别】: {ALL_CATEGORIES}

    要求：
    1. 不要输出任何推理过程、解释或多余的文字。
    2. 必须且只能输出一个严格的 JSON 格式结果。
    3. JSON 的键必须是 "Defect_Category"，值必须是你从候选类别中挑选出的名称。
    4. 如果因为图片像素低或者其他原因导致无法判断，则直接输出"无法判断"，不要随意猜测
    示例输出：
    {{"Defect_Category": "起皮"}}
    """

    # 🌟 核心修改：外层循环遍历 15 个类别的文件夹
    for category_folder in os.listdir(VAL_ROOT_DIR):
        category_path = os.path.join(VAL_ROOT_DIR, category_folder)

        # 只处理文件夹
        if not os.path.isdir(category_path):
            continue

        # 文件夹的名字就是当前的真实类别 (Ground Truth)
        true_category = category_folder

        # 如果文件夹名字不在我们的 15 类里，跳过（防止有隐藏文件夹）
        if true_category not in ALL_CATEGORIES:
            print(f"⚠️ 跳过未知文件夹: {true_category}")
            continue

        valid_extensions = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in os.listdir(category_path) if f.lower().endswith(valid_extensions)]

        print(f"\n📁 开始测试类别: 【{true_category}】 (共 {len(image_files)} 张图片)")

        # 内层循环：测试该类别文件夹下的所有图片
        for filename in image_files:
            total_images += 1
            class_stats[true_category]["total"] += 1

            image_path = os.path.join(category_path, filename)
            base64_image = encode_image(image_path)

            try:
                # 调用大模型
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt_vanilla},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]}
                    ],
                    temperature=0.0
                )

                raw_output = response.choices[0].message.content
                parsed_json = extract_json_from_text(raw_output)
                predicted_category = parsed_json.get("Defect_Category", "解析失败/未选择")

            except Exception as e:
                print(f"     [警告] 图片 {filename} API 调用或解析崩溃: {e}")
                predicted_category = "API调用失败"

            # 判断对错
            is_correct = (predicted_category == true_category)
            if is_correct:
                total_correct += 1
                class_stats[true_category]["correct"] += 1

            # 简洁打印，防止刷屏
            mark = "✅" if is_correct else "❌"
            print(f"  {mark} {filename} | 预测: {predicted_category}")

            # 记录本张图片的数据
            experiment_results.append({
                "Model_Name": MODEL_NAME,
                "Image_Name": filename,
                "True_Category": true_category,
                "Predicted_Category": predicted_category,
                "Is_Correct": is_correct
            })

    # ==========================================
    # 4. 数据保存与统计
    # ==========================================
    print("\n" + "=" * 65)
    print(f"🎉 实验一 ({MODEL_NAME} 全量裸考) 全部完成！")

    if experiment_results:
        try:
            csv_headers = experiment_results[0].keys()
            with open(OUTPUT_CSV_PATH, mode='w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writeheader()
                writer.writerows(experiment_results)
            print(f"💾 完整数据已保存至: {OUTPUT_CSV_PATH}")
        except Exception as e:
            print(f"\n❌ 保存 CSV 时出错: {e}")

        # 宏观与微观统计打印
        overall_acc = (total_correct / total_images) * 100 if total_images > 0 else 0
        print(f"\n📊 总体准确率: {overall_acc:.2f}% ({total_correct}/{total_images})")

        print("\n📈 各类别准确率拆解:")
        for cat in ALL_CATEGORIES:
            cat_total = class_stats[cat]["total"]
            cat_correct = class_stats[cat]["correct"]
            if cat_total > 0:
                acc = (cat_correct / cat_total) * 100
                print(f"  - {cat}: {acc:.1f}% ({cat_correct}/{cat_total})")
            else:
                print(f"  - {cat}: 无测试数据")
        print("=" * 65)


if __name__ == "__main__":
    run_full_experiment_1()