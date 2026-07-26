import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix
from transformers import AutoProcessor, AutoModel

# ==========================================
# 0. 解决 matplotlib 画图中文乱码问题
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 用户用黑体
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # Mac 用户解开这行注释
plt.rcParams['axes.unicode_minus'] = False


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 正在使用计算设备: {device}")

    print("⏳ 正在加载 SigLIP 模型...")
    model_id = r"./siglip-so400m-patch14-384"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    # ==========================================
    # 1. 定义病害映射与“未检出”类别
    # ==========================================
    disease_mapping = [
        {"name": "崩解 (Disintegration)","prompt": "loose granular debris and exposed stones in a broken concrete depression"},
        {"name": "剥落 (Spalling)", "prompt": "deep rough pit with sharp edges and exposed inner stones"},
        {"name": "起皮 (Scaling)", "prompt": "shallow flaking concrete surface revealing rough exposed particles"},
        {"name": "冷缝 (Cold Joint)", "prompt": "distinct straight linear boundary between concrete pours"},
        {"name": "蜂窝 (Honeycomb)", "prompt": "dense cluster of deep holes with exposed stones on concrete"},
        {"name": "气孔 (BugHole)", "prompt": "cluster of small tiny dark pits on smooth concrete"},
        {"name": "钢筋裸露 (Exposed Rebar)", "prompt": "exposed rusty metal steel rebar"},
        {"name": "定向裂缝 (Oriented Cracking)", "prompt": "single continuous meandering crack line on concrete"},
        {"name": "网状裂缝 (Mapped Cracking)", "prompt": "interconnected web-like network of cracks on concrete"},
        {"name": "泛碱 (Efflorescence)", "prompt": "white powdery crystalline stain patch on concrete"},
        {"name": "结壳 (Crusting)", "prompt": "broken brittle crust revealing smooth underlying concrete paste"},
        {"name": "钟乳石状析出 (Stalactite)", "prompt": "white solid icicle shape hanging from concrete ceiling"},
        {"name": "水渍 (Water Stain)", "prompt": "dark brown stain patch with sharp coffee ring edge"},
        {"name": "铁锈 (Rust Stain)", "prompt": "reddish brown rust stain dripping down concrete surface"},
        {"name": "生物生长污渍 (Biological Stain)", "prompt": "green fuzzy moss or algae patch on concrete"}
    ]

    texts = ["a photo of a " + item["prompt"] for item in disease_mapping]
    class_names = [item["name"] for item in disease_mapping]
    class_names.append("未检出 (Missed)")  # 增加第16个类别：未检出

    CONF_THRESHOLD = 0.00

    ROOT_DIR = r"C:\Users\admin\Desktop\15\data_class\val"

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    y_true = []
    y_pred = []

    print(f"\n🚀 开始读取 {ROOT_DIR} 进行 SigLIP 环境评估...")

    with torch.no_grad():
        for subdir in os.listdir(ROOT_DIR):
            subdir_path = os.path.join(ROOT_DIR, subdir)
            if not os.path.isdir(subdir_path): continue

            true_idx = -1

            # 💡 极简双语匹配：从 "中文 (English)" 格式中自动提取中英文进行比对
            for i, item in enumerate(disease_mapping):
                name_str = item["name"]
                if " (" in name_str:
                    zh_part = name_str.split(" (")[0]
                    en_part = name_str.split(" (")[1].replace(")", "")

                    if zh_part in subdir or en_part.lower() in subdir.lower():
                        true_idx = i
                        break
                else:
                    if name_str in subdir:
                        true_idx = i
                        break

            if true_idx == -1:
                print(f"⚠️ 警告：跳过无法识别类别的文件夹 -> {subdir}")
                continue

            print(f"📂 正在分析类别: 【{subdir}】 -> 映射为: 【{class_names[true_idx]}】")
            images = [f for f in os.listdir(subdir_path) if f.lower().endswith(valid_extensions)]

            for img_name in images:
                img_path = os.path.join(subdir_path, img_name)

                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception as e:
                    continue

                inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt").to(device)
                outputs = model(**inputs)

                probs = torch.sigmoid(outputs.logits_per_image)
                max_prob, pred_idx = torch.max(probs, dim=1)

                y_true.append(true_idx)

                if max_prob.item() < CONF_THRESHOLD:
                    y_pred.append(15)  # 未检出
                else:
                    y_pred.append(pred_idx.item())

    # ==========================================
    # 3. 计算准确率
    # ==========================================
    print("\n" + "=" * 50)
    print(f"📊 SigLIP 环境评估报告 (阈值: {CONF_THRESHOLD})")
    print("=" * 50)

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total = len(y_true)
    overall_acc = correct / total if total > 0 else 0
    print(f"🌟 SigLIP 总体分类准确率: {overall_acc * 100:.2f}% ({correct}/{total})")


    cm_normalized = confusion_matrix(y_true, y_pred, labels=range(15), normalize='true')

    annot_labels = []
    # 2. 循环只跑 15x15
    for i in range(15):
        row_labels = []
        for j in range(15):
            val = cm_normalized[i, j]
            if np.isnan(val) or val == 0.0:
                row_labels.append("")
            else:
                row_labels.append(f"{val:.2f}")
        annot_labels.append(row_labels)

    plt.figure(figsize=(15, 11))

    # 3. 热力图直接画 15x15
    sns.heatmap(cm_normalized, annot=annot_labels, fmt='', cmap='Oranges',
                xticklabels=class_names[:15], yticklabels=class_names[:15],  # 去掉未检出的名字
                annot_kws={"size": 10}, cbar_kws={'shrink': .8})

    plt.title(f'SigLIP 15x15 Confusion Matrix (No Threshold)', fontsize=18, pad=20)
    plt.xlabel('模型预测类别 (Predicted Label)', fontsize=14, labelpad=15)
    plt.ylabel('真实类别 (True Label)', fontsize=14, labelpad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = 'SigLIP_withoutNoisy_Confusion_Matrix_15x15.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n15x15 小数混淆矩阵图已保存为: {save_path}")

if __name__ == "__main__":
    main()