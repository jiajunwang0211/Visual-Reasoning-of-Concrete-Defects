import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib

# ==========================================
# 1. 路径与基础配置
# ==========================================
CSV_FILE_PATH = r"C:\Users\admin\Desktop\小论文文献\实验\实验1.csv"
# 【修改】保存为 TIFF 高清格式，符合 Elsevier 提交要求
# 1. 把后缀改为 .pdf
OUTPUT_IMAGE_PATH = r"C:\Users\admin\Desktop\小论文文献\实验\实验1.pdf"

# ==========================================
# 2. 标签系统分离设计 (解决 CSV内中文 与 图表纯英文 的冲突)
# ==========================================
# (A) 用于计算的 CSV 内部真实中文标签 (保持与你数据源一致)
CSV_TRUE_CATEGORIES = [
    "定向裂缝", "网状裂缝", "钢筋裸露", "剥落", "起皮",
    "崩解", "蜂窝", "水渍", "铁锈", "生物生长污渍",
    "泛碱", "结壳", "钟乳石状析出", "气孔", "冷缝"
]
CSV_PRED_CATEGORIES = CSV_TRUE_CATEGORIES + ["无法判断"]

# (B) 用于 Elsevier 论文图表展示的纯英文标签 (严格对齐)
ENG_TRUE_CATEGORIES = [
    "Oriented Cracking", "Mapped Cracking", "Exposed Rebar", "Spalling", "Scaling",
    "Disintegration", "Honeycomb", "Water Stain", "Rust Stain", "Biological Stain",
    "Efflorescence", "Crusting", "Stalactite", "Bughole", "Cold Joint"
]
ENG_PRED_CATEGORIES = ENG_TRUE_CATEGORIES + ["Unidentified"]  # 无法判断译为 Unidentified

# ==========================================
# 3. 字体与画图风格配置 (Elsevier 规范)
# ==========================================
# 【修改】使用学术标准的 Times New Roman 或 Arial，绝不能用中文字体
matplotlib.rcParams['font.family'] = 'Times New Roman'
# 也可以用：matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['axes.unicode_minus'] = False
# 调整全局字体大小基准
matplotlib.rcParams.update({'font.size': 12})


def plot_normalized_confusion_matrix():
    print("正在读取 CSV 数据...")
    try:
        df = pd.read_csv(CSV_FILE_PATH, encoding='utf-8-sig', on_bad_lines='skip')
        # df = pd.read_csv(CSV_FILE_PATH, encoding='gbk', on_bad_lines='skip')
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return

    # 提取真实标签和预测标签
    y_true = df['True_Category'].astype(str).tolist()
    y_pred = df['Predicted_Category'].astype(str).tolist()

    print("正在计算归一化混淆矩阵...")

    # 第一步：计算完整矩阵时，使用 CSV_PRED_CATEGORIES (中文) 去匹配数据
    cm_full = confusion_matrix(y_true, y_pred, labels=CSV_PRED_CATEGORIES)

    # 第二步：切掉最后一行无用行 (16 -> 15行)
    cm = cm_full[:len(CSV_TRUE_CATEGORIES), :]

    # 第三步：计算归一化矩阵
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = cm.astype('float') / (row_sums + np.finfo(float).eps)

    # ==========================================
    # 4. 绘制热力图 (国际学术规范版)
    # ==========================================
    # 调整画布比例，适应英文较长的单词
    plt.figure(figsize=(16, 12))

    # 【修改】注入英文标签 (ENG_PRED_CATEGORIES, ENG_TRUE_CATEGORIES)
    ax = sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                     xticklabels=ENG_PRED_CATEGORIES, yticklabels=ENG_TRUE_CATEGORIES,
                     annot_kws={"size": 11, "family": "Times New Roman"},  # 矩阵内数字也强制英文字体
                     vmin=0, vmax=1, cbar_kws={'label': 'Normalized Accuracy'})

    # 【修改】坐标轴标签全英文化
    # 注意：学术论文原图内通常不写 Title，因为会在 Word/LaTeX 下方写 Figure Caption
    # plt.title("Confusion Matrix of Large Language Model", fontsize=16, pad=20)

    plt.xlabel("Predicted Category", fontsize=14, labelpad=15, fontweight='bold')
    plt.ylabel("True Category", fontsize=14, labelpad=15, fontweight='bold')

    # 调整刻度字号和旋转角度 (英文较长，45度角更美观)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    plt.tight_layout()

    # ==========================================
    # 5. 保存与展示
    # ==========================================
    # 【修改】保存为 TIFF 格式，并设置 600 DPI 的超高分辨率
    # 2. 修改最后保存的代码（矢量图不需要设置 DPI）
    plt.savefig(OUTPUT_IMAGE_PATH, format='pdf', bbox_inches='tight')
    print(f"✅ 已生成极小体积的高清矢量图: {OUTPUT_IMAGE_PATH}")

    plt.show()


if __name__ == "__main__":
    plot_normalized_confusion_matrix()
