import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib

# ==========================================
# 1. 字体配置 (严格遵守 Elsevier 规范)
# ==========================================
# 绝不能使用 SimHei，统一替换为学术通用的 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams.update({'font.size': 12})  # 全局字号基准

# ==========================================
# 2. 建立中英文类别映射字典 (核心逻辑)
# ==========================================
# 用于将你电脑里的中文文件夹名，自动转换为合规的英文图片文件名
CATEGORY_MAP = {
    "定向裂缝": "Oriented_Cracking", "网状裂缝": "Mapped_Cracking", "钢筋裸露": "Exposed_Rebar",
    "剥落": "Spalling", "起皮": "Scaling", "崩解": "Disintegration",
    "蜂窝": "Honeycomb", "水渍": "Water_Stain", "铁锈": "Rust_Stain",
    "生物生长污渍": "Biological_Stain", "泛碱": "Efflorescence", "结壳": "Crusting",
    "钟乳石状析出": "Stalactite", "气孔": "Bughole", "冷缝": "Cold_Joint"
}

# ==========================================
# 3. 动态路径处理
# ==========================================
# 将这里的路径替换为你电脑上真实的 CSV 绝对路径
file_path = r'C:\Users\admin\Desktop\小论文文献\实验\图片\钟乳石状析出\消融实验结果.csv'

# 提取文件夹路径和中文类别名
folder_path = os.path.dirname(file_path)
cn_category_name = os.path.basename(folder_path)  # 例如: "剥落"

# 获取对应的英文名 (如果没找到，默认用 Unknown)
en_category_name = CATEGORY_MAP.get(cn_category_name, "Unknown_Category")

print(f"正在处理类别: {cn_category_name} -> {en_category_name}")

# 读取数据 (保留原来的 gbk 编码读取你的中文 CSV)
df = pd.read_csv(file_path, encoding='gbk')

# ==========================================
# 4. 数据处理与英文标签替换
# ==========================================
df['Accuracy'] = df['Is_Correct'].astype(int) * 100
grouped_data = df.groupby('Experiment_Group').agg({'WASC_Score': 'mean', 'Accuracy': 'mean'}).reset_index()

# 【修改】X 轴组别全英文化
group_labels = ['Baseline\n(All Features)', 'Mask_W3\n(Core)', 'Mask_W1\n(Medium)', 'Mask_W0\n(Noise)']
grouped_data['Experiment_Group'] = group_labels

# ==========================================
# 5. 绘制双 Y 轴图表
# ==========================================
# 稍微调整画布比例，Elsevier 双栏排版通常推荐较紧凑的尺寸
fig, ax1 = plt.subplots(figsize=(9, 4.2))
x = np.arange(len(group_labels))
width = 0.4

# 左侧 Y 轴 (WASC 柱状图)
color1 = '#4C72B0'
bars = ax1.bar(x, grouped_data['WASC_Score'], width, color=color1, alpha=0.8, label='Average WASC Score (%)')
display_name = en_category_name.replace('_', ' ')
ax1.set_xlabel(f'{display_name} Masking Experiment Group', fontsize=14, fontweight='bold')
ax1.set_ylabel('Feature Extraction Score (WASC %)', color=color1, fontsize=14, fontweight='bold')
ax1.set_ylim(0, 115)  # 顶部稍微留点余量给数字标签
ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(group_labels, fontsize=12)

for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontsize=11, color=color1, family='Times New Roman')

# 右侧 Y 轴 (Accuracy 折线图)
ax2 = ax1.twinx()
color2 = '#C44E52'
line = ax2.plot(x, grouped_data['Accuracy'], color=color2, marker='o', markersize=10, linewidth=3, label='Classification Accuracy (%)')
ax2.set_ylabel('Classification Accuracy (%)', color=color2, fontsize=14, fontweight='bold')
ax2.set_ylim(-5, 115)
ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)

for i, txt in enumerate(grouped_data['Accuracy']):
    ax2.annotate(f'{txt:.1f}%', (x[i], grouped_data['Accuracy'][i] - 7), ha='center', fontsize=11, color=color2, fontweight='bold', family='Times New Roman')

# 图例设置
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=11)
ax1.grid(axis='y', linestyle='--', alpha=0.4)

# 【修改】强制删去图内标题
# Elsevier 规定标题必须写在论文正文的 Figure Caption 中
# plt.title(...)  <-- 已删除

plt.tight_layout()

# ==========================================
# 6. 保存为出版级矢量图 (PDF)
# ==========================================
# 保存文件名为纯英文，例如: Spalling_Ablation_Result.pdf
save_filename = f'{en_category_name}_Ablation_Result.pdf'
save_path = os.path.join(r"C:\Users\admin\Desktop\小论文文献\论文图片\遮蔽性实验", save_filename)

# 矢量图格式，完美适配 Overleaf
plt.savefig(save_path, format='pdf', bbox_inches='tight')
plt.show()

print(f"🎉 搞定！图表已符合 Elsevier 纯英文+无标题标准！")
print(f"矢量图(PDF)已成功保存至: {save_path}")
