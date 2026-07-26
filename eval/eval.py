import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 1. 严谨读取数据
df = pd.read_csv(r"C:\Users\admin\Desktop\小论文文献\实验\实验2_final.csv", encoding='gbk')
# utf-8-sig
# 2. 核心清洗逻辑：抹平所有多余空格、换行符，强转字符串
y_true = df['True_Category'].astype(str).str.strip()
y_pred = df['Predicted_Category'].astype(str).str.strip()

# 3. 计算指标 (直接用于填入论文总表)
acc = accuracy_score(y_true, y_pred)
# 对于类别不平衡，一定要用 macro 宏平均！
p = precision_score(y_true, y_pred, average='macro', zero_division=0)
r = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

print("🏆 论文最终填报指标 🏆")
print("-" * 30)
print(f"准确率 (Overall ACC): {acc * 100:.2f}%")
print(f"精确率 (Macro-Precision): {p * 100:.2f}%")
print(f"召回率 (Macro-Recall): {r * 100:.2f}%")
print(f"平均 F1 (Macro-F1): {f1 * 100:.2f}%")
print("-" * 30)

# 4. 打印混淆矩阵详细数据 (用于写各个类别的分析)
print("\n📊 各病害类别详细诊断报告 📊")
report = classification_report(y_true, y_pred, zero_division=0, digits=4)
print(report)