# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


def run_mcnemar_test():
    # ==========================================
    # 1. 配置文件路径和列名（🔴 请在此处修改为您实际的 CSV 信息）
    # ==========================================
    # 填入你两个 CSV 文件的绝对或相对路径
    csv_a3_path = r"C:\Users\admin\Desktop\小论文文献\实验\实验2_CoT_NoDict.csv"  # 无字典(A3)的结果
    csv_a4_path = r"C:\Users\admin\Desktop\小论文文献\实验\实验2_final.csv"  # 有字典(A4)的结果

    # 填入 CSV 中对应的列名（假设两个表的真实标签列名一致）
    # 比如你的表头是 "true_label" 和 "predicted_label"
    true_col = "True_Category"
    pred_col = "Predicted_Category"

    # ==========================================
    # 2. 读取并对齐数据
    # ==========================================
    try:
        df_a3 = pd.read_csv(csv_a3_path, encoding='utf-8')
        df_a4 = pd.read_csv(csv_a4_path, encoding='gbk')
    except FileNotFoundError as e:
        print(f"❌ 找不到文件，请检查路径是否正确: {e}")
        return

    # 严谨校验：确保两份数据的长度完全一致
    if len(df_a3) != len(df_a4):
        print(f"⚠️ 警告：两份 CSV 文件的行数不一致！(A3有 {len(df_a3)} 行, A4有 {len(df_a4)} 行)")
        print("请确保它们是完全按照相同顺序排列的同一个验证集。")
        return

    true_labels = df_a3[true_col].values
    a3_preds = df_a3[pred_col].values
    a4_preds = df_a4[pred_col].values

    # ==========================================
    # 3. 计算预测对错的布尔矩阵
    # ==========================================
    a3_correct = (true_labels == a3_preds)
    a4_correct = (true_labels == a4_preds)

    # 统计 2x2 列联表的四个关键象限
    both_correct = np.sum(a3_correct & a4_correct)
    neither_correct = np.sum(~a3_correct & ~a4_correct)

    # 这两个是最关键的变量，代表两者的“分歧”
    a3_only_correct = np.sum(a3_correct & ~a4_correct)  # 字典导致模型犯错的数量
    a4_only_correct = np.sum(~a3_correct & a4_correct)  # 字典成功纠正大模型的数量

    # 构造 McNemar 检验需要的列联表
    table = [[both_correct, a3_only_correct],
             [a4_only_correct, neither_correct]]

    # ==========================================
    # 4. 执行严谨的 McNemar 检验
    # ==========================================
    # exact=True 代表使用精确二项分布检验，这对于 300 张图这样的小样本集是最严谨的标准
    result = mcnemar(table, exact=True)
    p_value = result.pvalue

    # ==========================================
    # 5. 打印完美对应回复信格式的报告
    # ==========================================
    print("\n" + "=" * 55)
    print("📊 核心消融实验 (A3 vs A4) 统计显著性报告")
    print("=" * 55)
    print(f"图片总数: {len(true_labels)} 张")
    print("-" * 55)
    print(f"✅ [A3和A4 都对] 的数量: {both_correct}")
    print(f"❌ [A3和A4 都错] 的数量: {neither_correct}")
    print(f"⚠️ [仅 A3 对, A4 错] (字典负面影响): {a3_only_correct}")
    print(f"🏆 [仅 A4 对, A3 错] (字典拯救幻觉): {a4_only_correct}")
    print("-" * 55)
    print(f"🎯 核心统计量 P-value: {p_value:.5f}")

    if p_value < 0.05:
        print("\n🎉 结论：p < 0.05，您的性能提升具有绝对的统计学显著性！")
        print("💡 请直接将以下话术填入回复信：")
        print(
            f'   "The exact McNemar\'s test yielded a p-value of {p_value:.4f} (p < 0.05), proving the statistically significant contribution of our Dictionary."')
    else:
        print("\n⚠️ 结论：p >= 0.05，统计上不显著。")
        print(
            "（如果出现这种情况，说明虽然总体准确率提高了，但两者分歧样本量太小，无法在严谨统计学上形成碾压局，需要考虑调整话术。）")


if __name__ == "__main__":
    run_mcnemar_test()

