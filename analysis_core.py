# -*- coding: utf-8 -*-
"""
基于关联规则挖掘的政府采购需求合理性评价算法实现
Implementation of Government Procurement Requirement Rationality Evaluation Algorithm
Based on Association Rule Mining

Author: Tang Jun
Date: 2025
Description: 
    This script implements the "Ideal Procurement Requirement Target Model" (IPRTM).
    It includes data preprocessing, Apriori association rule mining, 
    four-state classification, and intelligent realignment logic.
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

class ProcurementAnalyzer:
    def __init__(self, necessity_threshold=0.7, tolerance_threshold=0.3):
        """
        初始化评价模型
        :param necessity_threshold: 功能必要性阈值 (lambda), 对应论文设定的 0.7
        :param tolerance_threshold: 配置适宜性容忍阈值 (delta), 对应论文设定的 30%
        """
        self.lambda_th = necessity_threshold
        self.delta_th = tolerance_threshold
        self.rules = None
        self.benchmark_dict = {} # 存储行业基准值

    def preprocess_data(self, file_path):
        """
        数据预处理与离散化
        读取采购数据，将连续型参数（如内存大小）映射为区间标签
        """
        # 模拟读取数据，实际使用时请替换为 pd.read_excel(file_path)
        print(f"正在加载数据: {file_path}...")
        # 假设数据结构：Transaction_ID, Item, Value
        # 这里仅作逻辑演示，实际上传时请确保您的数据格式与代码读取方式一致
        pass

    def mine_association_rules(self, transactions_df, min_support=0.05):
        """
        核心步骤1：关联规则挖掘
        对应论文 2.2 节
        """
        print("开始关联规则挖掘...")
        # 独热编码 (One-hot encoding)
        basket = (transactions_df.groupby(['Transaction_ID', 'Item'])['Value']
                  .count().unstack().reset_index().fillna(0)
                  .set_index('Transaction_ID'))
        
        # 将数值转换为布尔值
        basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
        
        # 使用 Apriori 算法计算频繁项集
        frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
        
        # 生成关联规则
        # Metric 'lift' 用于表征功能必要性
        self.rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.1)
        print(f"挖掘完成，共生成 {len(self.rules)} 条规则。")
        return self.rules

    def calculate_deviation(self, param_name, param_value, benchmark_value):
        """
        核心步骤2：计算数值偏离度 (Dev)
        对应论文公式 (4)
        """
        if benchmark_value == 0:
            return 0
        dev = (param_value - benchmark_value) / benchmark_value
        return dev

    def classify_state(self, lift_value, deviation_value):
        """
        核心步骤3：IPRTM 四态分类判定
        对应论文 表3-1
        """
        abs_dev = abs(deviation_value)
        
        # 理想态 (Ideal): 必要性高 且 偏离度低
        if lift_value >= self.lambda_th and abs_dev <= self.delta_th:
            return "理想态 (Ideal)"
        
        # 控标态 (Rigged): 必要性低 (无论数值偏离与否，通常伴随非标数值)
        # 注意：论文中提到控标态通常共识度低
        elif lift_value < self.lambda_th:
            return "控标态 (Rigged)"
        
        # 冗余态 (Redundant): 必要性高 但 数值显著偏高
        elif lift_value >= self.lambda_th and deviation_value > self.delta_th:
            return "冗余态 (Redundant)"
        
        # 瓶颈态 (Bottleneck): 必要性高 但 数值显著偏低
        elif lift_value >= self.lambda_th and deviation_value < -self.delta_th:
            return "瓶颈态 (Bottleneck)"
        
        return "未知 (Unknown)"

    def analyze_parameters(self, params_data):
        """
        主分析流程
        :param params_data: 包含参数名、提取值、基准值、关联Lift值的DataFrame
        """
        results = []
        for index, row in params_data.iterrows():
            lift = row['Lift'] # 来自关联规则挖掘结果
            val = row['Value']
            bench = row['Benchmark'] # 来自行业基准库（如中位数）
            
            # 计算偏离度
            dev = self.calculate_deviation(row['Param'], val, bench)
            
            # 判定状态
            state = self.classify_state(lift, dev)
            
            results.append({
                'Parameter': row['Param'],
                'Value': val,
                'Lift': lift,
                'Deviation': dev,
                'State': state
            })
            
        return pd.DataFrame(results)

    def intelligent_realignment(self, analysis_result):
        """
        核心步骤4：智能归位纠偏机制
        对应论文 3.4 节
        """
        print("正在生成智能归位建议...")
        recommendations = []
        
        for index, row in analysis_result.iterrows():
            state = row['State']
            current_val = row['Value']
            
            # 针对 冗余态 (Redundant) -> 基准对齐
            if "Redundant" in state:
                # 建议修正为基准值的上限 (Benchmark * 1.3)
                # 对应论文：压缩过度配置空间
                target_val = row['Value'] / (1 + row['Deviation']) * (1 + self.delta_th)
                advice = f"建议降级配置。将 {current_val} 调整至 {target_val:.2f} (基准上限)"
                savings = "预计节约预算 > 30%"
                
            # 针对 控标态 (Rigged) -> 敏感性移除
            elif "Rigged" in state:
                advice = f"检测到排他性风险。建议移除参数 {current_val} 或改为 '≥ {current_val}' 的区间表述。"
                savings = "提升竞争开放度"
                
            else:
                advice = "保持现状"
                savings = "-"
                
            recommendations.append(advice)
            
        analysis_result['Recommendation'] = recommendations
        return analysis_result

# ==========================================
# 示例运行模块 (Example Usage)
# ==========================================
if __name__ == "__main__":
    # 1. 实例化分析器
    analyzer = ProcurementAnalyzer(necessity_threshold=0.7, tolerance_threshold=0.3)
    
    # 2. 模拟输入数据 (对应论文 4.2 节实证数据)
    # 这里手动构造几个典型案例用于演示代码逻辑
    mock_data = pd.DataFrame([
        # 案例1: 理想态 (32核CPU)
        {'Param': 'CPU_Cores', 'Value': 32, 'Benchmark': 32, 'Lift': 1.8}, 
        # 案例2: 控标态 (80GB内存) - Lift低，非标数值
        {'Param': 'Memory_GB', 'Value': 80, 'Benchmark': 64, 'Lift': 0.19},
        # 案例3: 冗余态 (1024GB内存) - Lift高(技术成熟)，但Dev极大
        {'Param': 'Memory_GB', 'Value': 1024, 'Benchmark': 64, 'Lift': 1.69},
        # 案例4: 瓶颈态 (8GB内存) - Lift高，但Dev为负
        {'Param': 'Memory_GB', 'Value': 8, 'Benchmark': 64, 'Lift': 1.2},
    ])
    
    print("--- 输入参数数据 ---")
    print(mock_data)
    
    # 3. 执行分析
    final_report = analyzer.analyze_parameters(mock_data)
    
    # 4. 执行纠偏
    final_report = analyzer.intelligent_realignment(final_report)
    
    print("\n--- 最终评价与纠偏报告 (对应论文表4-1与表4-2) ---")
    print(final_report[['Parameter', 'Value', 'Lift', 'Deviation', 'State', 'Recommendation']])

    # 5. 可视化 (生成类似论文图4-1的靶心图)
    # 实际项目中可取消注释以生成图表
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=final_report, x='Lift', y='Deviation', hue='State', s=100)
    plt.axvline(x=0.7, color='k', linestyle='--') # 必要性阈值
    plt.axhline(y=0.3, color='k', linestyle=':')  # 容忍上限
    plt.axhline(y=-0.3, color='k', linestyle=':') # 容忍下限
    plt.title("IPRTM Analysis Result")
    plt.show()
    """