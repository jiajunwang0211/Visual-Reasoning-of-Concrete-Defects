# Visual-Reasoning-of-Concrete-Defects
Visual Reasoning of Concrete Defects Based on Orthogonal Attribute Decoupling and Multimodal Large Model Fusion

本仓库是论文《Visual Reasoning of Concrete Defects Based on Orthogonal Attribute Decoupling and Multimodal Large Model Fusion》的开源代码实现。本项目对比了传统的“黑盒模式匹配”与本文提出的“白盒逻辑推理”在混凝土表面缺陷诊断中的性能表现。

## 1. 环境与配置

在运行代码前，请确保安装了必要的依赖环境。

### 1.1 安装依赖

pip install -r requirements.txt

### 1.2 API密钥配置
本项目的A3,A4,A5以及Ablation实验都依赖于大模型。在运行脚本前需要确保配置正确的Open API密钥。

## 2. 文件结构：
data/
├── data/                # 存放用于属性字典生成的输入数据与输出字典
├── experience/          # 存放各个多基线与消融实验的运行脚本及中间结果
└── eval/                # 存放统一的评估脚本与可视化代码

## 3. 项目模块
本项目的代码逻辑严格按照论文的实验流程分为三个核心模块：正交字典生成、核心对比实验（含消融）以及统一的评估指标计算。
### 3.1 模块一：字典生成
该模块包含了生成专家正交视觉属性字典的完整流水线，共分为4个子步骤，需按顺序执行。
| 步骤 | 脚本名称 | 功能描述 | 
| STEP1 | dict_gen_step1 | 根据病害名称生成对应的CQS问题 | 
| STEP2 | dict_gen_step2 | 根据生成的CQS问题进行属性维度发散 | 
| STEP3 | dict_gen_step3 | 对发散的属性维度进行正交化、去冗余等操作精炼 | 
| STEP4 | dict_gen_step4 | 使用精炼后的属性列表对测试类别进行重构测试 | 

### 3.2 模块二：多基线与消融实验 
该模块包含了各个实验以及消融实验的运行脚本。
| 实验组别 | 脚本名称 | 实验内容描述 | 
| A0 | A0_train | YOLOv8n-cls 黑盒监督微调分类测试 | 
| A1 | A1_train | YOLO_World 结合语义对齐的微调测试 | 
| A2 | A2 | SigLIP 纯全局图像分类零样本测试 | 
| A3 | A3 | GPT-4O 无约束零样本直接分类测试 | 
| A4 | A4 | GPT-4O 结合思维链 (CoT) 的零样本测试 | 
| A5 | A5 | 本文提出方法：GPT-4O 结合专家正交字典的白盒推理 | 
| Ablation | Ablation | 运行人为遮掩各个权重测试合理性的消融实验 | 

### 3.3 模块三：统一评估体系
该模块用于读取模块二中各模型生成的预测结果，并计算统一的评估指标，确保对比的公平性。
| 功能模块 | 脚本名称 | 评估指标说明 | 
| Ablation_bar | Ablation_bar | 用于将Ablation的结果进行可视化 | 
| confusion_matrix | confusion_matrix | 用于各个对比实验的混淆矩阵的可视化 | 
| eval | eval | 统一计算并输出所有实验组的Accuracy、Precision、Recall及F1-Score | 
| McNemar | McNemar | 用于进行实验组之间的显著性实验 | 

## 4. 实验结果
本研究在统一的测试环境下进行了对比实验，核心结果如下表所示：
| 实验组别 | 核心模型 | Accuracy | Precision | Recall | F1 | WAS |
| A0 | YOLOv8_cls | 83.77% | 63.83% | 43.85% | 51.98% | -- |
| A1 | YOLOv_World | 62.30% | 54.21% | 54.21% | 54.24% | -- |
| A2 | SigLIP | 45.31% | 51.36% | 38.00% | 42.07% | -- |
| A3 | GPT-4o | 64.91% | 65.01% | 62.21% | 60.86% | -- |
| A4 | GPT-4o+COT | 62.83% | 63.41% | 54.11% | 53.39% | -- |
| A5 | GPT-4o+专家正交字典 | 72.20% | 80.50% | 72.13% | 76.60% | 83.31 |


