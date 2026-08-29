# Visual Reasoning of Concrete Defects Based on Orthogonal Attribute Decoupling and Multimodal Large Model Fusion

This repository provides the official implementation for the paper *"Visual Reasoning of Concrete Defects Based on Orthogonal Attribute Decoupling and Multimodal Large Model Fusion"*. The project benchmarks traditional "black-box pattern matching" against the proposed "white-box logical reasoning" approach for concrete surface defect diagnosis.

## 1. Setup & Environment

Please ensure all required dependencies are installed before executing the scripts.

### 1.1 Installation

```bash
pip install -r requirements.txt
```
### 1.2 API Key Configuration
Experiments in groups A3, A4, A5, and the Ablation studies rely on Multimodal Large Language Models (MLLMs). Ensure a valid OpenAI API key is properly configured in your environment variables prior to execution.

## 2. Directory Structure

```text
data/
├── data/        # Stores input data and output dictionaries for attribute dictionary generation
├── experience/  # Stores execution scripts and intermediate results for baselines and ablation studies
└── eval/        # Stores unified evaluation scripts and visualization utilities
```

## 3. Project Modules
The project architecture strictly adheres to the experimental workflow described in the paper, categorized into three core modules: Orthogonal Dictionary Generation, Baseline & Ablation Experiments, and Unified Metric Evaluation.

### 3.1 Module 1: Dictionary Generation
This module implements the complete pipeline for constructing the expert orthogonal visual attribute dictionary, executed in 4 sequential steps:
| Step | Script Name | Description |
| :--- | :--- | :--- |
| **STEP1** | `dict_gen_step1` | Generates corresponding CQS questions based on defect names. |
| **STEP2** | `dict_gen_step2` | Expands attribute dimensions according to generated CQS questions. |
| **STEP3** | `dict_gen_step3` | Refines expanded attribute dimensions via orthogonalization and redundancy removal. |
| **STEP4** | `dict_gen_step4` | Reconstructs and evaluates target defect classes using the refined attribute list. |

### 3.2 Module 2: Baselines & Ablation Studies

| Group | Script Name | Description |
| :--- | :--- | :--- |
| **A0** | `A0_train` | YOLOv8n-cls black-box supervised fine-tuning classification test. |
| **A1** | `A1_train` | YOLO_World fine-tuning test with semantic alignment. |
| **A2** | `A2` | SigLIP zero-shot global image classification test. |
| **A3** | `A3` | GPT-4o unconstrained zero-shot direct classification test. |
| **A4** | `A4` | GPT-4o zero-shot classification with Chain-of-Thought (CoT) prompting. |
| **A5** | `A5` | **Proposed Method**: GPT-4o white-box reasoning guided by expert orthogonal dictionary. |
| **Ablation** | `Ablation` | Ablation experiments masking specific attribute weights to evaluate model rationality. |

### 3.3 Module 3: Unified Evaluation Framework

| Function | Script Name | Description |
| :--- | :--- | :--- |
| **Ablation_bar** | `Ablation_bar` | Visualizes ablation study performance metrics. |
| **confusion_matrix** | `confusion_matrix` | Generates confusion matrices across comparative baselines. |
| **eval** | `eval` | Computes and outputs Accuracy, Precision, Recall, and F1-Score for all test groups. |
| **McNemar** | `McNemar` | Performs statistical significance testing (McNemar's test) between experimental groups. |

### 4. Experimental Results

| Group | Core Model | Accuracy | Precision | Recall | F1-Score | WAS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A0** | YOLOv8_cls | 83.77% | 63.83% | 43.85% | 51.98% | -- |
| **A1** | YOLO_World | 62.30% | 54.21% | 54.21% | 54.24% | -- |
| **A2** | SigLIP | 45.31% | 51.36% | 38.00% | 42.07% | -- |
| **A3** | GPT-4o | 64.91% | 65.01% | 62.21% | 60.86% | -- |
| **A4** | GPT-4o + CoT | 62.83% | 63.41% | 54.11% | 53.39% | -- |
| **A5** | GPT-4o + Expert Orthogonal Dict | **72.20%** | **80.50%** | **72.13%** | **76.60%** | **83.31** |