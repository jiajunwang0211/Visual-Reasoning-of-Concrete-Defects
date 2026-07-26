from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载预训练模型 (少样本绝对不能从头练，必须用官方预训练权重)
    # 建议用 yolov8s.pt (Small版本)，比 n 版本稍微聪明一点，适合 15 个类别
    model = YOLO('yolov8s.pt')

    print("🚀 启动少样本特化训练策略...")

    results = model.train(
        # 基础配置
        data='data.yaml',
        project='runs/detect',
        name='concrete_few_shot_1',
        device=0,  # 如果用CPU改为 'cpu'

        # 1. 训练轮数与早停策略
        epochs=300,  # 少样本需要多看几遍数据，拉高到 300 轮
        patience=50,  # 早停机制：如果连续 50 轮准确率都没提升，自动停止，防止过拟合死记硬背

        # 2. 批次大小
        batch=16,  # 根据你的显存调整(显存不够改8或4)。不要设太大，小 batch 带来的噪声有助于跳出局部最优
        imgsz=640,  # 默认640，如果你的病害非常小(比如Bughole气孔)，显存够可以尝试拉高到 800 或 1024

        # 3. 优化器与学习率
        optimizer='AdamW',  # AdamW 优化器自带权重衰减，比默认的 SGD 更适合小数据集，收敛更快
        lr0=0.001,  # 初始学习率稍微调小一点，防止预训练权重被破坏
        weight_decay=0.01,  # L2 正则化，强迫模型不要过度依赖某几个特征，防过拟合

        # 4. 疯狂的数据增强 (Data Augmentation) - 少样本的救星
        mosaic=1.0,  # 100% 开启马赛克增强 (把4张图拼成1张，极大增加背景复杂度和目标数量)
        mixup=0.15,  # 15% 概率开启图像混合 (把两张图半透明叠在一起，让模型学到更本质的特征)
        degrees=10.0,  # 随机旋转图片 ±10度 (混凝土病害角度多变)
        translate=0.1,  # 随机平移 10%
        scale=0.5,  # 随机缩放图片大小 (±50%)
        shear=2.0,  # 随机剪切 2度
        perspective=0.0,  # 透视变换 (混凝土表面多为平面，可以不设或者设很小如 0.0001)
        flipud=0.3,  # 30% 概
        hsv_h=0.015,  # 随机调整色调 (让模型对光率上下翻转 (除了钟乳石Stalactite有方向，大部分病害上下翻转成立)
        fliplr=0.5,  # 50% 概率左右翻转照/颜色不那么敏感)
        hsv_s=0.7,  # 随机调整饱和度
        hsv_v=0.4,  # 随机调整明度

        # 5. 其他正则化策略
        dropout=0.1,  # 随机丢弃 10% 的神经元，让模型变成“残疾”去学，学出来的特征更鲁棒 (仅YOLOv8部分版本支持)
        workers=4  # 开启多线程加载数据 (如果 Windows 报错，改成 workers=0)
    )

    print("✅ 训练结束！请前往 runs/detect/concrete_few_shot_1 查看 results.png 训练曲线！")