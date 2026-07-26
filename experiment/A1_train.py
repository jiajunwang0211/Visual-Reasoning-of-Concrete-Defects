from ultralytics import YOLOWorld

if __name__ == '__main__':
    # 1. 加载 YOLO-World 预训练模型 (注意这里的类名变了，后缀带 -world)
    model = YOLOWorld('yolov8s-world.pt')

    print("🚀 启动 YOLO-World 开放词汇微调特化策略...")

    results = model.train(
        # 基础配置
        data='data.yaml',
        project='runs/detect',
        name='world_concrete_finetune',
        device=0,

        # ---------------------------------------------------
        # 🧠 【YOLO-World 专属调优核心参数】 🧠
        # ---------------------------------------------------

        # 1. 【新增】冻结网络主干 (Freeze) - 极其重要！
        # 冻结前 10 层（Backbone），只微调检测头。这样既能大幅降低显存消耗，
        # 又能防止 YOLO-World 强大的通用视觉提取能力被破坏。
        freeze=10,

        # 2. 优化器与学习率 (大幅调低)
        # 大模型微调绝对不能用 0.001 的学习率，会破坏权重！必须降一个数量级。
        optimizer='AdamW',
        lr0=0.0001,         # 💡 修改点：从 0.001 降为 1e-4
        weight_decay=0.01,

        # 3. 训练轮数 (缩短)
        # YOLO-World 领悟力极强，不需要 300 轮，通常 50-100 轮就能收敛。
        epochs=100,         # 💡 修改点：降为 100
        patience=20,        # 早停也相应缩短

        # 4. 批次大小与图像尺寸 (保持你的原样)
        batch=16,
        imgsz=640,

        # 5. 数据增强 (微调大模型时，过度增强会破坏图文对齐，适当减弱 mixup)
        mosaic=1.0,
        mixup=0.0,          # 💡 修改点：建议关闭 mixup。把两张病害图叠在一起容易让文本编码器混淆
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.3,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        workers=4
    )

    print("✅ YOLO-World 微调结束！")