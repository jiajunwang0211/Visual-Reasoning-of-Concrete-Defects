from ultralytics import YOLOWorld

if __name__ == '__main__':

    model = YOLOWorld('yolov8s-world.pt')


    results = model.train(
        data='data.yaml',
        project='runs/detect',
        name='world_concrete_finetune',
        device=0,

        freeze=10,

        optimizer='AdamW',
        lr0=0.0001,
        weight_decay=0.01,


        epochs=100,
        patience=20,

        batch=16,
        imgsz=640,


        mosaic=1.0,
        mixup=0.0,         
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