from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('yolov8s.pt')

    results = model.train(

        data='data.yaml',
        project='runs/detect',
        name='concrete_few_shot_1',
        device=0,


        epochs=300,
        patience=50,


        batch=16,
        imgsz=640,


        optimizer='AdamW',
        lr0=0.001,
        weight_decay=0.01,


        mosaic=1.0,
        mixup=0.15,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.3,
        hsv_h=0.015,
        fliplr=0.5,
        hsv_s=0.7,
        hsv_v=0.4,

        dropout=0.1,
        workers=4
    )

