"""
YOLO11 segmentation training for maize microspot (viable / unviable).
To improve from ~70-75% accuracy: augmentation is enabled; consider larger model, more data, and tuning conf/iou.
"""
from ultralytics import YOLO
from types import SimpleNamespace

cfg = SimpleNamespace(
    model=SimpleNamespace(
        expt_name="maize_microspot_yolo11s_seg_ep500",
        model_name="yolo11s-seg.yaml",
        pretrained="yolo11s-seg.pt",
        project="./runs/yolov11s_seg",
    ),
    train=SimpleNamespace(
        random_seed=42,
        epochs=500,
        lr0=0.01,
        lrf=0.0001,
        batch=6,
        patience=100,
        input_size=2048,
        # Accuracy improvements: cosine LR and label smoothing often help segmentation
        cos_lr=True,
        label_smoothing=0.1,
    ),
    data=SimpleNamespace(
        config_file="/data/rajhans-workspace/microspot_data/yolo_dataset_correct/data_config.yaml",
    ),
    augment=SimpleNamespace(
        degrees=180.0,
        translate=0.2,
        scale=0.7,
        shear=10.0,
        flipud=0.5,
        mixup=0.15,
        copy_paste=0.15,
        mosaic=0.15,
        hsv_h=0.005,
        hsv_s=0.3,
        hsv_v=0.3,
    ),
)

# Load pretrained YOLO (recommended for training)
model = YOLO(cfg.model.pretrained)

# Full training WITH augmentation (was previously commented out — critical for accuracy)
results = model.train(
    data=cfg.data.config_file,
    epochs=cfg.train.epochs,
    patience=cfg.train.patience,
    imgsz=cfg.train.input_size,
    name=cfg.model.expt_name,
    project=cfg.model.project,
    batch=cfg.train.batch,
    lr0=cfg.train.lr0,
    lrf=cfg.train.lrf,
    cos_lr=getattr(cfg.train, "cos_lr", True),
    label_smoothing=getattr(cfg.train, "label_smoothing", 0.1),
    augment=True,
    degrees=cfg.augment.degrees,
    shear=cfg.augment.shear,
    translate=cfg.augment.translate,
    scale=cfg.augment.scale,
    hsv_h=cfg.augment.hsv_h,
    hsv_s=cfg.augment.hsv_s,
    hsv_v=cfg.augment.hsv_v,
    flipud=cfg.augment.flipud,
    mixup=cfg.augment.mixup,
    copy_paste=cfg.augment.copy_paste,
    mosaic=cfg.augment.mosaic,
)
print(results)

# Validation (use same or slightly larger imgsz as typical deployment)
metrics = model.val(data=cfg.data.config_file, batch=cfg.train.batch, imgsz=2880, plots=True)
print(metrics)


