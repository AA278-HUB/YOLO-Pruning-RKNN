from ultralytics import YOLO
path=r'ultralytics/cfg/models/11/yolo11_MAFPN_modifyX_uniRepLKv5_v2.yaml'
model = YOLO(path)
results = model.train(data='coco.yaml', epochs=10, imgsz=640, batch=8, device=[0], name='yolo11_MAFPN_modifyX_uniRepLKv5_v2', prune=False)
gi