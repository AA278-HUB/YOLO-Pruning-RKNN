# prune.py
# 用于 YOLO11 自定义模型的剪枝脚本（基于 heyongxin233/YOLO-Pruning-RKNN 仓库逻辑）
# 模型：yolo11_MAFPN_modifyX_uniRepLKv5_v2.yaml（含 UniRepLKv5 大核 + BiFPN 等自定义模块）

from ultralytics import YOLO

# ====================== 配置路径 ======================
yaml_path = r'ultralytics/cfg/models/11/yolo11_MAFPN_modifyX_uniRepLKv5_v2.yaml'
data_path = r"Custom_dataset_cfg/vehicle_orientation_mini.yaml"
model_name = r"yolo11_MAFPN_modifyX_uniRepLKv5_v2"

# 加载模型（确保你的自定义模块已在 ultralytics/nn/modules/block.py 或对应文件中定义）
model = YOLO(yaml_path)


def prunetrain(
        train_epochs,
        prune_epochs=0,
        quick_pruning=True,
        prune_ratio=0.5,
        prune_iterative_steps=1,
        data=data_path,
        name=model_name,
        imgsz=640,
        batch=8,
        device=[0],
        sparse_training=False,
        workers=8,  # 数据加载 worker 数（4060 Ti 可设 8-12）
        patience=50,  # 早停 patience（避免过拟合）
):
    """
    剪枝训练函数
    quick_pruning=True: 直接剪枝 + 训练（适合快速测试）
    quick_pruning=False: 先正常训练 → 剪枝 → 微调（推荐，精度更高）
    """
    common_args = {
        'data': data,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'name': name,
        'workers': workers,
        'patience': patience,
        'amp': True,  # 混合精度，省显存
        'cache': False,  # 如果数据集小可设 'disk' 或 True
    }

    if not quick_pruning:
        assert train_epochs > 0 and prune_epochs > 0, \
            "Quick Pruning is not set. prune_epochs must > 0."

        print(f"Step 1: Normal training for {train_epochs} epochs (no pruning)")
        model.train(
            epochs=train_epochs,
            prune=False,
            sparse_training=sparse_training,
            **common_args
        )

        print(f"Step 2: Pruning + finetune for {prune_epochs} epochs")
        return model.train(
            epochs=prune_epochs,
            prune=True,
            prune_ratio=prune_ratio,
            prune_iterative_steps=prune_iterative_steps,
            sparse_training=sparse_training,
            **common_args
        )
    else:
        print(f"Quick Pruning: direct prune + train for {train_epochs} epochs")
        return model.train(
            epochs=train_epochs,
            prune=True,
            prune_ratio=prune_ratio,
            prune_iterative_steps=prune_iterative_steps,
            sparse_training=sparse_training,
            **common_args
        )

if __name__ == "__main__":
    # Windows: needed for multiprocessing when using DataLoader workers.
    from torch.multiprocessing import freeze_support
    freeze_support()

    # ====================== 推荐运行：Normal Pruning（针对 RTX 4060 Ti） ======================
    prunetrain(
        quick_pruning=False,  # False = 先训 → 剪枝 → 再训（精度最佳）
        train_epochs=3,  # 剪枝前训练轮数（测试先跑通）
        prune_epochs=2,  # 剪枝后微调轮数（测试先跑通）
        imgsz=640,
        batch=16,  # 自动 batch（最安全，适配 4060 Ti 8GB/16GB）
        # batch=8,                    # 如果自动失败，手动设 6-8（8GB 卡）或 12-16（16GB 卡）
        device=[0],
        name=model_name + '_prune40_iter2',  # 保存目录区分
        prune_ratio=0.40,  # 40% 起步（你的 UniRepLK 大核敏感，先别 50%）
        prune_iterative_steps=2,  # 分 2 次剪（更稳，精度掉得少）
        sparse_training=True,  # 开启 BN 稀疏正则化（强烈推荐！）
        workers=8,  # 数据加载线程
        patience=50  # 早停
    )

    # ====================== 备选：Quick Pruning（快速测试用，注释掉） ======================
    # prunetrain(
    #     quick_pruning=True,
    #     train_epochs=15,              # 直接剪 + 训
    #     imgsz=640,
    #     batch=-1,
    #     device=[0],
    #     name=model_name + '_quick_prune40',
    #     prune_ratio=0.40,
    #     prune_iterative_steps=2,
    #     sparse_training=True,
    #     workers=8,
    #     patience=50
    # )

    print("剪枝训练启动完成！")
    print("监控显存：运行时开终端输入 'watch -n 0.5 nvidia-smi'")
    print("剪枝后模型保存在 runs/train/.../weights/best.pt")
    print("如 OOM，尝试 batch=6 或 prune_ratio=0.3")