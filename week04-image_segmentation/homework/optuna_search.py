import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
torch.cuda.set_device(0)


import optuna
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
sys.path.append(str(SRC_DIR))

from datamodules.coco_person import CocoPersonDataModule
from models.unet_late_fusion import UNetResNet50LateFusion
from person_seg_module import PersonSegModule


def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [8])
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
    num_epochs = trial.suggest_int("num_epochs", 30, 50)

    depth_num_layers = trial.suggest_categorical("depth_num_layers", [4])
    depth_mid_channels = trial.suggest_categorical("depth_mid_channels", [16])
    depth_out_channels = trial.suggest_categorical("depth_out_channels", [4])
    depth_num_resblocks = trial.suggest_categorical("depth_num_resblocks", [1])
    final_channels = trial.suggest_categorical("final_channels", [16, 32, 64, 128])
    activation = trial.suggest_categorical("activation", ["silu", "relu"])

    hparams = {
        "batch_size": batch_size,
        "learning_rate": lr,
        "num_epochs": num_epochs,
        "depth_num_layers": depth_num_layers,
        "depth_mid_channels": depth_mid_channels,
        "depth_out_channels": depth_out_channels,
        "depth_num_resblocks": depth_num_resblocks,
        "final_channels": final_channels,
        "activation": activation,
    }

    print(
        f"\nTrial {trial.number}: bs={batch_size}, lr={lr:.6f}, ep={num_epochs}, "
        f"d_layers={depth_num_layers}, d_mid={depth_mid_channels}, "
        f"d_out={depth_out_channels}, d_blocks={depth_num_resblocks}, "
        f"final={final_channels}, "
        f"activation={activation}, "
    )

    data_dir = CURRENT_DIR / "data/coco"
    train_ann = data_dir / "annotations/instances_val2017_train_labeled500.json"
    val_ann = data_dir / "annotations/instances_val2017_val.json"
    test_ann = data_dir / "annotations/instances_val2017_test.json"
    depth_dir = data_dir / "depth/dpt_hybrid"

    datamodule = CocoPersonDataModule(
        data_dir=str(data_dir),
        train_ann=str(train_ann),
        val_ann=str(val_ann),
        test_ann=str(test_ann),
        depth_dir=str(depth_dir),
        batch_size=batch_size,
        image_size=(360, 480),
        num_workers=4,
        pin_memory=True,
    )

    model = UNetResNet50LateFusion(
        num_classes=2,
        pretrained=True,
        depth_num_layers=depth_num_layers,
        depth_mid_channels=depth_mid_channels,
        depth_out_channels=depth_out_channels,
        depth_num_resblocks=depth_num_resblocks,
        final_channels=final_channels,
        activation=activation,
    )

    lit_model = PersonSegModule(
        model=model, learning_rate=lr, weight_decay=1e-4, optimizer="adamw"
    )

    logger = TensorBoardLogger("optuna_logs", name="trial_4", version=trial.number)

    hparams_path = Path("optuna_logs/trial_test_hparams") / f"version_{trial.number}" / "hparams.yaml"
    hparams_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hparams_path, "w") as f:
        yaml.dump(hparams, f)

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        enable_checkpointing=False,
        logger=logger,
        enable_progress_bar=False,
        check_val_every_n_epoch=5,
        log_every_n_steps=50,
    )

    try:
        trainer.fit(lit_model, datamodule=datamodule)
        metrics = trainer.test(lit_model, datamodule=datamodule, verbose=False)
        score = metrics[0]["test/mIoU"]
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        score = 0.0
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=20)
    args = parser.parse_args()

    pl.seed_everything(42)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    base_config_path = CURRENT_DIR / "configs/unet_late_fusion.json"
    with open(base_config_path, "r") as f:
        config = json.load(f)

    config["data"]["batch_size"] = trial.params["batch_size"]
    config["training"]["learning_rate"] = trial.params["learning_rate"]
    config["training"]["max_epochs"] = trial.params["num_epochs"]

    config["model"]["depth_num_layers"] = trial.params["depth_num_layers"]
    config["model"]["depth_mid_channels"] = trial.params["depth_mid_channels"]
    config["model"]["depth_out_channels"] = trial.params["depth_out_channels"]
    config["model"]["depth_num_resblocks"] = trial.params["depth_num_resblocks"]
    config["model"]["final_channels"] = trial.params["final_channels"]
    config["model"]["activation"] = trial.params["activation"]

    output_path = CURRENT_DIR / "configs/unet_late_fusion_best.json"
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved best config to {output_path}")


if __name__ == "__main__":
    main()
