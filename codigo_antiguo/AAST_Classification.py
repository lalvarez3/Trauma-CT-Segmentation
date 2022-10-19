import glob
import random
import tempfile
import os
import time
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
import imageio
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning
import torch
import torch.optim as optim
from ipywidgets.widgets import *
from monai.apps import load_from_mmar
from monai.apps.mmars import RemoteMMARKeys
from monai.config import DtypeLike, KeysCollection, print_config
from monai.config.type_definitions import NdarrayOrTensor
from monai.data import Dataset, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR
from monai.networks.utils import copy_model_state
from monai.optimizers import generate_param_groups
from monai.transforms import (
        AddChanneld,
        AsDiscrete,
        Compose,
        CropForegroundd,
        EnsureChannelFirstd,
        EnsureType,
        KeepLargestConnectedComponent,
        LoadImaged,
        Orientationd,
        RandCropByLabelClassesd,
        RandCropByPosNegLabeld,
        RandFlipd,
        RandGaussianNoised,
        RandRotate90d,
        RandRotated,
        RandScaleIntensityd,
        RandShiftIntensityd,
        RandSpatialCropSamplesd,
        Resized,
        ScaleIntensityRanged,
        Spacingd,
        SpatialPadd,
        ToTensord,
)
from monai.transforms.croppad.array import CenterSpatialCrop, SpatialPad
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.spatial.array import Resize
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.transforms.utils import create_grid
from monai.utils import InterpolateMode, ensure_tuple_rep, set_determinism
from monai.utils.deprecate_utils import deprecated_arg
from monai.utils.enums import TraceKeys
from monai.utils.module import optional_import
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type
from monai.visualize import blend_images, matshow3d
from pl_bolts.callbacks import PrintTableMetricsCallback
from tqdm import tqdm
import wandb
print_config()
from monai.losses import GeneralizedWassersteinDiceLoss
from monai.transforms.intensity.array import ScaleIntensityRange


class RemoveDicts(MapTransform, InvertibleTransform):

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.verbose = verbose

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            self.push_transform(d, key)
        # print(d["image_meta_dict"]["filename_or_obj"])
        a = {
            "image": d["image"],
            "label": d["label"],
        }
        if d.get('path', None):
            a["path"] = d["path"]
        else:
            a["path"] = d["image_meta_dict"]["filename_or_obj"]
        if self.verbose:
            print(a["path"])
        d = a
        return d

    def inverse(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            d[key] = d[key]
            # Remove the applied transform
            self.pop_transform(d, key)
        return d


class NNUnetScaleIntensity(MapTransform):

    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ScaleIntensityRange`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        a_min: intensity original range min.
        a_max: intensity original range max.
        b_min: intensity target range min.
        b_max: intensity target range max.
        clip: whether to perform clip after scaling.
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def _compute_stats(self, volume, mask):
        volume = volume.copy()
        mask = np.greater(mask, 0)  # get only non-zero positive pixels/labels
        volume = volume * mask
        volume = np.ma.masked_equal(volume, 0).compressed()
        if len(volume) == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        median = np.median(volume)
        mean = np.mean(volume)
        std = np.std(volume)
        mn = np.min(volume)
        mx = np.max(volume)
        percentile_99_5 = np.percentile(volume, 99.5)
        percentile_00_5 = np.percentile(volume, 00.5)
        # print(median, mean, std, mn, mx, percentile_99_5, percentile_00_5)
        return median, mean, std, mn, mx, percentile_99_5, percentile_00_5

    def __init__(
        self,
        keys: KeysCollection,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            stats = self._compute_stats(d[key], d["label"])
            d[key] = np.clip(d[key], stats[6], stats[5])
            d[key] = (d[key] - stats[1]) / stats[2]
        return d

class LoadNPZ(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
    ) -> None:
        super().__init__(keys)
        self.keys = keys

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        npy_data = np.load(d["path"])["data"]
        for i, key in enumerate(self.keys):
            d[key] = npy_data[i, :, :, :]

        return d

class KeepOnlyClass(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        classes_to_keep: List[int],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.class_to_keep = classes_to_keep

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            new_classes = []
            for class_to_keep in self.class_to_keep:
                new_classes.append(np.where(d[key] == class_to_keep, class_to_keep, 0))

            chans = np.stack(new_classes, axis=0)
            new_labels = np.max(chans, axis=0)

            d[key] = new_labels
        return d

BS = 1
PRETRAINED = False
TRANSFER_LEARNING = False
N_WORKERS_LOADER = 0
N_WORKERS_CACHE = 0
CACHE_RATE = 0
SEED = 42
SEED = 0
IMG_SIZE = (96, 96, 96)
VAL_SIZE = (256, 256, 256)
SAVE_PATH = "U:\\lauraalvarez\traumaAI\Liver_Segmentation\lightning_logs/"
run_idx = len(os.listdir("U:\\lauraalvarez\\traumaAI\Liver_Segmentation\\wandb"))
RUN_NAME = f"Predict_Segmentation_{run_idx+1}_UNETR"
pytorch_lightning.seed_everything(SEED)
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = "spleen_data"
print(root_dir)


class Net(pytorch_lightning.LightningModule):
    def __init__(self, train_img_size, val_img_size, n_output):
        super().__init__()
        self.train_img_size = train_img_size
        self.val_img_size = val_img_size
        self.n_output = n_output

        self._model = UNETR(
            in_channels=1,
            out_channels=3,
            img_size=IMG_SIZE,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            conv_block=True,
            dropout_rate=0.0,
        )

        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True, jaccard=True)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        self.train_dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        self.best_val_acc = 0
        self.best_val_epoch = 0

        self.save_hyperparameters()  # save hyperparameters

    def prepare_data(self):

        # images preprocessed from nnunet
        train_images = sorted(
            glob.glob(
                os.path.join(
                    "U:\\lauraalvarez",
                    "nnunet",
                    "nnUNet_cropped_data",
                    "Task510_LiverTraumaDGX",
                    "*.npz",
                )
            )
        )
        data_dicts = [{"path": image_name, "class": None} for image_name in train_images]

        test_images = sorted(
            glob.glob(
                os.path.join(
                    "U:\\lauraalvarez",
                    "nnunet",
                    "nnUNet_raw_data",
                    "Task510_LiverTraumaDGX",
                    "imagesTs",
                    "*.nii.gz",
                )
            )
        )

        test_labels = sorted(
            glob.glob(
                os.path.join(
                    "U:\\lauraalvarez",
                    "nnunet",
                    "nnUNet_raw_data",
                    "Task510_LiverTraumaDGX",
                    "labelsTs",
                    "*.nii.gz",
                )
            )
        )

        data_dicts_test = [
            {"image": image_name, "label": label_name, "class": None}
            for image_name, label_name in zip(test_images, test_labels)
        ]

        random.shuffle(data_dicts)
        train_files, val_files = data_dicts, data_dicts_test
        print("validation files", val_files)
        # print("training files", train_files)
        print("len(train_files)", len(train_files))
        print("len(validation files)", len(val_files))

        # set deterministic training for reproducibility
        set_determinism(seed=SEED)

        # define the data transforms

        # Computed for the randomCropByLabel transformation based on outputs
        if self.n_output == 3:
            ratios = [1, 1, 2]
        else:
            ratios = [1, 1]

        train_transforms = Compose(
            [
                LoadNPZ(keys=["image", "label"]),
                RemoveDicts(keys=["image", "label"]),
                KeepOnlyClass(keys=["label"], classes_to_keep=[0, 1, 3]),
                AddChanneld(keys=["image", "label"]),
                CropForegroundd(keys=["image", "label"], source_key="label"),
                ToTensord(keys=["image", "label"]),
                RandCropByLabelClassesd(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    num_classes=self.n_output,
                    num_samples=4,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.10,
                ),
            ]
        )

        # define the data transforms
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                RemoveDicts(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                NNUnetScaleIntensity(keys=["image"]),
                ToTensord(keys=["image", "label"]),
            ]
        )

        self.train_ds = Dataset(
            data=train_files,
            transform=train_transforms,
        )

        self.val_ds = Dataset(
            data=val_files,
            transform=val_transforms,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=0.1,
            momentum=0.99,
            nesterov=True,
            weight_decay=3e-05,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (1 - self.current_epoch / 1000) ** 0.9,
            last_epoch=self.current_epoch - 1,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=BS,
            shuffle=True,
            num_workers=N_WORKERS_LOADER,
            collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=N_WORKERS_LOADER,
        )
        return val_loader

    def predict_dataloader(self):
        predict_dataloader = torch.utils.data.DataLoader(
            self.val_ds, batch_size=1, shuffle=False, num_workers=N_WORKERS_LOADER
        )
        return predict_dataloader

    def forward(self, x):
        return self._model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)

        outputs = [self.post_pred(i) for i in decollate_batch(output)]

        labels_1 = [self.post_label(i) for i in decollate_batch(labels)]

        self.train_dice_metric(y_pred=outputs, y=labels_1)

        loss = self.loss_function(output, labels)
        tensorboard_logs = {"train_loss": loss.item()}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        dice_liver, dice_injure = self.train_dice_metric.aggregate()
        self.train_dice_metric.reset()
        # calculating average loss
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        # logging using tensorboard logger
        self.log("dice loss", avg_loss)
        # lnp.lnp(f"Dice Loss: {avg_loss}")
        # self.log("train background dice", dice_background)
        self.logger.experiment.log({"train liver dice", dice_liver})
        self.logger.experiment.log({"train injure dice", dice_injure})
        # self.log("train liver dice", dice_liver)
        # self.log("train injure dice", dice_injure)

        self.logger.experiment.log({"dice loss": avg_loss})

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        # filenames = batch["image_meta_dict"]["filename_or_obj"]
        filenames = batch["path"]
        post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=3)])
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=3)])

        roi_size = (96, 96, 96)
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward
        )
        loss = self.loss_function(outputs, labels)
        predicition = {
            "output": torch.nn.Softmax()(outputs),
            "image": images,
            "label": labels,
            "filename": filenames,
        }
        outputs = [post_pred(i) for i in decollate_batch(outputs)]

        labels = [post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        return {
            "dice_metric": self.dice_metric,
            "val_number": len(outputs),
            "prediction": predicition,
            "val_loss": loss,
        }

    def validation_epoch_end(self, outputs):

        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_loss = torch.tensor(val_loss / num_items)

        post_pred_dice = Compose(
            [
                EnsureType(),
                AsDiscrete(argmax=True, to_onehot=3),
                KeepLargestConnectedComponent(
                    [1, 2], is_onehot=True, independent=False, connectivity=2
                ),
            ]
        )
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=3)])
        dice_liver, dice_injure = self.dice_metric.aggregate()

        self.dice_metric.reset()
        tensorboard_logs = {
            "dice_metric": dice_injure,
        }

        predictions = [x["prediction"] for x in outputs]

        if (
            self.current_epoch % 25 == 0
            or dice_injure - self.best_val_dice > 0.1
            or self.current_epoch == self.trainer.max_epochs - 1
        ):
            test_dt = wandb.Table(
                columns=[
                    "epoch",
                    "filename",
                    "combined output",
                    "dice_value_liver",
                    "dice_value_injure",
                    "ground_truth",
                    "class predicted",
                ]
            )
            # figure = computeROC(predictions)
            # self.logger.experiment.log({"ROC": figure, "epoch": self.current_epoch})

            for i, prediction in enumerate(predictions):
                filename = os.path.basename(prediction["filename"][0])
                output_one = [
                    post_pred_dice(i) for i in decollate_batch(prediction["output"])
                ]
                label_one = [
                    post_label(i) for i in decollate_batch(prediction["label"])
                ]
                self.dice_metric(y_pred=output_one, y=label_one)
                dice_value_liver, dice_value_injure = self.dice_metric.aggregate()
                self.dice_metric.reset()
                class_predicted, _, ground_truth = get_classification_info(prediction)
                blended = make_gif(prediction, filename=i)
                row = [
                    self.current_epoch,
                    filename,
                    wandb.Image(blended),
                    dice_value_liver,
                    dice_value_injure,
                    int(ground_truth[0]),
                    class_predicted,
                ]
                test_dt.add_data(*row)

            self.logger.experiment.log({f"SUMMARY_EPOCH_{self.current_epoch}": test_dt})

        if dice_injure > self.best_val_dice:
            self.best_val_dice = dice_injure.item()
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current liver dice: {dice_liver:.4f}"
            f"current injure  dice: {dice_injure:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.log("dice_metric_liver", dice_liver.item(), prog_bar=True)
        self.log("dice_metric_injure", dice_injure.item(), prog_bar=True)
        self.log("val_loss", mean_val_loss, prog_bar=True)
        return {"log": tensorboard_logs}

    def predict_step(self, batch, batch_idx):
        print("predicting...")
        images, labels = batch["image"], batch["label"]
        post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=3)])
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=3)])
        roi_size = (160, 160, 160)
        sw_batch_size = 2
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward
        )
        predicition = {"output": outputs, "image": images, "label": labels}
        outputs = [post_pred(i) for i in decollate_batch(outputs)]

        labels = [
            post_label(torch.unsqueeze(i, 0)).squeeze() for i in decollate_batch(labels)
        ]
        dice_metric = self.dice_metric(y_pred=outputs, y=labels)
        return {"prediction": predicition, "dice_metric": dice_metric}



class Log_and_print:
    def __init__(self, run_name, tb_logger=None):
        self.tb_logger = tb_logger
        self.run_name = run_name
        self.str_log = "run_name" + "\n  \n"

    def lnp(self, tag):
        print(self.run_name, time.asctime(), tag)
        self.str_log += str(time.asctime()) + " " + str(tag) + "  \n"

    def dump_to_tensorboard(self):
        if not self.tb_logger:
            print("No tensorboard logger")
        self.tb_logger.experiment.add_text("log", self.str_log)


tempdir = tempfile.mkdtemp()
wandb.setup(wandb.Settings(mode="disabled", program=__name__, program_relpath=__name__, disable_code=True))
wandb.init(dir=tempdir)
# %%
try:
    wandb.finish()
except:
    print("Wandb not initialized")

# %%
lnp = Log_and_print(RUN_NAME)
lnp.lnp("Loggers start")
lnp.lnp("ts_script: " + str(time.time()))

wandb_logger = pytorch_lightning.loggers.WandbLogger(
    project="traumaIA",
    name=RUN_NAME,
)

lnp.lnp("MAIN callbacks")
l_callbacks = []
cbEarlyStopping = pytorch_lightning.callbacks.early_stopping.EarlyStopping(
    monitor="val_loss", patience=500, mode="max"
)
l_callbacks.append(cbEarlyStopping)


checkpoint_dirpath = SAVE_PATH + "checkpoints/"
checkpoint_filename = SAVE_PATH[:-1] + "_" + RUN_NAME + "_Best"
lnp.lnp("checkpoint_dirpath: " + checkpoint_dirpath)
lnp.lnp("checkpoint_filename: " + checkpoint_filename)
cbModelCheckpoint = pytorch_lightning.callbacks.ModelCheckpoint(
    monitor="dice_metric_injure",
    mode="max",
    dirpath=checkpoint_dirpath,
    filename=checkpoint_filename,
)
l_callbacks.append(cbModelCheckpoint)


checkpoint_dirpath = SAVE_PATH + "checkpoints/"
checkpoint_filename = SAVE_PATH[:-1] + "_" + RUN_NAME + "_Last"
lnp.lnp("checkpoint_dirpath: " + checkpoint_dirpath)
lnp.lnp("checkpoint_filename: " + checkpoint_filename)
cbModelCheckpointLast = pytorch_lightning.callbacks.ModelCheckpoint(
    every_n_epochs=1,
    dirpath=checkpoint_dirpath,
    filename=checkpoint_filename,
)
l_callbacks.append(cbModelCheckpointLast)

l_callbacks.append(PrintTableMetricsCallback())
from pytorch_lightning.callbacks import LearningRateMonitor

lr_monitor = LearningRateMonitor(logging_interval="epoch")
l_callbacks.append(lr_monitor)


# initialise the LightningModule
lnp.lnp(" Start Trainining process...")
net = Net(
    train_img_size=IMG_SIZE, val_img_size=VAL_SIZE, n_output=3
)  # .load_from_checkpoint("lightning_logs/checkpoints/lightning_logs_Predict_Segmentation_240_Last-v1.ckpt", train_img_size=IMG_SIZE, val_img_size=VAL_SIZE, n_output=3)
wandb_logger.watch(net)

# set up loggers and checkpoints
log_dir = os.path.join(root_dir, "logs")

# initialise Lightning's trainer.
trainer = pytorch_lightning.Trainer(
    gpus=[0],
    max_epochs=1000,
    fast_dev_run=False,
    auto_lr_find=False,
    logger=wandb_logger,
    enable_checkpointing=True,
    num_sanity_val_steps=1,
    log_every_n_steps=1,
    callbacks=l_callbacks,
)

# train
result_pred2 = trainer.fit(net)
wandb.alert(title="Train finished", text="The train has finished")
wandb.finish()