# %% [markdown]
# #### Setup imports

# %%
import numpy as np
from monai.data.image_reader import ImageReader, ITKReader
from ipywidgets.widgets import *
import ipywidgets as widgets

import matplotlib.pyplot as plt
import pytorch_lightning
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    EnsureType,
    EnsureChannelFirstd,
    RandFlipd,
    RandRotated,
    ToTensord,
    Resized,
    RandSpatialCropSamplesd,
    RandRotate90d,
    RandShiftIntensityd,
    KeepLargestConnectedComponent,
    RandCropByPosNegLabeld,
    RandCropByLabelClassesd,
    RandSpatialCropd
)

import wandb
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceFocalLoss, GeneralizedDiceLoss
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, Dataset, LMDBDataset
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import tempfile
import os
import glob
from tqdm import tqdm
import numpy as np
from monai.data import DataLoader
import os
import glob
from monai.transforms.spatial.array import Resize

from copy import deepcopy
from enum import Enum
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.networks.layers import AffineTransform
from monai.networks.layers.simplelayers import GaussianFilter
from monai.transforms.croppad.array import CenterSpatialCrop, SpatialPad
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.spatial.array import (
    Resize,
)
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.transforms.utils import create_grid
from monai.utils import (
    InterpolateMode,
    ensure_tuple_rep,
)
from monai.utils.deprecate_utils import deprecated_arg
from monai.utils.enums import TraceKeys
from monai.utils.module import optional_import
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type
from monai.apps import load_from_mmar
from monai.apps.mmars import RemoteMMARKeys
from monai.networks.utils import copy_model_state
from monai.optimizers import generate_param_groups
import torch.optim as optim
import time
from pl_bolts.callbacks import PrintTableMetricsCallback
from monai.visualize import matshow3d, blend_images
import imageio
print_config()
from monai.losses import GeneralizedWassersteinDiceLoss
import random
import shutil

# %% [markdown]
# #### Preprocessing

# %%
class KeepOnlyClass(MapTransform, InvertibleTransform):

    def __init__(
        self,
        keys: KeysCollection,
        class_to_keep: int,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.class_to_keep = class_to_keep

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            self.push_transform(d, key)
            d[key] = np.where(d[key] == 255, 1, 0)
            # values = d[key]
            # print("NP UNIQUE", np.unique(values))
            # print('BEFORE EYE', d[key].shape)
            # n_values = np.max(values) + 1
            # d[key]= np.eye(n_values)[values]
            # print('AFTER EYE', d[key].shape)
            # d[key] = np.squeeze(d[key])
            # print('AFTER squeeze', d[key].shape)
            # print("NP UNIQUE AFTER", np.unique(d[key]))
            # if d[key].ndim == 2:
            #     zeros = np.zeros(d[key].shape)
            #     d[key] = np.stack([d[key], zeros], axis=-1)
            #     print(d[key].shape)
            # print("NP UNIQUE AFTER AFTER 0", np.unique(d[key][:,:,0]))
            # print("NP UNIQUE AFTER AFTER 1", np.unique(d[key][:,:,1]))

        return d

# %%
class RemoveAlpha(MapTransform):

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key == "label":
                d[key] = d[key][...,:1]
            else:
                d[key] = d[key][...,:3]
        return d

# %%
class ToGrayScale(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        normalize: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.normalize = normalize

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = d[key][..., :1]
            if self.normalize:
                d[key] = d[key] / 255

        return d


# %% [markdown]
# #### Flags

# %%
PRETRAINED = False
TRANSFER_LEARNING = False
N_WORKERS_LOADER = 0
N_WORKERS_CACHE = 0
CACHE_RATE = 0
SEED = 42
BS = 16

# %% [markdown]
# #### Define the LightningModule

# %%
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = "spleen_data"
print(root_dir)

# %%
class Net(pytorch_lightning.LightningModule):
    def __init__(self, train_img_size, val_img_size, n_output):
        super().__init__()
        self.train_img_size = train_img_size
        self.val_img_size = val_img_size
        self.n_output = n_output

        self._model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=self.n_output,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True, jaccard=True)
        self.post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        self.train_dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.save_hyperparameters() # save hyperparameters

        # self.logger.expe.init(self.hparams)

    def prepare_data(self):
        # set up the correct data path
        train_images = sorted(
            glob.glob(
                os.path.join(
                    "u://", "lauraalvarez", "data", "vascular_injuries", "png", "imagesTr", "*.png"
                )
            )
        )
        train_labels = [img.replace('imagesTr', 'labelsTr') for img in train_images]

        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(train_images, train_labels)
        ]

        test_images = sorted(
            glob.glob(
                os.path.join(
                    "U:\\", "lauraalvarez", "data", "vascular_injuries", "png", "imagesTs", "*.png"
                )
            )
        )
        test_labels = [img.replace('imagesTs', 'labelsTs') for img in test_images]

        data_dicts_test = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(test_images, test_labels)
        ]

        random.shuffle(data_dicts)
        train_files, val_files = data_dicts[:1000], data_dicts_test[:10]
        # print("validation files", val_files)
        # print("training files", train_files)
        print("len(train_files)", len(train_files))
        print("train labels 0 y 1",train_files[0], train_files[1])
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
                LoadImaged(keys=["image", "label"], reader='pilreader'),
                RemoveAlpha(keys=["image", "label"]),
                KeepOnlyClass(keys=["label"], class_to_keep=255),
                ToGrayScale(keys=["image"], normalize=True),
                EnsureChannelFirstd(keys=["image", "label"]),
                Resized(keys=["image", "label"], spatial_size=(256,256)),
                # RandSpatialCropd(keys=["image", "label"], roi_size=self.train_img_size,random_size=True),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
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
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                ToTensord(keys=["image", "label"]),
            ]
        )

        # define the data transforms
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                RemoveAlpha(keys=["image", "label"]),
                KeepOnlyClass(keys=["label"], class_to_keep=255),
                ToGrayScale(keys=["image"], normalize=True),
                EnsureChannelFirstd(keys=["image", "label"]),
                ToTensord(keys=["image", "label"]),
            ]
        )


        self.train_ds = Dataset(
            data=train_files,
            transform=train_transforms,
        #     cache_dir=os.path.join(
        #             "/mnt/chansey/", "lauraalvarez", "data", "vascular_injuries", "png", "cache"
        #         )
        )

        self.val_ds = Dataset(
            data=val_files,
            transform=val_transforms,
            # cache_dir=os.path.join(
            #         "/mnt/chansey/", "lauraalvarez", "data", "vascular_injuries", "png", "cache"
            #     )
        )

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self._model.parameters(), 1e-3)
    #     if PRETRAINED:
    #         optimizer = torch.optim.AdamW(self.params, lr=5e-4, weight_decay=1e-5)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, factor=0.05, patience=100, min_lr=1e-7)
    #     return {'optimizer':optimizer, 'lr_scheduler':scheduler, 'monitor':"val_loss", "interval": "epoch"}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self._model.parameters(), 1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (1 - self.current_epoch / 1000) ** 0.9,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
            "interval": "epoch",
        }

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=BS, shuffle=True, num_workers=N_WORKERS_LOADER,
            collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_ds, batch_size=1, shuffle=False, num_workers=N_WORKERS_LOADER,
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
        # lnp.lnp(images.shape)
        # output = self.forward(images)
        roi_size = (96, 96)
        sw_batch_size = 4
        output = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward
        )
        outputs = [self.post_pred(i) for i in decollate_batch(output)]

        labels_1 = [
            self.post_label(i) for i in decollate_batch(labels)
        ]

        self.train_dice_metric(y_pred=outputs, y=labels_1)

        loss = self.loss_function(output, labels)
        tensorboard_logs = {"train_loss": loss.item()}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        dice_injure = self.train_dice_metric.aggregate()
        dice_injure = dice_injure[0]
        self.train_dice_metric.reset()
        # calculating average loss
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        # logging using tensorboard logger
        self.log("dice loss", avg_loss)
        # self.log("train liver dice", dice_liver)
        self.log("train injure dice", dice_injure)

        self.logger.experiment.log({"dice loss": avg_loss})

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        # filenames = batch["path"]
        post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
        roi_size = (96, 96)
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward
        )
        loss = self.loss_function(outputs, labels)
        predicition ={"output": torch.nn.Softmax()(outputs), "image": images, "label": labels}
        outputs = [post_pred(i) for i in decollate_batch(outputs)]

        labels = [
            post_label(i) for i in decollate_batch(labels)
        ]
        self.dice_metric(y_pred=outputs, y=labels)
        return {"dice_metric": self.dice_metric, "val_number": len(outputs), "prediction": predicition, "val_loss": loss}


    def validation_epoch_end(self, outputs):

        val_loss, num_items = 0, 0
        mean_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        # for output in outputs:
        #     val_loss += output["val_loss"].sum().item()
        #     num_items += output["val_number"]
        # mean_val_loss = torch.tensor(val_loss / num_items)

        post_pred_dice = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2), KeepLargestConnectedComponent([1], is_onehot=True, independent=False)])
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
        dice_injure = self.dice_metric.aggregate()
        dice_injure = dice_injure[0]
        lnp.lnp(dice_injure)

        self.dice_metric.reset()
        tensorboard_logs = {
            "dice_metric": dice_injure,
        }
        predictions = [x["prediction"] for x in outputs]

        if self.current_epoch % 25 == 0 or dice_injure - self.best_val_dice > 0.1 or self.current_epoch == self.trainer.max_epochs - 1:
            # test_dt = wandb.Table(columns = ['epoch', 'filename', 'combined output','dice_value_liver', 'dice_value_injure', 'ground_truth', 'class predicted'])
            # figure = computeROC(predictions)
            # self.logger.experiment.log({"ROC": figure, "epoch": self.current_epoch})

            for i, prediction in enumerate(predictions):
                # filename = os.path.basename(prediction["filename"][0])
                output_one = [post_pred_dice(i) for i in decollate_batch(prediction["output"])]
                label_one = [post_label(i) for i in decollate_batch(prediction["label"])]
                self.dice_metric(y_pred=output_one, y=label_one)
                dice_value_injure = self.dice_metric.aggregate()
                self.dice_metric.reset()
                # class_predicted, _, ground_truth = get_classification_info(prediction)
                # blended = make_gif(prediction, filename=i)
                # row = [self.current_epoch, filename, wandb.Image(blended), dice_value_liver, dice_value_injure, int(ground_truth[0]), class_predicted]
                # test_dt.add_data(*row)

            # self.logger.experiment.log({f"SUMMARY_EPOCH_{self.current_epoch}" : test_dt})

        if dice_injure > self.best_val_dice:
            self.best_val_dice = dice_injure.item()
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            # f"current liver dice: {dice_liver:.4f}"
            f"current injure  dice: {dice_injure:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        # self.log("dice_metric_liver", dice_liver.item(), prog_bar=True)
        self.log("dice_metric_injure", dice_injure.item(), prog_bar=True)
        self.log("val_loss", mean_val_loss, prog_bar=True)
        return {"log": tensorboard_logs}

    def predict_step(self, batch, batch_idx):
        print('predicting...')
        images, labels = batch["image"], batch["label"]
        post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
        roi_size = (160, 160, 160)
        sw_batch_size = 2
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward
        )
        predicition ={"output": outputs, "image": images, "label": labels}
        outputs = [post_pred(i) for i in decollate_batch(outputs)]

        labels = [
            post_label(torch.unsqueeze(i, 0)).squeeze() for i in decollate_batch(labels)
        ]
        dice_metric = self.dice_metric(y_pred=outputs, y=labels)
        return {"prediction": predicition, "dice_metric": dice_metric}


# %% [markdown]
# ## Create Gif Function

# %%
def make_gif(prediction, filename):
    def _save_gif(volume, filename):
        volume = volume.astype(np.float64) / np.max(volume) # normalize the data to 0 - 1
        volume = 255 * volume # Now scale by 255
        volume = volume.astype(np.uint8)
        path_to_gif = os.path.join("gifs", f'{filename}.gif')
        if not os.path.exists("gifs"):
            print("Creating gifs directory")
            os.mkdir("gifs")
        imageio.mimsave(path_to_gif, volume)
        return path_to_gif
    post_pred_blending = Compose([EnsureType(), AsDiscrete(argmax=False),KeepLargestConnectedComponent([1,2], is_onehot=False, independent=False)])
    prediction["output"] = [post_pred_blending(i) for i in decollate_batch(prediction["output"])]
    selected = {"output": prediction["output"][0], "image": prediction["image"][0], "label": prediction["label"][0]}

    selected = Resized(keys=["image", "label"], spatial_size=(160, 160, 160))(selected)
    selected = Resized(keys=["output"], spatial_size=(160, 160, 160))(selected)

    selected = {"output": selected["output"].unsqueeze(0), "image": selected["image"].unsqueeze(0), "label": selected["label"].unsqueeze(0)}

    # print('true label:', selected['label'].shape)
    pred = torch.argmax(selected['output'], dim=1).detach().cpu().numpy()
    true_label = selected['label'][0].detach().cpu().numpy()
    image = selected['image'][0].cpu().numpy()
    # print('true label:', true_label.shape)

    blended_true_label = blend_images(image, true_label, alpha=0.7)
    blended_final_true_label = torch.from_numpy(blended_true_label).permute(1,2,0,3)

    blended_prediction = blend_images(image, pred, alpha=0.7)
    blended_final_prediction = torch.from_numpy(blended_prediction).permute(1,2,0,3)

    volume_pred = blended_final_prediction[:,:,:,:]
    volume_label = blended_final_true_label[:,:,:,:]
    volume_pred = np.squeeze(volume_pred).permute(3,0,1,2).cpu()
    volume_label = np.squeeze(volume_label).permute(3,0,1,2).cpu()
    volume_img = torch.tensor(image).permute(3,1,2,0).repeat(1,1,1,3).cpu()

    volume = torch.hstack((volume_img, volume_pred, volume_label))

    volume_path = _save_gif(volume.numpy(), f"blended-{filename}")


    return volume_path

# %%
def get_classification_info(prediction):
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

    ground_truth = [
            1 if (post_label(i)[2,:,:,:].cpu() == 1).any() else 0 for i in decollate_batch(prediction['label'])
        ]

    test = prediction['output'].cpu()
    prediction_1 = torch.argmax(test, dim=1)

    class_2_mask = (prediction_1 == 2).cpu()
    if class_2_mask.any():
        prediction = torch.max(test[:,2,:,:,:]).item()
    else:
        prediction = np.max(np.ma.masked_array(test[:,2,:,:,:], mask=class_2_mask))

    unique_values = torch.unique(prediction_1)
    predicted_class = 1 if 2 in unique_values else 0

    return predicted_class, prediction, ground_truth

def computeROC(predictions):
    from sklearn.metrics import roc_curve, auc # roc curve tools

    g_truths = []
    preds = []
    for prediction in predictions:
        _, predict, ground_truth = get_classification_info(prediction)
        g_truths.extend(ground_truth)
        preds.append(predict)

    preds = np.asarray(preds)
    ground_truth = np.asarray(g_truths)
    fpr, tpr, _ = roc_curve(g_truths, preds)
    roc_auc = auc(fpr,tpr)

    fig, ax = plt.subplots(1,1)
    ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    return fig

# %% [markdown]
# ## Run the training

# %% [markdown]
# ### Variables

# %%
SEED = 0
IMG_SIZE = (96,96,96)
VAL_SIZE = (256,256,256)
SAVE_PATH = "lightning_logs/"
run_idx = len(os.listdir("wandb"))
RUN_NAME = f"Predict_Segmentation_VI_{run_idx+1}"
pytorch_lightning.seed_everything(SEED)

# %% [markdown]
# ### Loggers

# %%
mmar = {
    RemoteMMARKeys.ID: "clara_pt_liver_and_tumor_ct_segmentation_1",
    RemoteMMARKeys.NAME: "clara_pt_liver_and_tumor_ct_segmentation",
    RemoteMMARKeys.FILE_TYPE: "zip",
    RemoteMMARKeys.HASH_TYPE: "md5",
    RemoteMMARKeys.HASH_VAL: None,
    RemoteMMARKeys.MODEL_FILE: os.path.join("models", "model.pt"),
    RemoteMMARKeys.CONFIG_FILE: os.path.join("config", "config_train.json"),
    RemoteMMARKeys.VERSION: 1,
}

def save_checkpoint(state, name):
    file_path = "checkpoints/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    epoch = state["epoch"]
    save_dir = file_path + name + str(epoch)
    torch.save(state, save_dir)
    print(f"Saving checkpoint for epoch {epoch} in: {save_dir}")

def save_state_dict(state, name):
    file_path = "checkpoints/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    save_dir = file_path + f"{name}_best"
    torch.save(state, save_dir)
    print(f"Best accuracy so far. Saving model to:{save_dir}")

# %%
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

tempdir = None

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

# %% [markdown]
# ### CALLBACKS

# %%
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
    monitor="dice_metric_injure", mode="max", dirpath=checkpoint_dirpath, filename=checkpoint_filename,
)
l_callbacks.append(cbModelCheckpoint)


checkpoint_dirpath = SAVE_PATH + "checkpoints/"
checkpoint_filename = SAVE_PATH[:-1] + "_" + RUN_NAME + "_Last"
lnp.lnp("checkpoint_dirpath: " + checkpoint_dirpath)
lnp.lnp("checkpoint_filename: " + checkpoint_filename)
cbModelCheckpointLast = pytorch_lightning.callbacks.ModelCheckpoint(
   every_n_epochs = 1, dirpath=checkpoint_dirpath, filename=checkpoint_filename,
)
l_callbacks.append(cbModelCheckpointLast)

l_callbacks.append(PrintTableMetricsCallback())
from pytorch_lightning.callbacks import LearningRateMonitor
lr_monitor = LearningRateMonitor(logging_interval='epoch')
l_callbacks.append(lr_monitor)

# %% [markdown]
# ### Training

# %%
# initialise the LightningModule
lnp.lnp(" Start Trainining process...")
net = Net(train_img_size=IMG_SIZE, val_img_size=VAL_SIZE, n_output=2)#.load_from_checkpoint("lightning_logs/checkpoints/lightning_logs_Predict_Segmentation_222_Last.ckpt", train_img_size=IMG_SIZE, val_img_size=VAL_SIZE, n_output=3)
wandb_logger.watch(net)

# set up loggers and checkpoints
log_dir = os.path.join(root_dir, "logs")

from pytorch_lightning.profiler import AdvancedProfiler

profiler = AdvancedProfiler(dirpath="U:\\lauraalvarez\\traumaAI\\Liver_Segmentation\\profiler_logs", filename="output.txt")

# initialise Lightning's trainer.
trainer = pytorch_lightning.Trainer(
    default_root_dir="lightning_logs/checkpoints",
    gpus=[0],
    max_epochs=1,
    # fast_dev_run=True,
    auto_lr_find=False,
    logger=wandb_logger,
    enable_checkpointing=True,
    num_sanity_val_steps=0,
    log_every_n_steps=1,
    callbacks=l_callbacks,
    profiler=profiler
)

# train
result_pred2 = trainer.fit(net)
wandb.alert(
    title="Train finished",
    text="The train has finished"
)
wandb.finish()
shutil.rmtree(tempdir)


# %%
# predictions = trainer.predict(net)

# %%
# post_pred = Compose([EnsureType(), AsDiscrete(argmax=True),KeepLargestConnectedComponent([1,2], is_onehot=False, independent=False)])


# %%
# output_one = [post_pred(i) for i in decollate_batch(predictions[1]["prediction"]["output"])]

# %%
# def _save_gif(volume, filename):
#     volume = volume.astype(np.float64) / np.max(volume) # normalize the data to 0 - 1
#     volume = 255 * volume # Now scale by 255
#     volume = volume.astype(np.uint8)
#     path_to_gif = os.path.join("gifs", f'{filename}.gif')
#     if not os.path.exists("gifs"):
#         print("Creating gifs directory")
#         os.mkdir("gifs")
#     imageio.mimsave(path_to_gif, volume)
#     return path_to_gif

# %%
# _save_gif(output_one[0].permute(3,1,2,0).cpu().numpy(), "test-after")
# _save_gif(predictions[1]["prediction"]["output"][0].permute(3,1,2,0).cpu().numpy(), "test-before")

# %% [markdown]
# ### Tests Gifs

# %%
# prediction = predictions[1]["prediction"]
# selected = {"output":  output_one[0], "image": prediction["image"][0], "label": prediction["label"][0]}
# selected = Resized(keys=["image", "label"], spatial_size=(160, 160, 160))(selected)
# selected = Resized(keys=["output"], spatial_size=(160, 160, 160))(selected)
# selected = {"output": selected["output"].unsqueeze(0), "image": selected["image"].unsqueeze(0), "label": selected["label"].unsqueeze(0)}
# # print('true label:', selected['label'].shape)
# pred = selected["output"][0].detach().cpu().numpy()
# true_label = selected['label'][0].detach().cpu().numpy()
# image = selected['image'][0].cpu().numpy()
# # print('true label:', true_label.shape)

# blended_true_label = blend_images(image, true_label, alpha=0.3)
# blended_final_true_label = torch.from_numpy(blended_true_label).permute(1,2,0,3)

# blended_prediction = blend_images(image, pred, alpha=0.3)
# blended_final_prediction = torch.from_numpy(blended_prediction).permute(1,2,0,3)

# volume_pred = blended_final_prediction[:,:,:,:]
# volume_label = blended_final_true_label[:,:,:,:]
# volume_pred = np.squeeze(volume_pred).permute(3,0,1,2).cpu()
# volume_label = np.squeeze(volume_label).permute(3,0,1,2).cpu()
# volume_img = torch.tensor(image).permute(3,1,2,0).repeat(1,1,1,3).cpu()

# volume = torch.hstack((volume_img, volume_pred, volume_label))

# volume_path = _save_gif(volume.numpy(), f"blended-test-1")

# %%
# prediction = predictions[1]["prediction"]

# selected = {"output": prediction["output"][0], "image": prediction["image"][0], "label": prediction["label"][0]}
# selected = Resized(keys=["image", "label"], spatial_size=(160, 160, 160))(selected)
# selected = Resized(keys=["output"], spatial_size=(160, 160, 160))(selected)
# selected = {"output": torch.argmax(selected["output"], 0).unsqueeze(0), "image": selected["image"].unsqueeze(0), "label": selected["label"].unsqueeze(0)}

# pred = selected["output"].detach().cpu().numpy()
# true_label = selected['label'][0].detach().cpu().numpy()
# image = selected['image'][0].cpu().numpy()

# blended_true_label = blend_images(image, true_label, alpha=0.3)
# blended_final_true_label = torch.from_numpy(blended_true_label).permute(1,2,0,3)

# blended_prediction = blend_images(image, pred, alpha=0.3)
# blended_final_prediction = torch.from_numpy(blended_prediction).permute(1,2,0,3)

# volume_pred = blended_final_prediction[:,:,:,:]
# volume_label = blended_final_true_label[:,:,:,:]
# volume_pred = np.squeeze(volume_pred).permute(3,0,1,2).cpu()
# volume_label = np.squeeze(volume_label).permute(3,0,1,2).cpu()
# volume_img = torch.tensor(image).permute(3,1,2,0).repeat(1,1,1,3).cpu()

# volume = torch.hstack((volume_img, volume_pred, volume_label))

# volume_path = _save_gif(volume.numpy(), f"blended-test-1-org")



