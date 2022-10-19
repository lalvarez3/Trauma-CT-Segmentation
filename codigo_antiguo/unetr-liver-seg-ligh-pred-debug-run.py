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
    SpatialPadd,
    RandGaussianNoised,
    RandScaleIntensityd,
    
)

# from monailabel.scribbles.transforms import (
#     MakeISegUnaryd,
#     ApplyCRFOptimisationd,
# )
from monai.networks.blocks import CRF
import wandb
from monai.networks.nets import UNet, UNETR
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceFocalLoss, GeneralizedDiceLoss
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, Dataset, LMDBDataset
from monai.config import print_config
from monai.apps import download_and_extract
import torch

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

from monai.transforms.intensity.array import (
    ScaleIntensityRange,
)


# %% [markdown]
# #### Preprocessing

# %%
class InterpolateMode(Enum):
    NEAREST = "nearest"
    LINEAR = "linear"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    TRILINEAR = "trilinear"
    AREA = "area"


InterpolateModeSequence = Union[
    Sequence[Union[InterpolateMode, str]], InterpolateMode, str
]

class ResizedC(MapTransform, InvertibleTransform):

    backend = Resize.backend

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Union[Sequence[int], int],
        size_mode: str = "all",
        mode: InterpolateModeSequence = InterpolateMode.AREA,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.resizer = Resize(spatial_size=spatial_size, size_mode=size_mode)
        self.spatial_size = spatial_size

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, mode, align_corners in self.key_iterator(
            d, self.mode, self.align_corners
        ):
            self.push_transform(
                d,
                key,
                extra_info={
                    "mode": mode.value if isinstance(mode, Enum) else mode,
                    "align_corners": align_corners
                    if align_corners is not None
                    else TraceKeys.NONE,
                },
            )
            init_slice = int(d[key].shape[-1]*0.15)
            end_slice = int(d[key].shape[-1]*0.1)
            # Reduce Size in Memory
            if key == "label":
                d[key] = d[key].astype(np.int8)
                if d[key].shape[-1] > 600: d[key] = d[key][:,:,:,init_slice:-end_slice] #

                if d["image_meta_dict"].get("PatientName", None) and d["image_meta_dict"]["PatientName"].startswith("NI") and len(d[key].shape) != 4:
                    # print(d[key].shape)
                    liver_channel = np.where((d[key] != 6), 0, d[key])
                    liver_channel = np.where((liver_channel == 6), 1, liver_channel)
                    # liver_channel = np.expand_dims(liver_channel, 0)
                    w, h, z = self.spatial_size
                    liver_channel = self.resizer(liver_channel, align_corners=align_corners)
                    background = np.ones((1, z, w, h), dtype=np.float16) - liver_channel
                    empty_injures = np.zeros((1, z, w, h), dtype=np.float16)
                    resized = [background, liver_channel, empty_injures]
                    d[key] = np.stack(resized).astype(np.int8).squeeze()

                else:
                    label = d[key]
                    w, h, z = self.spatial_size
                    resized = list()
                    background = np.ones((1, w, h, z), dtype=np.int8)
                    for i, channel in enumerate([0, 2]):  # TODO: desharcodead
                        resized.append(
                            self.resizer(
                                np.expand_dims(label[channel, :, :, :], 0),
                                align_corners=align_corners,
                            )
                        )

                    background -= resized[0] # + resized[1]
                    resized = [background] + resized
                    d[key] = np.stack(resized).astype(np.int8).squeeze()
            else:
                if d[key].shape[-1] > 600: d[key] = d[key][:,:,:,init_slice:-end_slice]
                d[key] = self.resizer(d[key], align_corners=align_corners)

        keys = ['spacing', 'original_affine', 'affine', 'spatial_shape', 'original_channel_dim', 'filename_or_obj']
        new_label_metadata = dict()
        for key in keys:
            new_label_metadata[key] = d["label_meta_dict"].get(key, 0)

        d["label_meta_dict"] = new_label_metadata

        if "PatientID" not in d["image_meta_dict"]:
            d["image_meta_dict"]["PatientID"] = "0"
        if "PatientName" not in d["image_meta_dict"]:
            d["image_meta_dict"]["PatientName"] = "0"
        if "SliceThickness" not in d["image_meta_dict"]:
            d["image_meta_dict"]["SliceThickness"] = "0"
        return d

    def inverse(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            orig_size = transform[TraceKeys.ORIG_SIZE]
            mode = transform[TraceKeys.EXTRA_INFO]["mode"]
            align_corners = transform[TraceKeys.EXTRA_INFO]["align_corners"]
            # Create inverse transform
            inverse_transform = Resize(
                spatial_size=orig_size,
                mode=mode,
                align_corners=None
                if align_corners == TraceKeys.NONE
                else align_corners,
            )
            # Apply inverse transform
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d

# %%
class adaptOverlay(MapTransform, InvertibleTransform):

    backend = Resize.backend

    def __init__(
        self,
        keys: KeysCollection,
        size_mode: str = "all",
        mode: InterpolateModeSequence = InterpolateMode.AREA,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))

    def __adapt_overlay__(self, overlay_path, mha_path, label):
        import SimpleITK as sitk
        if label.shape[-1] == 6:
            return label
        # Load the mha
        mha_data = sitk.ReadImage(mha_path)
        mha_org = mha_data.GetOrigin()[-1]
        # Load the mha image
        mha_img = sitk.GetArrayFromImage(mha_data)
        original_z_size = mha_img.shape[0]

        # Load the overlay
        overlay_data = sitk.ReadImage(overlay_path)
        overlay_org = overlay_data.GetOrigin()[-1]

        overlay_init = np.abs(1/mha_data.GetSpacing()[-1]*(mha_org-overlay_org) )

        lower_bound = int(overlay_init)
        upper_bound = label.shape[-1]
        zeros_up = lower_bound
        zeros_down = original_z_size - (upper_bound + lower_bound)
        new = list()

        if zeros_up > 0:
            new.append(np.zeros((label.shape[0], label.shape[1], zeros_up), dtype=label.dtype))

        new.append(label)

        if zeros_down > 0:
            new.append(np.zeros((label.shape[0], label.shape[1], zeros_down), dtype=label.dtype))

        label = np.concatenate(new, axis=2)

        return label


    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, mode, align_corners in self.key_iterator(
            d, self.mode, self.align_corners
        ):
            self.push_transform(
                d,
                key,
                extra_info={
                    "mode": mode.value if isinstance(mode, Enum) else mode,
                    "align_corners": align_corners
                    if align_corners is not None
                    else TraceKeys.NONE,
                },
            )
            # Reduce Size in Memory
            if key == "label":
                d[key] = d[key].astype(np.int8)
                if d["image_meta_dict"].get("PatientName", None) and d["image_meta_dict"]["PatientName"].startswith("NI"):
                    file_path = d["label_meta_dict"]["filename_or_obj"]
                    data_path = d["image_meta_dict"]["filename_or_obj"]
                    d[key] = self.__adapt_overlay__(file_path, data_path, d[key])
        return d

    def inverse(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            orig_size = transform[TraceKeys.ORIG_SIZE]
            mode = transform[TraceKeys.EXTRA_INFO]["mode"]
            align_corners = transform[TraceKeys.EXTRA_INFO]["align_corners"]
            # Create inverse transform
            inverse_transform = Resize(
                spatial_size=orig_size,
                mode=mode,
                align_corners=None
                if align_corners == TraceKeys.NONE
                else align_corners,
            )
            # Apply inverse transform
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d

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
        print(d["label"].shape)
        for key in self.key_iterator(d):
            self.push_transform(d, key)
            d[key] = np.where((d[key] != self.class_to_keep), 0, d[key])
            d[key] = np.where((d[key] == self.class_to_keep), 1, d[key])
            d[key] = d[key][d[key] == self.class_to_keep]
        return d

    def inverse(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            d[key] = d[key]
            # Remove the applied transform
            self.pop_transform(d, key)
        return d

# %%
class RemoveDicts(MapTransform, InvertibleTransform):

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        verbose:bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.verbose = verbose

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            self.push_transform(d, key)
        # print(d["image_meta_dict"]["filename_or_obj"])
        a = {"image": d["image"], "label": d["label"], "path": d["image_meta_dict"]["filename_or_obj"]}
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


# %%
class PrintInfo(MapTransform, InvertibleTransform):

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            self.push_transform(d, key)
        # print(d["image_meta_dict"]["filename_or_obj"])
        # a = {"image": d["image"], "label": d["label"], "path": d["image_meta_dict"]["filename_or_obj"]}
        print(d["path"])
        # d = a
        return d

    def inverse(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            d[key] = d[key]
            # Remove the applied transform
            self.pop_transform(d, key)
        return d

# %%
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
        mask = np.greater(mask, 0) # get only non-zero positive pixels/labels
        volume = volume * mask
        volume = np.ma.masked_equal(volume,0).compressed()
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

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            stats = self._compute_stats(d[key], d['label'])
            d[key] = np.clip(d[key], stats[6], stats[5])
            d[key] = (d[key] - stats[1]) / stats[2]
        return d

# %% [markdown]
# #### Flags

# %%

SEED = 0
PRETRAINED = False
TRANSFER_LEARNING = False
N_WORKERS_LOADER = 28
N_WORKERS_CACHE = 0
CACHE_RATE = 0
SEED = 7
BS = 1
MAX_EPOCHS = 500
PATCH_SIZE = (128,128,128)
# HOME = "U:\\" #"/mnt/chansey" # #"/mnt/netcache/diag" 
HOME =  "/mnt/chansey" # #"/mnt/netcache/diag" 
IMG_SIZE = (128,128,128)
VAL_SIZE = (256,256,256)
SAVE_PATH = os.path.join(HOME, "lauraalvarez", "traumaAI", "Liver_Segmentation", "lightning_logs")
run_idx = len(os.listdir(os.path.join(HOME, "lauraalvarez", "traumaAI", "Liver_Segmentation", "wandb")))
RUN_NAME = f"Predict_Segmentation_UNETR_{run_idx+1}"
pytorch_lightning.seed_everything(SEED)

# %% [markdown]
# #### Define the LightningModule

# %%
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = os.path.join(HOME,"lauraalvarez","traumaAI", "Liver_Segmentation", "spleen_data")
print(root_dir)

# %%
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
            dropout_rate=0.00,
        )


        # self.loss_function = DiceFocalLoss(to_onehot_y=True, softmax=True, jaccard=True)
        self.loss_function = DiceCELoss(softmax=True, to_onehot_y=True, jaccard =False, ce_weight=torch.FloatTensor([0.3, 0.5, 0.7]))
        self.post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=3)])
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=3)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        self.train_dice_metric = DiceMetric(
            include_background=False, reduction="mean_batch"
        )
        self.best_val_dice = 0
        self.best_val_epoch = 0

        # dist_mat = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.5], [1.0, 0.5, 0.0]], dtype=np.float32)
        # self.loss_function = GeneralizedWassersteinDiceLoss(dist_matrix=dist_mat)

        self.save_hyperparameters()  # save hyperparameters

        # self.logger.expe.init(self.hparams)

    # def forward(self, x):
    #     out = self._model(x)
    #     out = self.crf(out, x)
    #     return out

    def prepare_data(self):
        # set up the correct data path
        train_images = sorted(
            glob.glob(
                os.path.join(
                    HOME,
                    "lauraalvarez",
                    "nnunet",
                    "nnUNet_raw_data",
                    "Task510_LiverTraumaDGX",
                    "imagesTr",
                    "*.nii.gz",
                )
            )
        )
        train_labels = [img.replace("imagesTr", "labelsTr") for img in train_images]
        train_labels = [img.replace("_0000", "") for img in train_labels]


        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(train_images, train_labels)
        ]

        test_images = sorted(
            glob.glob(
                os.path.join(
                    HOME,
                    "lauraalvarez",
                    "nnunet",
                    "nnUNet_raw_data",
                    "Task510_LiverTraumaDGX",
                    "imagesTs",
                    "*.nii.gz",
                )
            )
        )
        test_labels = [img.replace("imagesTs", "labelsTs") for img in test_images]
        train_labels = [img.replace("_0000", "") for img in test_labels]


        data_dicts_test = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(test_images, test_labels)
        ]

        random.shuffle(data_dicts)
        validation_lim = int(len(data_dicts) * 0.9)
        train_files, val_files, test_files = data_dicts[:validation_lim], data_dicts[validation_lim:], data_dicts_test
        print("validation files", val_files)
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
                LoadImaged(keys=["image", "label"]),
                RemoveDicts(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                # NNUnetScaleIntensity(keys=["image"]),
                ToTensord(keys=["image", "label"]),
                RandCropByLabelClassesd(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=PATCH_SIZE,
                    ratios=ratios,
                    num_classes=self.n_output,
                    num_samples=2,
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
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                ToTensord(keys=["image", "label"]),
            ]
        )

        # self.train_ds =  LMDBDataset( #LMDBDataset
        #     data=train_files,
        #     transform=train_transforms,
        #     cache_dir=os.path.join(
        #         HOME,
        #         "lauraalvarez",
        #         "data",
        #         "lmbd_v3",
        #         "LiverTr"
        #     ),
        # )

        self.train_ds =  CacheDataset( #LMDBDataset
            data=train_files,
            transform=train_transforms,
            cache_rate=0.1,
        )

        # self.val_ds = LMDBDataset( #Dataset(
        #     data=val_files,
        #     transform=val_transforms,
        #     cache_dir=os.path.join(
        #         HOME,
        #         "lauraalvarez",
        #         "data",
        #         "lmbd_v3",
        #         "LiverVs"
        #     ),
        # )

        self.val_ds = Dataset( #Dataset(
            data=val_files,
            transform=val_transforms,
        )


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self._model.parameters(), 1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', verbose=True, 
            factor=0.05, patience=60, min_lr=10e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
            "interval": "epoch",
        }

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=BS,
            shuffle=True,
            num_workers=N_WORKERS_LOADER,
            collate_fn=list_data_collate,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=N_WORKERS_LOADER,
            # pin_memory=True,
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
        tensorboard_logs = {"batch_train_loss": loss.item()}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        dice_liver, dice_injure = self.train_dice_metric.aggregate()
        avg_loss = torch.stack([x["loss"] for x in outputs if x != 0]).mean()
        self.log("train_loss", avg_loss,  prog_bar=True)
        self.log("train_dice_liver", dice_liver, prog_bar=True)
        self.log("train_dice_injury", dice_injure, prog_bar=True)
        self.train_dice_metric.reset()
      
    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        filenames = batch["path"]
        post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=3)])
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=3)])

        roi_size = PATCH_SIZE
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images, roi_size, 
            sw_batch_size, self.forward)

        with torch.cuda.amp.autocast():
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
            "val_dice_metric": self.dice_metric,
            "val_number": len(outputs),
            "prediction": predicition,
            "val_loss": loss,
        }

    def validation_epoch_end(self, outputs):

        mean_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()   
        post_pred_dice = Compose([
                EnsureType(),
                AsDiscrete(argmax=True, to_onehot=3),
                KeepLargestConnectedComponent(
                    [1, 2], is_onehot=True, independent=True
                ),
            ]
        )
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=3)])
        dice_liver, dice_injury = self.dice_metric.aggregate()
        self.dice_metric.reset()
        tensorboard_logs = { "val_dice_metric": dice_injury,}
        predictions = [x["prediction"] for x in outputs]

        if (
            self.current_epoch % 25 == 0
            or dice_injury - self.best_val_dice > 0.1
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

        if dice_injury > self.best_val_dice:
            self.best_val_dice = dice_injury.item()
            self.best_val_epoch = self.current_epoch
        lnp.lnp(
            f"current epoch: {self.current_epoch}, "
            f"current liver dice: {dice_liver:.4f}, "
            f"current injure  dice: {dice_injury:.4f} "
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.log("val_dice_metric_liver", dice_liver.item(), prog_bar=True)
        self.log("val_dice_metric_injury", dice_injury.item(), prog_bar=True)
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
    post_pred_blending = Compose([EnsureType(), AsDiscrete(argmax=True),KeepLargestConnectedComponent([1,2], is_onehot=False, independent=True)])
    prediction["output"] = [post_pred_blending(i) for i in decollate_batch(prediction["output"])]
    selected = {"output": prediction["output"][0], "image": prediction["image"][0], "label": prediction["label"][0]}

    selected = Resized(keys=["image", "label"], spatial_size=(160, 160, 160))(selected)
    selected = Resized(keys=["output"], spatial_size=(160, 160, 160))(selected)

    selected = {"output": selected["output"], "image": selected["image"].unsqueeze(0), "label": selected["label"].unsqueeze(0)}


    pred = selected['output'].detach().cpu().numpy()
    true_label = selected['label'][0].detach().cpu().numpy()
    image = selected['image'][0].cpu().numpy()
    
    blended_true_label = blend_images(image, true_label, alpha=0.7)
    blended_final_true_label = torch.from_numpy(blended_true_label).permute(1,2,0,3)

    blended_prediction = blend_images(image, pred, alpha=0.7)

    blended_final_prediction = torch.from_numpy(blended_prediction).permute(1,2,0,3)

    volume_pred = blended_final_prediction[:,:,:,:]
    volume_label = blended_final_true_label[:,:,:,:]
    volume_pred = volume_pred.permute(3,0,1,2).cpu()
    volume_label = np.squeeze(volume_label).permute(3,0,1,2).cpu()
    volume_img = torch.tensor(image).permute(3,1,2,0).repeat(1,1,1,3).cpu()

    volume = torch.hstack((volume_img, volume_pred, volume_label))

    volume_path = _save_gif(volume.numpy(), f"blended-{filename}")
       
    
    return volume_path


def get_classification_info(prediction):
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=3)])

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

try:
    wandb.finish()
except:
    print("Wandb not initialized")

lnp = Log_and_print(RUN_NAME)
lnp.lnp("Loggers start")
lnp.lnp("ts_script: " + str(time.time()))

wandb_logger = pytorch_lightning.loggers.WandbLogger(
    project="traumaIA",
    name=RUN_NAME,
    save_dir=SAVE_PATH,
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
    monitor="val_dice_metric_injury", mode="max", dirpath=checkpoint_dirpath, filename=checkpoint_filename, 
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


if '__main__' == __name__:
    lnp.lnp(" Start Trainining process...")
    net = Net(train_img_size=IMG_SIZE, val_img_size=VAL_SIZE, n_output=3)#.load_from_checkpoint("lightning_logs/checkpoints/lightning_logs_Predict_Segmentation_240_Last-v1.ckpt", train_img_size=IMG_SIZE, val_img_size=VAL_SIZE, n_output=3)
    wandb_logger.watch(net)

    # set up loggers and checkpoints
    log_dir = os.path.join(root_dir, "logs")

    # initialise Lightning's trainer.
    trainer = pytorch_lightning.Trainer(
        default_root_dir="lightning_logs/checkpoints",
        gpus=[0],
        max_epochs=MAX_EPOCHS,
        fast_dev_run=False,
        auto_lr_find=False,
        logger=wandb_logger,
        enable_checkpointing=True,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        callbacks=l_callbacks,
        move_metrics_to_cpu=False,
        accumulate_grad_batches=16,
        benchmark=True,

    )

    # train
    result_pred2 = trainer.fit(net)
    wandb.alert(
        title="Train finished", 
        text="The train has finished"
    )
    wandb.finish()


