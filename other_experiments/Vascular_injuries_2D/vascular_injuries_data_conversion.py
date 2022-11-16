# %% [markdown]
# ### Imports

# %%
import os
from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ToTensord,
    SaveImaged,
    Spacingd,
    EnsureTyped,
    AsChannelLastd,
    AsChannelFirstd,
    AsDiscreted,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    FillHolesd,
    RandCropByLabelClassesd,
    Resized, RandFlipd, RandRotate90d,
)
from monai.transforms.transform import MapTransform
from monai.transforms.inverse import InvertibleTransform
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
from monai.transforms.intensity.array import (
    ScaleIntensityRangePercentiles,
)
import matplotlib.pyplot as plt
from ipywidgets.widgets import * 
import ipywidgets as widgets
import matplotlib.pyplot as plt
import glob 
import torch
import os
# import cv2

# %% [markdown]
# ### Transformations

# %%
class RemoveDicts(MapTransform, InvertibleTransform):

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
        a = {"image": d["image"], "label": d["label"], "path": d["image_meta_dict"]["filename_or_obj"]}
        # print(a["path"])
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
class ScaleIntensityRangePercentilesd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ScaleIntensityRangePercentiles`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        lower: lower percentile.
        upper: upper percentile.
        b_min: intensity target range min.
        b_max: intensity target range max.
        clip: whether to perform clip after scaling.
        relative: whether to scale to the corresponding percentiles of [b_min, b_max]
        channel_wise: if True, compute intensity percentile and normalize every channel separately.
            default to False.
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = ScaleIntensityRangePercentiles.backend

    def __init__(
        self,
        keys: KeysCollection,
        lower: float,
        upper: float,
        b_min: Optional[float],
        b_max: Optional[float],
        clip: bool = False,
        relative: bool = False,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scaler = ScaleIntensityRangePercentiles(lower, upper, b_min, b_max, clip, relative, channel_wise, dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key])
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
        print(median, mean, std, mn, mx, percentile_99_5, percentile_00_5)
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

# %%
class ClosePreprocessing(MapTransform):
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

    def __init__(
        self,
        keys: KeysCollection,
        kernel_size: int = 10,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.kernel = np.ones((kernel_size,kernel_size),np.uint8)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        closed_slices = list()
        for slice in range(d["label"].shape[-1]):
            result = cv2.morphologyEx(d["label"][0, :, :, slice], cv2.MORPH_CLOSE, self.kernel)
            closed_slices.append(result)

        d["label"] = torch.Tensor(np.stack(closed_slices)).permute(1, 2, 0).unsqueeze(0)
        return d

# %%
from PIL import Image
class WriteToPNG(MapTransform):
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

    def __init__(
        self,
        keys: KeysCollection,
        output_dir: str,
        mode:str,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.output_dir = output_dir
        self.mode = mode

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        d["image"] = d["image"].detach().cpu().numpy()
        d["label"] = d["label"].detach().cpu().numpy()
        for slice in range(d["image"].shape[1]):
            filename = os.path.basename(d["image_meta_dict"]["filename_or_obj"]).split(".")[0] + f"_{slice}.png"
            if 5 in d["label"][0,slice,:,:]:

                if self.mode == "train":
                    save_dir_img = os.path.join(self.output_dir, 'imagesTr', filename)
                    save_dir_label = os.path.join(self.output_dir, 'labelsTr', filename)
                else:
                    save_dir_img = os.path.join(self.output_dir, 'imagesTs', filename)
                    save_dir_label = os.path.join(self.output_dir, 'labelsTs', filename)

                if not os.path.exists(os.path.dirname(save_dir_img)):
                    print(f"Creating directory: {os.path.dirname(save_dir_img)}")
                    os.makedirs(os.path.dirname(save_dir_img))
                
                if not os.path.exists(os.path.dirname(save_dir_label)):
                    print(f"Creating directory: {os.path.dirname(save_dir_label)}")
                    os.makedirs(os.path.dirname(save_dir_label))

                print(f"Saving to {save_dir_img}")
                plt.imsave(save_dir_img, d["image"][0, slice, :, :], cmap="gray")
                print(f"Saving to {save_dir_label}")
                plt.imsave(save_dir_img, d["label"][0, slice, :, :], cmap="gray")
                
        return d

train_images = sorted( glob.glob( os.path.join( "U://","lauraalvarez","data", "vascular_injuries", "nnunet", "imagesTr", "*.nii.gz") ) )
train_labels = sorted( glob.glob( os.path.join( "U://","lauraalvarez","data", "vascular_injuries", "nnunet", "labelsTr", "*.nii.gz") ) )
data_dicts = [ {"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels) ]
test_images = sorted( glob.glob( os.path.join( "U://","lauraalvarez","data", "vascular_injuries", "nnunet", "imagesTs", "*.nii.gz" ) ) )
test_labels = sorted( glob.glob( os.path.join( "U://","lauraalvarez","data", "vascular_injuries", "nnunet", "labelsTs", "*.nii.gz") ) )
data_dicts_test = [{"image": image_name, "label": label_name} for image_name, label_name in zip(test_images, test_labels) ]

transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        RemoveDicts(keys=["image", "label"]),
        AsChannelFirstd(keys=["label"]),
        AddChanneld(keys=["image", "label"]),
        # AsDiscreted(keys=["label"], argmax=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # Spacingd( keys=["image", "label"], pixdim=(1.5, 1.5, 1), mode=("bilinear", "nearest"),),
        CropForegroundd(keys=["image", "label"], source_key="label"),
        NNUnetScaleIntensity(keys=["image"]),
        # ClosePreprocessing(keys=["label"]),
        WriteToPNG(keys=["image", "label"], output_dir="U:\\lauraalvarez\data\vascular_injuries\png_2", mode="train"),
        ToTensord(keys=["image", "label"]),
    ]
)

error_cases = list()
for data_dict in data_dicts[1:]:
   data_dict = transforms(data_dict)
    # except Exception as e:
    #         error_cases.append(data_dict)
print(f"{len(error_cases)} error cases")
print(error_cases)


