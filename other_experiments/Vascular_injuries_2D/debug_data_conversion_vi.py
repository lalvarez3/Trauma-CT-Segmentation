# %% [markdown]
# ### Imports

# %%
from enum import unique
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
    CropForeground,
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
class CustomCropForegroundd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        source_key: str,
        allow_missing_keys: bool = False,
        **np_kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.cropper = CropForeground(margin=0, **np_kwargs)
        self.transform = CropForegroundd(keys=["image", "label"], source_key="label", margin=20)
        # CropForegroundd(keys=["image", "label"], source_key="label"),

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        original_label = d['label']
        img_name = os.path.basename(d["image_meta_dict"]["filename_or_obj"])
        unique_labels = np.unique(original_label) 
        print(unique_labels)

        if len(unique_labels) == 1:
            raise ValueError(f"Only one label found in the image. Please check the image {img_name}.")

        # Check if there is one VI or two
        elif np.sum(unique_labels > 4) == 1:
            idx = np.select(unique_labels > 4, unique_labels)
            margin = 1
            if idx == 5: 
                d["label"] = np.where((original_label%2 == 0), 0, original_label)
            else: 
                d["label"] = np.where((original_label%2 != 0), 0, original_label)
            # label_organ = np.where((original_label != idx2), 0, original_label)
            # label_vi = np.where((original_label != idx), 0, original_label)
            # labels = [label_organ, label_vi]
            # chans = np.stack(labels, axis=0)
            # d["label"] = np.max(chans, axis=0)
            d = CropForegroundd(keys=["image", "label"], source_key="label", margin=margin)(d)
            cropped_label =  d["label"] # crop ROI
            vi_label = np.where(cropped_label != idx, 0, cropped_label)
            box_start, box_end = self.cropper.compute_bounding_box(img=vi_label)
            new_label = vi_label[:, box_start[0]:box_end[0], :, :]
            d["image"] = d["image"][:, box_start[0]:box_end[0], :, :]
            d["label"] = new_label


        else:
            spleen_labels = np.where((original_label%2 != 0), 0, original_label)
            vi_spleen_labels = np.where((original_label != 6), 0, original_label)
            liver_labels = np.where((original_label%2 == 0), 0, original_label)
            vi_liver_labels = np.where((original_label != 5), 0, original_label)
            new_d = {'image': d['image'], 'label_spleen': spleen_labels, 'label_liver': liver_labels, 'label_vi_spleen': vi_spleen_labels, 'label_vi_liver': vi_liver_labels}
            spleen_d = CropForegroundd(keys=['image', 'label_spleen', 'label_vi_spleen'], source_key='label_spleen')(new_d)
            spleen_d.pop('label_liver') # crop ROI
            liver_d = CropForegroundd(keys=['image', 'label_liver', 'label_vi_liver'], source_key='label_liver')(new_d)
            liver_d.pop('label_spleen') # crop ROI
            box_start_spleen, box_end_spleen = CropForeground().compute_bounding_box(img=spleen_d['label_vi_spleen'])
            box_start_liver, box_end_liver = CropForeground().compute_bounding_box(img=liver_d['label_vi_liver'])
            new_label_spleen = spleen_d["label_vi_spleen"][:, box_start_spleen[0]:box_end_spleen[0], :, :]
            new_label_liver = liver_d["label_vi_liver"][:, box_start_liver[0]:box_end_liver[0], :, :]
            new_image_spleen = spleen_d["image"][:, box_start_spleen[0]:box_end_spleen[0], :, :]
            new_image_liver = liver_d["image"][:, box_start_liver[0]:box_end_liver[0], :, :]
            d["image"] = [new_image_spleen, new_image_liver]
            d["label"] = [new_label_spleen, new_label_liver]
            

        return d

# %%
from PIL import Image
class WriteToPNG(MapTransform):

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
    
    def save_dict(self, d, file):
        keys = ['label', 'image']
        for key in keys:
            if isinstance(d[key], torch.Tensor):
                d[key] = d[key].detach().cpu().numpy()
            for slice in range(d[key].shape[1]):
                filename = file + f"_{slice}.png"
                if key == "image":
                    if self.mode == "train":
                        save_dir = os.path.join(self.output_dir, 'imagesTr', filename)
                    else:
                        save_dir = os.path.join(self.output_dir, 'imagesTs', filename)
                else:
                    if self.mode == "train":
                        save_dir = os.path.join(self.output_dir, 'labelsTr', filename)
                    else:
                        save_dir = os.path.join(self.output_dir, 'labelsTs', filename)
                if not os.path.exists(os.path.dirname(save_dir)):
                    print(f"Creating directory: {os.path.dirname(save_dir)}")
                    os.makedirs(os.path.dirname(save_dir))
                unique_labels, counts = np.unique(d["label"][0, slice, :, :], return_counts=True)
                if np.sum(unique_labels > 4) > 0:
                    print(f"Saving to {save_dir}")
                    to_save = np.transpose(d[key], (0, 1, 3, 2))
                    plt.imsave(save_dir, np.flip(to_save,2)[0, slice, :, :], cmap="gray")



    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        if len(d["image"]) == 2:
            d_spleen = {'image': d['image'][0], 'label': d['label'][0]}
            d_liver = {'image': d['image'][1], 'label': d['label'][1]}
            file = os.path.basename(d["image_meta_dict"]["filename_or_obj"]).split(".")[0]
            file_splen = file + "_spleen"
            file_liver = file + "_liver"
            self.save_dict(d_spleen, file_splen)
            self.save_dict(d_liver, file_liver)

        else:
            file = os.path.basename(d["image_meta_dict"]["filename_or_obj"]).split(".")[0]
            self.save_dict(d, file)
            

        return d

# %% [markdown]
# ### Dataset

# %%
train_images = sorted( glob.glob( os.path.join( "U:/lauraalvarez/","data", "vascular_injuries", "nii", "imagesTr", "*.nii.gz") ) )
train_labels = sorted( glob.glob( os.path.join( "U:/lauraalvarez/","data", "vascular_injuries", "nii", "labelsTr", "*.nii.gz") ) )
data_dicts = [ {"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels) ]
test_images = sorted( glob.glob( os.path.join( "U:/lauraalvarez/","data", "vascular_injuries", "nii", "imagesTs", "*.nii.gz" ) ) )
test_labels = sorted( glob.glob( os.path.join( "U:/lauraalvarez/","data", "vascular_injuries", "nii", "labelsTs", "*.nii.gz") ) )
data_dicts_test = [{"image": image_name, "label": label_name} for image_name, label_name in zip(test_images, test_labels) ]

# %%
# transforms_bsl = Compose([ LoadImaged(keys=["image", "label"]), AddChanneld(keys=["image"]), ToTensord(keys=["image", "label"]),])
transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        # RemoveDicts(keys=["image", "label"]),
        AsChannelFirstd(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # AsDiscreted(keys=["label"], argmax=True),
        # Spacingd( keys=["image", "label"], pixdim=(1.5, 1.5, 1), mode=("bilinear", "nearest"),),
        # CropForegroundd(keys=["image", "label"], source_key="label"),
        NNUnetScaleIntensity(keys=["image"]),
        CustomCropForegroundd(keys=["label"], source_key="label"),
        # ClosePreprocessing(keys=["label"]),
        WriteToPNG(keys=["image", "label"], output_dir="U:\\lauraalvarez\\data\\vascular_injuries\\png\\", mode="train"),
        # ToTensord(keys=["image", "label"]),
    ]
)

# injure_org = transforms_bsl(data_dicts)
error_cases = list()
for data_dict in data_dicts_test[1:]:
        data_dict = transforms(data_dict)
print(f"{len(error_cases)} error cases")
print(error_cases)
# injure_crop = transforms(data_dicts)
# print(injure_crop["image"].shape, injure_crop["label"].shape)

# %%
blended_true_label = blend_images(injure_crop["image"], injure_crop["label"], alpha=0.9)
blended_final_true_label_closed = blended_true_label.permute(1,2,0,3)
print(blended_final_true_label_closed.shape)

# %%
from monai.visualize import matshow3d, blend_images
import torch 

def dicom_animation(slice):
    plt.figure(figsize=(18, 6))
    plt.title(f"liver no injured ")
    plt.imshow(blended_final_true_label_closed[:, :, :, slice], cmap="bone")
    plt.show()

interact(dicom_animation, slice=(0, blended_final_true_label_closed.shape[-1]-1))

# %%
from monai.visualize import matshow3d, blend_images
import torch 

def dicom_animation(slice):
    plt.figure(figsize=(18, 6))
    plt.title(f"liver no injured ")
    plt.imshow(blended_final_true_label[:, :, :, slice], cmap="bone")
    plt.show()

interact(dicom_animation, slice=(0, blended_final_true_label.shape[-1]-1))


