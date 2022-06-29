import os

from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ToTensord,
    Resized,
    AsChannelLastd,
    AsChannelFirstd,
    AsDiscrete,
    CropForeground,
    SpatialCropd,
    AsDiscreted,
    ScaleIntensityRanged,
    EnsureType,
    KeepLargestConnectedComponent,
    KeepLargestConnectedComponentd
)
import glob
from monai.transforms.transform import MapTransform
from monai.transforms.inverse import InvertibleTransform
from monai.data import decollate_batch
import SimpleITK as sitk

from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np

from monai.visualize import matshow3d, blend_images
import torch
from monai.metrics import DiceMetric

import csv


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
        volume = np.ma.masked_equal(
            volume.copy().astype(np.int16) * np.greater(mask, 0), 0
        ).compressed()
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

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            stats = self._compute_stats(d[key], d["label"])
            d[key] = np.clip(d[key], stats[6], stats[5])
            d[key] = (d[key] - stats[1]) / stats[2]
        return d


import imageio


def save_csv(output_path, task_name, data):
    import csv

    base_path = os.path.join(
        "U:\\",
        "lauraalvarez",
        "nnunet",
        "nnUNet_raw_data",
        task_name,
        "out",
        "gifs",
        output_path)

    keys = data[0].keys()
    a_file = open(base_path, "w+")
    dict_writer = csv.DictWriter(a_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(data)
    a_file.close()


def _save_gif(volume, filename, task_name="Task504_LiverTrauma"):
    volume = volume.astype(np.float64) / np.max(volume)  # normalize the data to 0 - 1
    volume = volume * 255  # Now scale by 255
    volume = volume.astype(np.uint8)
    base_path = os.path.join(
        "U:\\",
        "lauraalvarez",
        "nnunet",
        "nnUNet_raw_data",
        task_name,
        "out",
        "gifs")
    path_to_gif = os.path.join(base_path, f"{filename}.mp4")
    if not os.path.exists(base_path):
        print("Creating gifs directory")
        os.mkdir(base_path)
    imageio.mimsave(path_to_gif, volume, fps=5)
    return path_to_gif


def create_figs(image_path, prediction_path, label_path, task_name="Task506_VITrauma"):
    images = glob.glob(
        os.path.join(
            "U:\\",
            "lauraalvarez",
            "nnunet",
            "nnUNet_raw_data",
            task_name,
            "imagesTs",
            "*.nii.gz",
        )
    )
    predicitions = glob.glob(
        os.path.join(
            "U:\\",
            "lauraalvarez",
            "nnunet",
            "nnUNet_raw_data",
            task_name,
            "out",
            "*.nii.gz",
        )
    )
    true_labels = glob.glob(
        os.path.join(
            "U:\\",
            "lauraalvarez",
            "nnunet",
            "nnUNet_raw_data",
            task_name,
            "labelsTs",
            "*.nii.gz",
        )
    )

    data_dicts_test = [
        {"image": image_name, "label": label_name, "tLabel": true_name}
        for image_name, label_name, true_name in zip(images, predicitions, true_labels)
    ]

    csv_list = []
    for data in data_dicts_test[:]:
        print(f"Infering for \n\t image:{data['image']}, \n\t label: {data['label']}, \n\t true label: {data['tLabel']}")
        normal_plot = Compose(
            [
                LoadImaged(keys=["image", "label", "tLabel"]),
                AsChannelFirstd(keys=["image", "label", "tLabel"]),
                AddChanneld(keys=["label", "image", "tLabel"]),
                KeepLargestConnectedComponentd(keys=["label"], applied_labels=[1,2], is_onehot=False, independent=False),
                Orientationd(
                    keys=[
                        "image",
                        "label",
                    ],
                    axcodes="RAS",
                ),
                # NNUnetScaleIntensity(keys=["image"]),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
            ]
        )
        basename = os.path.basename(data["image"])
        injures = normal_plot(data)
        post_pred = Compose([AsDiscrete(to_onehot=3)])
        post_label = Compose([AsDiscrete(to_onehot=3)])
        outputs = torch.Tensor(np.expand_dims(post_pred(torch.Tensor(injures["label"])), 0))
        labels = torch.Tensor(np.expand_dims(post_label(torch.Tensor(injures["tLabel"])), 0))
        dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        dice_metric(y_pred=outputs, y=labels)
        dice_score_organ, dice_score_injure = dice_metric.aggregate()

        dict_data = {
            "image": basename,
            "dice_score_organ": dice_score_organ.numpy(),
            "dice_score_injure": dice_score_injure.numpy(),
        }
        csv_list.append(dict_data)

        post_plotting = Compose([EnsureType(), AsDiscrete(argmax=False)])
        injures["label"] = post_plotting(injures["label"])
        inj = dict(injures)
        inj = Resized(keys=["image", "label", "tLabel"], spatial_size=(160, 160, 160))(
            inj
        )
        # injures["label"] = np.expand_dims(injures["label"],0)
        # injures["image"] = np.expand_dims(injures["image"],0)

        blended_label_in = blend_images(inj["image"], inj["label"], 0.5)
        blended_final = blended_label_in.permute(1, 2, 0, 3)

        blended_true_label = blend_images(inj["image"], inj["tLabel"], 0.5)
        blended_true_label = torch.from_numpy(blended_true_label).permute(1, 2, 0, 3)

        volume = torch.hstack(
            (
                torch.from_numpy(inj["image"]).permute(1, 2, 0, 3).repeat(1, 1, 3, 1),
                blended_final,
                blended_true_label,
            )
        )
        volume = volume.permute(0, 1, 3, 2)

        volume_path = _save_gif(volume.numpy(), f"{basename}", task_name)
        # _save_gif(blended_true_label.numpy().transpose(0, 1, 3, 2), f"{basename}_True", task_name)
        # _save_gif(blended_final.numpy().transpose(0, 1, 3, 2), f"{basename}_Pred", task_name)

        print(f"Saved {volume_path}")
        save_csv("summary.csv",task_name, csv_list)


def main():
    create_figs("", "", "")  # TODO Desharcodear


if __name__ == "__main__":
    main()
