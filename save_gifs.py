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
    EnsureTyped,
    KeepLargestConnectedComponent,
    KeepLargestConnectedComponentd,
    LabelToContour,
    FillHolesd,
    Rand3DElasticd,
)

from skimage.metrics import adapted_rand_error
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
from monai.metrics import DiceMetric, MeanIoU
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import csv
import cv2

# import cc3d
# import morphsnakes as ms
import cv2
import imageio
from collections import Counter
from skimage.segmentation import (
    morphological_chan_vese,
    morphological_geodesic_active_contour,
    inverse_gaussian_gradient,
    expand_labels,
)
from skimage.morphology import disk, dilation, binary_dilation, ball, cube, closing
from scipy import ndimage
from monai.data import MetaTensor
import skimage.measure as measure

from utils import injury_postprocessing, save_gif, save_csv


OUT_FOLDER = "out_unet"
GIF_FOLDER = "nnunet_gifs_full"
ORGAN = "Liver"


def create_figs(
    task_name="Task510_LiverTraumaDGX",
):  # Task510_LiverTraumaDGX Task511_SpleenTraumaCV
    """
    Create the figures for the gif.

    Args:
        image_path (str): Path to the image.
        prediction_path (str): Path to the prediction.
        label_path (str): Path to the label.
        task_name (str, optional): Name of the task.
    """
    # Path to the image
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
    # Path to the prediction
    predicitions = glob.glob(
        os.path.join(
            "U:\\",
            "lauraalvarez",
            "nnunet",
            "nnUNet_raw_data",
            task_name,
            OUT_FOLDER,
            "*.nii.gz",
        )
    )
    # Path to the label
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

    data_dicts_test = (
        [  # list of dictionaries with the paths to the images, predictions and labels
            {"image": image_name, "label": label_name, "tLabel": true_name}
            for image_name, label_name, true_name in zip(
                images, predicitions, true_labels
            )
        ]
    )

    csv_list = []
    for data in data_dicts_test[:]:
        print(
            f"Infering for image:{data['image']}, label: {data['label']}, true label: {data['tLabel']}"
        )
        normal_plot = Compose(
            [
                LoadImaged(keys=["image", "label", "tLabel"]),  # load the images
                AsChannelFirstd(
                    keys=["image", "label", "tLabel"]
                ),  # make sure the channel is the first dimension
                AddChanneld(
                    keys=["label", "image", "tLabel"]
                ),  # add a channel dimension
                CropForegroundd(
                    keys=["image", "label", "tLabel"], source_key="image"
                ),  # crop the foreground
                ScaleIntensityRanged(  # scale the intensity
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                KeepLargestConnectedComponentd(
                    keys=["label"],
                    applied_labels=[1, 2],
                    is_onehot=False,
                    independent=False,
                ),  # keep the largest connected component
                injury_postprocessing(
                    keys=["image", "label"],
                    organ=ORGAN,
                    settings={
                        "iterations": 2,
                        "smoothing": 2,
                        "balloon": 0,
                        "threshold": "auto",
                        "sigma": 2,
                        "alpha": 7000,
                    },
                ),  # postprocess the prediction
                EnsureTyped(
                    keys=["label", "image", "tLabel"], data_type="tensor"
                ),  # make sure the data is a tensor
            ]
        )

        basename = os.path.basename(data["image"])  # get the basename of the image
        injures = normal_plot(data)  # apply the transforms

        # Note: metrics can also be computed runing the file run_metrics (faster inference)
        post_pred = Compose(
            [AsDiscrete(to_onehot=3)]
        )  # transform the prediction to onehot
        post_label = Compose([AsDiscrete(to_onehot=3)])  # transform the label to onehot
        outputs = torch.Tensor(
            np.expand_dims(post_pred(injures["label"]), 0)
        )  # convert to tensor
        labels = torch.Tensor(
            np.expand_dims(post_label(torch.Tensor(injures["tLabel"])), 0)
        )  # convert to tensor
        dice_metric = DiceMetric(
            include_background=False, reduction="mean_batch"
        )  # initialize the dice metric
        jaccard = MeanIoU(
            include_background=False, reduction="mean_batch"
        )  # initialize the jaccard metric
        dice_metric(y_pred=outputs, y=labels)  # compute the dice metric
        jaccard(y_pred=outputs, y=labels)  # compute the jaccard metric
        (
            dice_score_organ,
            dice_score_injury,
        ) = dice_metric.aggregate()  # get the dice score
        (
            jaccard_score_organ,
            jaccard_score_injury,
        ) = jaccard.aggregate()  # get the jaccard score
        _, precision_organ, recall_organ = adapted_rand_error(
            labels[:, 1, :, :, :].numpy().astype(int),
            outputs[:, 1, :, :, :].numpy().astype(int),
        )  # compute the precision and recall
        _, precision_injury, recall_injury = adapted_rand_error(
            labels[:, 2, :, :, :].numpy().astype(int),
            outputs[:, 2, :, :, :].numpy().astype(int),
        )  # compute the precision and recall

        # untoogle if you want to run a multiorgan segmentation
        # dice_score_organ_1, dice_score_organ_2, dice_score_injure_1, dice_score_injure_2 = dice_metric.aggregate()

        # save the results in a csv file
        dict_data = {
            "image": basename,
            "dice_score_organ": dice_score_organ.numpy(),
            "dice_score_injury": dice_score_injury.numpy(),
            "jaccard_score_organ": jaccard_score_organ.numpy(),
            "jaccard_score_injury": jaccard_score_injury.numpy(),
            "precision_organ": precision_organ,
            "recall_organ": recall_organ,
            "precision_injury": precision_injury,
            "recall_injury": recall_injury,
        }

        csv_list.append(dict_data)
        save_gif = True

        if save_gif == True:  # save the gif
            post_plotting = Compose(
                [AsDiscrete(argmax=False), EnsureType(data_type="tensor")]
            )  # transform the prediction to onehot
            injures["label"] = post_plotting(
                injures["label"]
            )  # post process the prediction
            inj = dict(injures)  # convert to dictionary
            inj = Resized(
                keys=["image", "label", "tLabel"], spatial_size=(512, 512, 512)
            )(
                inj
            )  # resize the images
            blended_label_in = blend_images(
                inj["image"], inj["label"], 0.5
            )  # blend the prediction with the image
            blended_final = blended_label_in.permute(1, 2, 0, 3)  # permute the images
            blended_true_label = blend_images(
                inj["image"], inj["tLabel"], 0.5
            ).numpy()  # blend the golden label with the image
            blended_true_label = torch.from_numpy(blended_true_label).permute(
                1, 2, 0, 3
            )  # permute the images

            volume = torch.hstack(  # stack the images in the Y axis
                (
                    inj["image"].permute(1, 2, 0, 3).repeat(1, 1, 3, 1),
                    blended_final,
                    blended_true_label,
                )
            )
            volume = volume.permute(0, 1, 3, 2)

            volume_path = save_gif(
                volume.numpy(), f"{basename}", task_name, OUT_FOLDER, GIF_FOLDER
            )  # save the gif/video
            print(f"Saved {volume_path}")

        save_csv(
            "summary.csv", task_name, csv_list, OUT_FOLDER, GIF_FOLDER
        )  # save the csv file
    print("Finished")


def main():
    create_figs()  # TODO Desharcodear


if __name__ == "__main__":
    main()
