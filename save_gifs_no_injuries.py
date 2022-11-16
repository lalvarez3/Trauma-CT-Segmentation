import glob
import os

import numpy as np
import torch
from monai.transforms import (
    AddChanneld,
    AsChannelFirstd,
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureType,
    KeepLargestConnectedComponentd,
    LoadImaged,
    Resized,
    ScaleIntensityRanged,
)
from monai.visualize import blend_images

from utils import injury_postprocessing, save_csv

OUT_FOLDER = "out_no_injuries"
GIF_FOLDER = "gifs_no_injuries_2"
ORGAN = "Spleen"  # "Liver"
TASK = "Task511_SpleenTraumaCV"  #'Task510_LiverTraumaDGX'


def create_figs(task_name=TASK):
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
            "Task513_NIExtendedTrauma",
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

    data_dicts_test = [  # list of dictionaries with images and labels
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(images, predicitions)
    ]

    csv_list = []  # list of dictionaries with the metrics
    for data in data_dicts_test[:]:
        print(f"Infering for \n\t image:{data['image']}, \n\t label: {data['label']}")
        normal_plot = Compose(
            [
                LoadImaged(keys=["image", "label"]),  # load the image and the label
                AsChannelFirstd(
                    keys=["image", "label"]
                ),  # make sure the image and the label are in the right format
                AddChanneld(
                    keys=["label", "image"]
                ),  # add a channel to the image and the label
                CropForegroundd(
                    keys=["image", "label"], source_key="image"
                ),  # crop the image and the label
                ScaleIntensityRanged(  # scale the intensity of the image
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                KeepLargestConnectedComponentd(  # keep the largest connected component
                    keys=["label"],
                    applied_labels=[1, 2],
                    is_onehot=False,
                    independent=False,
                    connectivity=None,
                ),
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
                ),
            ]
        )
        basename = os.path.basename(data["image"])  # get the name of the image
        injuries = normal_plot(data)  # apply the transforms
        predicted_labels = np.unique(
            injuries["label"]
        )  # get the unique values of the label

        dict_data = {
            "image": basename,
            "predicted_labels": predicted_labels,
        }

        csv_list.append(dict_data)  # append the dictionary to the list
        save_gif = True  # Just for debugging purposes
        if save_gif == True:
            post_plotting = Compose(
                [EnsureType(), AsDiscrete(argmax=False)]
            )  # make sure the image and the label are in the right format
            injuries["label"] = post_plotting(injuries["label"])  # apply the transforms
            inj = dict(injuries)  # copy the dictionary
            inj = Resized(keys=["image", "label"], spatial_size=(512, 512, 512))(
                inj  # resize the image and the label
            )

            blended_label_in = blend_images(
                inj["image"], inj["label"], 0.5
            )  # blend the image and the label
            blended_final = blended_label_in.permute(
                1, 2, 0, 3
            )  # permute the image and the label

            volume = torch.hstack(  # stack the image and the label
                (
                    inj["image"].permute(1, 2, 0, 3).repeat(1, 1, 3, 1),
                    blended_final,
                )
            )
            volume = volume.permute(0, 1, 3, 2)
            volume_path = save_gif(
                volume.numpy(), f"{basename}", task_name, OUT_FOLDER, GIF_FOLDER
            )  # save the plot as a gif/video
            print(f"Saved {volume_path}")
        save_csv("summary.csv", task_name, csv_list, OUT_FOLDER, GIF_FOLDER)


def main():
    create_figs()


if __name__ == "__main__":
    main()
