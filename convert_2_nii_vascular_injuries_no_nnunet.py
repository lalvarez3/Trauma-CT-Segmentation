"""
    Convert Vasculair Injuries dataset to nifti format prepared to be injested by nn-Unet
"""
import glob
import os
import shutil
import sys

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# change here for different task name


def copy_and_rename(
    old_location, old_file_name, new_location, new_filename, delete_original=False
):
    """Copy and rename files

    Args:
        old_location (str): Location of the file to be copied
        old_file_name (str): Name of the file to be copied
        new_location (str): Location where the file will be copied
        new_filename (str): name of the new file
        delete_original (bool, optional): True if the original file should be removed. Defaults to False.
    """
    shutil.copy(os.path.join(old_location, old_file_name), new_location)
    os.rename(
        os.path.join(new_location, old_file_name),
        os.path.join(new_location, new_filename),
    )
    if delete_original:
        os.remove(os.path.join(old_location, old_file_name))


def make_if_dont_exist(folder_path, overwrite=False):
    """
    creates a folder if it does not exists
    input:
    folder_path : relative path of the folder which needs to be created
    over_write :(default: False) if True overwrite the existing folder
    """
    if os.path.exists(folder_path):

        if not overwrite:
            print(f"{folder_path} exists.")
        else:
            print(f"{folder_path} overwritten")
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)

    else:
        os.makedirs(folder_path)
        print(f"{folder_path} created!")


def adapt_overlay(overlay_path, mha_data, label):
    """Function to make sure that the overlay has the same shape as the original MHA

    Args:
        overlay_path (str): Path to the overlay
        mha_data (Array): Array of the original MHA
        label (str): Label to be applied to the overlay

    Returns:
        Array: New Image Array with the right overalay matching the original MHA
    """
    # Load the mha
    mha_org = mha_data.GetOrigin()[-1]
    # Load the mha image
    mha_img = sitk.GetArrayFromImage(mha_data)

    if mha_img.shape == label.shape:
        return label

    original_z_size = mha_img.shape[0]

    # Load the overlay
    overlay_data = sitk.ReadImage(overlay_path)
    overlay_org = overlay_data.GetOrigin()[-1]

    overlay_init = np.abs(1 / mha_data.GetSpacing()[-1] * (mha_org - overlay_org))

    lower_bound = int(overlay_init)
    upper_bound = label.shape[0]
    zeros_up = lower_bound
    zeros_down = original_z_size - (upper_bound + lower_bound)
    new = list()

    if zeros_up > 0:
        new.append(
            np.zeros((zeros_up, label.shape[1], label.shape[2]), dtype=label.dtype)
        )

    new.append(label)

    if zeros_down > 0:
        new.append(
            np.zeros((zeros_down, label.shape[1], label.shape[2]), dtype=label.dtype)
        )

    label = np.concatenate(new, axis=0)

    return label


def one_channel_overlay(img):
    """Generate a One channel overlay

    Args:
        img (Array): Array of the original MHA overlay

    Returns:
        Array: Array of the overlay with one channel
    """
    mha_img = sitk.GetArrayFromImage(img)
    mha_img = mha_img.astype(np.int8)

    if len(mha_img.shape) == 3:
        z, h, w = mha_img.shape
        c = 1
    else:
        c, z, h, w = mha_img.shape

    new_labels = np.zeros((z, h, w), np.int8)

    # each channel is a different label, we want 0(liver = 1) and 2(liver injure = 2) and the background to be 0
    # 0:liver 1:spleen 2 :liver_injure 3:Spleen_injure 4:VLI 5:VSI
    labels = [0, 1, 2, 3, 4, 5]
    channels = list()

    for i, channel in enumerate(labels):
        c = mha_img[channel, :, :, :]
        c = np.where((c == 1), i + 1, c)
        channels.append(c)

    # we got channels splitted with 1 each. if we add them we should get what we want
    chans = np.stack(channels, axis=0)
    new_labels = np.max(chans, axis=0)

    # values = np.unique(new_labels)
    # vi_classes = []
    # if not 5 in values: # Remove structues of the organs not containing VI
    #     new_labels = np.where(new_labels == 1, 0, new_labels) # remove liver
    #     new_labels = np.where(new_labels == 3, 0, new_labels) # remove liver injury
    # else: vi_classes.append('liver')

    # if not 6 in values:
    #     new_labels = np.where(new_labels == 2, 0, new_labels) # remove spleen
    #     new_labels = np.where(new_labels == 4, 0, new_labels) # remove spleen injury
    # else: vi_classes.append('spleen')

    return new_labels


def save_csv(output_path, data):
    """Save the data to a csv file to see which ID´s has been updated

    Args:
        output_path (str): Path to the output folder
        data (Array): Data to be saved
    """
    import csv

    keys = data[0].keys()
    a_file = open(output_path, "w")
    dict_writer = csv.DictWriter(a_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(data)
    a_file.close()


def channel_first(img_mask, img):
    """Convert the Overlay to channel first

    Args:
        img_mask (Array): Overlay Array
        img (Array): Original MHA Array

    Returns:
        Array: Overlay matching channel first as MHA
    """
    img = sitk.GetArrayFromImage(img)
    if img.shape[0] != img_mask.shape[0]:
        if img.shape[0] == img_mask.shape[-1] and img.shape[-1] == img_mask.shape[0]:
            img_mask = img_mask.transpose((2, 1, 0))
    return img_mask


def convert_dataset(MODE="train", file_identifier="VI_", task_name="nii"):
    """Convert the MHA dataset folder into Nifti format ready to be used with nnUnet

    Args:
        MODE (str): Path to the output folder
        file_identifier (str): Prefix to save the converted images correctly
        task_name (str): Name of the task to be used in the nnUnet and as a folder
    """

    home = "U:\\"

    if MODE == "train":
        name = "Tr"
    else:
        name = "Ts"

    train_images = sorted(
        glob.glob(
            os.path.join(
                home,
                "lauraalvarez",
                "data",
                "vascular_injuries",
                "mha",
                f"images{name}",
                "*.mha",
            )
        )
    )

    BASE_PATH = os.path.join(
        home,
        "lauraalvarez",
        "data",
        "vascular_injuries",
    )

    task_folder_name = os.path.join(BASE_PATH, task_name)

    if MODE == "train":
        train_image_dir = os.path.join(task_folder_name, "imagesTr")
        train_label_dir = os.path.join(task_folder_name, "labelsTr")

    else:
        train_image_dir = os.path.join(task_folder_name, "imagesTs")
        train_label_dir = os.path.join(task_folder_name, "labelsTs")

    make_if_dont_exist(task_folder_name, overwrite=False)
    make_if_dont_exist(train_image_dir)
    make_if_dont_exist(train_label_dir)

    no_vascular_injuries = list()

    for i in tqdm(range(0, len(train_images))):

        patient_id = os.path.basename(train_images[i]).split(".")[0]
        save_filename = file_identifier + patient_id + ".nii.gz"

        print("Converting {} into {}".format(patient_id, save_filename))
        try:
            if not os.path.exists(os.path.join(train_image_dir, save_filename)):
                # read the original image
                img = sitk.ReadImage(train_images[i])
                # Get the array from the image and recreate it without any extra metadata
                img_array = sitk.GetArrayFromImage(img)
                new_img = sitk.GetImageFromArray(img_array)
                # Add the metadata we want to keep
                new_img.SetSpacing(img.GetSpacing())
                new_img.SetOrigin(img.GetOrigin())
                new_img.SetDirection(img.GetDirection())
                # Orient the image to RAS
                img = sitk.DICOMOrient(new_img, "RAS")
                print(
                    "Saving to  {}".format(os.path.join(train_image_dir, save_filename))
                )
                sitk.WriteImage(img, os.path.join(train_image_dir, save_filename))

            filename = os.path.basename(train_images[i])
            labelpath = os.path.join(
                home,
                "lauraalvarez",
                "data",
                "vascular_injuries",
                "mha",
                f"labels{name}",
                filename,
            )
            print(f"\nConverting mask for {labelpath}")

            if not os.path.exists(os.path.join(train_label_dir, save_filename)):
                try:
                    img_mask = sitk.ReadImage(labelpath)
                except Exception as e:
                    print(e)
                    print("Error reading {}".format(labelpath))
                    continue
                img = sitk.ReadImage(os.path.join(train_image_dir, save_filename))
                # Adapt the channel
                img_array = one_channel_overlay(img_mask)
                # Transform the label to the same size as the image
                img_array = channel_first(img_array, img)
                img_array = adapt_overlay(labelpath, img, img_array)
                img_array = sitk.GetImageFromArray(img_array)

                print(
                    "Saving to  {}".format(os.path.join(train_label_dir, save_filename))
                )
                sitk.WriteImage(img_array, os.path.join(train_label_dir, save_filename))
                print("Reading image again...")

                img_array = sitk.ReadImage(os.path.join(train_label_dir, save_filename))
                print("Orienting image...")
                img_array = sitk.DICOMOrient(img_array, "RAS")
                print(f"spacing: {img_array.GetSpacing()}")
                img_array.SetSpacing(img.GetSpacing())
                img_array.SetOrigin(img.GetOrigin())
                print(f"img shape {img_array.GetSize()}")
                print(f"img spacing {img_array.GetSpacing()}")
                print(f"img origin {img_array.GetOrigin()}")
                print(f"shape: {img_array.GetSize()}")
                print(f"spacing: {img_array.GetSpacing()}")
                print(f"origin: {img_array.GetOrigin()}")
                print("Saving image again...")
                sitk.WriteImage(img_array, os.path.join(train_label_dir, save_filename))

            # assert (img_array.GetArrayFromImage().shape == img.GetArrayFromImage().shape)
            # Save the csv file for each iteration in case of error
        except Exception as e:
            print(e)
            print("Error reading {}".format(train_images[i]))
            continue

    print(no_vascular_injuries)

    print(f"Finished converting {MODE} to NII")


def main():
    """Main function to convert the MHA dataset into Nifti format"""
    MODES = ["train"]
    for mode in MODES:
        convert_dataset(MODE=mode, file_identifier="VI_", task_name="nii")


if __name__ == "__main__":
    main()
