"""
    Convert Vascular Injuries dataset to nifti format to be input to nn-Unet
"""

import csv
import glob
import SimpleITK as sitk
import os
import sys
import numpy as np
import shutil
from tqdm import tqdm


# change here for different task name


def copy_and_rename(
    old_location, old_file_name, new_location, new_filename, delete_original=False
):

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

    # Calculate the difference in z size
    overlay_init = np.abs(1 / mha_data.GetSpacing()[-1] * (mha_org - overlay_org)) # calculate init position of overlay
    lower_bound = int(overlay_init) # lower bound of the overlay
    upper_bound = label.shape[0]  # upper bound of the overlay
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


def one_channel_overlay(img, organ):
    mha_img = sitk.GetArrayFromImage(img)
    mha_img = mha_img.astype(np.int8)

    if len(mha_img.shape) == 3:
        z, h, w = mha_img.shape
        c = 1
    else:
        c, z, h, w = mha_img.shape

    new_labels = np.zeros((z, h, w), np.int8)

    # Depending on the number of channels, we have to adapt the overlay (GC has 1 working, the other has 6 channels)
    # We want 0 and 2 for channels and
    if c == 1:
        if organ == "liver":
            # we need to transform Liver (6) to 1
            liver_channel = np.where((mha_img != 6), 0, mha_img)
            liver_channel = np.where((liver_channel == 6), 1, liver_channel)
            new_labels = liver_channel
        elif organ== "spleen":
            spleen_channel = np.where((mha_img != 1), 0, mha_img)
            spleen_channel = np.where((spleen_channel == 1), 1, spleen_channel)
            new_labels = spleen_channel
        else:
            liver_channel = np.where((mha_img != 6), 0, mha_img)
            liver_channel = np.where((liver_channel == 6), 1, liver_channel)
            spleen_channel = np.where((mha_img != 1), 0, mha_img)
            spleen_channel = np.where((spleen_channel == 1), 1, spleen_channel)
            new_labels = liver_channel + spleen_channel
    else:
        # each channel is a different label, we want 0(liver = 1) and 2(liver injure = 2) and the background to be 0
        # 0:liver 1:spleen 2 :liver_injure 3:Spleen_injure 4:VLI 5:VSI
        if organ == "liver":
            labels = [0,2]
        elif organ == "spleen":
            labels = [1,3]
        else:
            labels = [0,1,2,3]
        channels = list()
        for i, channel in enumerate(labels):
            c = mha_img[channel, :, :, :]
            c = np.where((c == 1), i + 1, c)
            channels.append(c)

        # we got channels splitted with 1 each. if we add them we should get what we want
        chans = np.stack(channels, axis=0)
        new_labels = np.max(chans, axis=0)

    return new_labels


def save_csv(output_path, data):
    import csv

    keys = data[0].keys()
    a_file = open(output_path, "w")
    dict_writer = csv.DictWriter(a_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(data)
    a_file.close()


def check_valid(path, filename):
    file_path = os.path.join(path, filename)
    if not os.path.exists(file_path):
        return False
    try:
        img = sitk.ReadImage(file_path)
        if img.GetSize()[0] == 0:
            return False
    except:
        return False

    return True

def get_correct_index(index, filename, csv_data):
    """
    This function is used to get the correct index of the csv file.
    """
    exist = filename in csv_data
    if exist:
        index = csv_data.index(filename)
    else:
        index = len(csv_data)

    return index

def channel_first(img_mask, img):
    img = sitk.GetArrayFromImage(img)
    if img.shape[0] != img_mask.shape[0]:
        if img.shape[0] == img_mask.shape[-1] and img.shape[-1] == img_mask.shape[0]:
            img_mask = img_mask.transpose((2, 1, 0))
    return img_mask


def convert_dataset(MODE, file_identifier="TRM", organ="spleen", task_name="Task505_SpleenTrauma"):

    if MODE == "train": name = 'Tr'
    else: name = 'Ts'

    home = "/mnt/chansey"

    train_images = sorted(
        glob.glob(
            os.path.join(
                home, "lauraalvarez", "data", organ, f"images{name}", "*.mha"
            )
        )
    )

    BASE_PATH = os.path.join(
        home,
        "lauraalvarez",
        "nnunet",
        "nnUNet_raw_data_base",
        "nnUNet_raw_data",
    )

    task_folder_name = os.path.join(BASE_PATH, task_name)

    if os.path.exists(os.path.join(task_folder_name, f"equivalence_{MODE}.csv")):
        with open(os.path.join(task_folder_name, f"equivalence_{MODE}.csv"), "r") as f:
            reader = csv.reader(f)
            csv_data = list(reader)[1:]
            equivalence_l = [{"mha": row[0], "nii": row[1]} for row in csv_data]
    else:
        equivalence_l = list()



    if MODE == "train":
        train_image_dir = os.path.join(task_folder_name, "imagesTr")
        train_label_dir = os.path.join(task_folder_name, "labelsTr")

    else:
        train_image_dir = os.path.join(task_folder_name, "imagesTs")
        train_label_dir = os.path.join(task_folder_name, "labelsTs")

    make_if_dont_exist(task_folder_name, overwrite=False)
    make_if_dont_exist(train_image_dir)
    make_if_dont_exist(train_label_dir)
    # make_if_dont_exist(test_dir,overwrite= False)

    no_spleen = list()

    # load the csv file with the data

    for i in tqdm(range(0, len(train_images))):

        # i = get_correct_index(i, os.path.basename(train_images[i]).split(".")[0], equivalence_l)
        save_filename = file_identifier + "_%03i_0000.nii.gz" % i
        equiv = {
            "mha": os.path.basename(train_images[i]).split(".")[0],
            "nii": save_filename,
        }
        if equiv not in equivalence_l:
            equivalence_l.append(equiv)
            print("Converting {}".format(train_images[i]))
        else:
            print(f"{equiv} already exists. Skipping")
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
                # save the NII file
                print("Saving to  {}".format(os.path.join(train_image_dir, save_filename)))
                sitk.WriteImage(img, os.path.join(train_image_dir, save_filename))
            # Read the new image we just created and use it to obtain metadata
            # duplicate readings but it does not work okay otherwise.. -.-"
            img = sitk.ReadImage(os.path.join(train_image_dir, save_filename))
            filename = os.path.basename(train_images[i])
            # get the correct index of the csv file
            labelpath = os.path.join(
                home,
                "lauraalvarez",
                "data",
                organ,
                f"labels{name}",
                filename,
            )

            save_filename = file_identifier + "_%03i.nii.gz" % i
            if not os.path.exists(os.path.join(train_label_dir, save_filename)):
                print(f"Converting mask for {labelpath}")
                try:
                    img_mask = sitk.ReadImage(labelpath)
                except Exception as e:
                    print(e)
                    print("Error reading {}".format(labelpath))
                    continue
                # Adapt the channel
                img_array = one_channel_overlay(img_mask, organ)
                # Transform the label to the same size as the image
                if len(np.unique(img_array)) < 3:
                    no_spleen.append({"file": os.path.basename(labelpath), "labels": np.unique(img_array)})

                img_array = channel_first(img_array, img)
                img_array = adapt_overlay(labelpath, img, img_array)
                img_array = sitk.GetImageFromArray(img_array)

                print("Saving to  {}".format(os.path.join(train_label_dir, save_filename)))
                sitk.WriteImage(img_array, os.path.join(train_label_dir, save_filename))
                print("Reading image again...")

                img_array = sitk.ReadImage(os.path.join(train_label_dir, save_filename))
                print("Orienting image...")
                img_array = sitk.DICOMOrient(img_array, "RAS")
                print(f"spacing: {img_array.GetSpacing()}")
                img_array.SetSpacing(img.GetSpacing())
                img_array.SetOrigin(img.GetOrigin())
                print(f"img shape {img.GetSize()}")
                print(f"img spacing {img.GetSpacing()}")
                print(f"img origin {img.GetOrigin()}")
                print(f"shape: {img_array.GetSize()}")
                print(f"spacing: {img_array.GetSpacing()}")
                print(f"origin: {img_array.GetOrigin()}")
                print("Saving image again...")
                sitk.WriteImage(img_array, os.path.join(train_label_dir, save_filename))
            else:
                print(f"{save_filename} already exists. Skipping")
        # Save the csv file for each iteration in case of error
        except Exception as e:
            print(e)
            print("Error reading {}".format(train_images[i]))
            continue

    save_csv(
        os.path.join(task_folder_name, f"equivalence_{MODE}.csv"), equivalence_l
    )

    if len(no_spleen) > 0:
        save_csv(
            os.path.join(task_folder_name, f"no_{organ}_{MODE}.csv"), no_spleen
        )
    else:
        print(f"All labels for {organ} found")

    print(no_spleen)

    print(f"Finished converting {MODE} to NII")


def main():
    task_name = "Task510_LiverTraumaDGX"
    MODES = ["test"]
    for MODE in MODES:
        organ = "Liver"
        if organ == "Liver":
            name = "TLIV"
        elif organ == "no_injuries_extended":
            name = "TNI"
        elif organ == "spleen_liver":
            name = "TSpLi"
        else:
            name = "TRMSPL"
        convert_dataset(MODE, name, organ.lower(), task_name)

if __name__ == "__main__":
    main()
