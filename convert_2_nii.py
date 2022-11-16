"""
    Convert trauma dataset to nifti format to be input to nn-Unet, logic supports liver, spleen and vascular injuries datasets
"""

import csv
import glob
import SimpleITK as sitk
import os
import numpy as np
import shutil
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

    # Calculate the difference in z size
    overlay_init = np.abs(1 / mha_data.GetSpacing()[-1] * (mha_org - overlay_org)) # calculate init position of overlay
    lower_bound = int(overlay_init) # lower bound of the overlay
    upper_bound = label.shape[0] # upper bound of the overlay
    zeros_up = lower_bound # number of zeros to be added at the beginning
    zeros_down = original_z_size - (upper_bound + lower_bound) # number of zeros to be added at the end
    
    new = list()

    # Add zeros at the beginning
    if zeros_up > 0:
        new.append(
            np.zeros((zeros_up, label.shape[1], label.shape[2]), dtype=label.dtype)
        )

    new.append(label)

    # Add zeros at the end
    if zeros_down > 0:
        new.append(
            np.zeros((zeros_down, label.shape[1], label.shape[2]), dtype=label.dtype)
        )

    label = np.concatenate(new, axis=0)

    return label


def one_channel_overlay(img, organ):
    """Generate a One channel overlay

    Args:
        img (Array): Array of the original MHA overlay

    Returns:
        Array: Array of the overlay with one channel
    """
    mha_img = sitk.GetArrayFromImage(img)
    mha_img = mha_img.astype(np.int8)

    # Create a new array with the same shape as the original MHA
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
            # Convert to binary: we need to isolate the liver and transform the label (original label set to 6) to 1
            liver_channel = np.where((mha_img != 6), 0, mha_img)
            liver_channel = np.where((liver_channel == 6), 1, liver_channel)
            new_labels = liver_channel
        elif organ== "spleen":
            # Convert to binary: we need to isolate the spleen and transform the label (original label set to 6) to 1
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
        elif organ == "vascular_injuries":
            # labels = [4,5] # untoggle if both organs are needed
            labels = [5]
        else:
            labels = [0,1,2,3,4,5] # if all consitions want to be included in classification
        
        # merge labels
        channels = list()
        for i, channel in enumerate(labels):
            c = mha_img[channel, :, :, :]
            c = np.where((c == 1), i + 1, c)
            channels.append(c)

        # stack labels into one channel
        chans = np.stack(channels, axis=0)
        new_labels = np.max(chans, axis=0)

    return new_labels


def save_csv(output_path, data):
    """Save the data to a csv file to keep track of ID´s that have been converted

    Args:
        output_path (str): Path to the output folder
        data (Array): Data to be saved
    """
    import csv

    keys = data[0].keys() # get the ID
    a_file = open(output_path, "w") # open the file
    dict_writer = csv.DictWriter(a_file, keys) # create the writer
    dict_writer.writeheader() # write the header
    dict_writer.writerows(data) # write the data
    a_file.close() # close the file


def channel_first(img_mask, img):
    """Convert the Overlay to channel first

    Args:
        img_mask (Array): Overlay Array
        img (Array): Original MHA Array

    Returns:
        Array: Overlay matching channel first as MHA
    """
    img = sitk.GetArrayFromImage(img) # get the MHA as array
    if img.shape[0] != img_mask.shape[0]: # if the overlay is not the same size as the MHA, we need to resize it
        if img.shape[0] == img_mask.shape[-1] and img.shape[-1] == img_mask.shape[0]: # if the overlay is channel last, we need to convert it to channel first
            img_mask = img_mask.transpose((2, 1, 0)) # convert to channel first
    return img_mask


def convert_dataset(MODE, file_identifier="TRM", organ="vascular_injuries", task_name="Task505_SpleenTrauma"):
    """
    
    Convert the MHA dataset folder into Nifti format ready to be used with nnUnet

    Args:
        MODE (str): Path to the output folder
        file_identifier (str): Prefix to save the converted images correctly
        task_name (str): Name of the task to be used in the nnUnet and as a folder
    """
    
    if MODE == "train": name = 'Tr' # if we are converting the training set
    else: name = 'Ts' # if we are converting the test set
    home = "/mnt/chansey/" # path to the dataset

    # path to the MHA files
    train_images = sorted(
        glob.glob(
            os.path.join(
                home, "lauraalvarez", "data", organ, "mha", f"images{name}", "*.mha"
            )
        )
    )

    # root path to save niis
    BASE_PATH = os.path.join(
        home,
        "lauraalvarez",
        "nnunet",
        "nnUNet_raw_data_base",
        "nnUNet_raw_data",
    )

    # path to save the new task for the converted images
    task_folder_name = os.path.join(BASE_PATH, task_name)

    # path to the csv file to keep track of the converted images
    if os.path.exists(os.path.join(task_folder_name, f"equivalence_{MODE}.csv")): # if the csv file already exists, we need to load it
        with open(os.path.join(task_folder_name, f"equivalence_{MODE}.csv"), "r") as f: # open the csv file
            reader = csv.reader(f) # read the csv file
            csv_data = list(reader)[1:] # get the data
            equivalence_l = [{"mha": row[0], "nii": row[1]} for row in csv_data] # load the past equivalence ids
    else:
        equivalence_l = list()


    if MODE == "train": # if we are converting the training set
        image_dir = os.path.join(task_folder_name, "imagesTr") # path to save the converted training images
        label_dir = os.path.join(task_folder_name, "labelsTr") # path to save the converted training labels

    else:
        image_dir = os.path.join(task_folder_name, "imagesTs") # path to save the converted test images
        label_dir = os.path.join(task_folder_name, "labelsTs") # path to save the converted test labels

    make_if_dont_exist(task_folder_name, overwrite=False) # create the task folder
    make_if_dont_exist(image_dir) # create the folder to save the converted images
    make_if_dont_exist(label_dir) # create the folder to save the converted labels


    # List to store all ids of scans that do not have labels for the injuries
    no_injuries = list()


    # loop to convert and save the images and labels 
    for i in tqdm(range(0, len(train_images))):

        # i = get_correct_index(i, os.path.basename(train_images[i]).split(".")[0], equivalence_l)
        save_filename = file_identifier + "_%03i_0000.nii.gz" % i
        equiv = {
            "mha": os.path.basename(train_images[i]).split(".")[0],
            "nii": save_filename,
        }
        # Add to list of converted files
        if equiv not in equivalence_l:
            equivalence_l.append(equiv)
            print("")
            print("Converting {}".format(train_images[i]))
        else:
            print(f"{equiv} already exists. Skipping")
        try:
            if not os.path.exists(os.path.join(image_dir, save_filename)):
                # read the original image
                img = sitk.ReadImage(train_images[i])
                # Get the array from the image and recreate it without any extra metadata
                img_array = sitk.GetArrayFromImage(img)
                new_img = sitk.GetImageFromArray(img_array)
                # Add the metadata we want to keep
                new_img.SetSpacing(img.GetSpacing()) # set the spacing
                new_img.SetOrigin(img.GetOrigin()) # set the origin
                new_img.SetDirection(img.GetDirection()) # set the direction
                # Orient the image to RAS
                img = sitk.DICOMOrient(new_img, "RAS")
                # save the NII file
                print("Saving to  {}".format(os.path.join(image_dir, save_filename)))
                sitk.WriteImage(img, os.path.join(image_dir, save_filename))
            # Read the new image we just created and use it to obtain metadata
            # duplicate readings but it does not work okay otherwise.. -.-"
            img = sitk.ReadImage(os.path.join(image_dir, save_filename))
            filename = os.path.basename(train_images[i])
            # get the correct index of the csv file
            labelpath = os.path.join(
                home,
                "lauraalvarez",
                "data",
                organ,
                "mha",
                f"labels{name}",
                filename,
            )

            save_filename = file_identifier + "_%03i.nii.gz" % i # name of the file
            if not os.path.exists(os.path.join(label_dir, save_filename)):
                print(f"Converting mask for {labelpath}")
                try:
                    img_mask = sitk.ReadImage(labelpath)
                except Exception as e:
                    print(e)
                    print("Error reading {}".format(labelpath))
                    continue
                # Adapt the channel
                img_array = one_channel_overlay(img_mask, organ)

                # If overlay does not have an injury, record finding
                if len(np.unique(img_array)) <= 1:
                    no_injuries.append({"file": os.path.basename(labelpath), "labels": np.unique(img_array)})

                img_array = channel_first(img_array, img) # convert to channel first format
                img_array = adapt_overlay(labelpath, img, img_array) # adapt the overlay
                img_array = sitk.GetImageFromArray(img_array) # conver array to itk image

                print("Saving to  {}".format(os.path.join(label_dir, save_filename)))
                sitk.WriteImage(img_array, os.path.join(label_dir, save_filename)) 
                print("Reading image again...")
            
                img_array = sitk.ReadImage(os.path.join(label_dir, save_filename))
                print("Orienting image...")
                img_array = sitk.DICOMOrient(img_array, "RAS") # orient the image
                print(f"spacing: {img_array.GetSpacing()}")
                img_array.SetSpacing(img.GetSpacing()) # set the spacing
                img_array.SetOrigin(img.GetOrigin())  # set the origin
                print(f"img shape {img.GetSize()}")
                print(f"img spacing {img.GetSpacing()}")
                print(f"img origin {img.GetOrigin()}")
                print(f"shape: {img_array.GetSize()}")
                print(f"spacing: {img_array.GetSpacing()}")
                print(f"origin: {img_array.GetOrigin()}")
                print("Saving image again...")
                sitk.WriteImage(img_array, os.path.join(label_dir, save_filename)) # save the image
            else:
                print(f"{save_filename} already exists. Skipping")
        # Save the csv file for each iteration in case of error
        except Exception as e:
            print(e)
            print("Error reading {}".format(train_images[i]))
            continue

    # Save the csv file with the equivalence between the original and the new files
    save_csv(
        os.path.join(task_folder_name, f"equivalence_{MODE}.csv"), equivalence_l
    )

    # Save the csv file with the files that do not have injuries
    if len(no_injuries) > 0:
        save_csv(
            os.path.join(task_folder_name, f"no_{organ}_{MODE}.csv"), no_injuries
        )
    else:
        print(f"All labels for {organ} found")

    print(no_injuries)

    print(f"Finished converting {MODE} to NII")


def main():
    task_name = "Task514_VISpleenTrauma" # name of the task
    MODES = ["train", "test"] # modes to convert
    for MODE in MODES:
        organ = "vascular_injuries" # name of the organ: vascular_injuries, spleen, liver
        name = "VI" # identifier for the files "VI" for vascular injuries, "TRMSPL" for spleen, "TLIV" for liver
        convert_dataset(MODE, name, organ.lower(), task_name) # convert the dataset

if __name__ == "__main__":
    main()
