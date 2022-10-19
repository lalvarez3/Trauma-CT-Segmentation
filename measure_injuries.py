import csv
import glob
import SimpleITK as sitk
import os
import numpy as np
import shutil
from tqdm import tqdm
import skimage.measure as measure
from scipy import ndimage


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


def get_connected_components_size(init_label, selected_label):
    result = {}
    init_label_ = init_label.copy()
    foreground = np.where((init_label_ != selected_label), 0, init_label_)
    labelling, label_count = measure.label(
        foreground == selected_label, return_num=True
    )
    init_clusters = np.unique(labelling, return_counts=True)
    for n in range(1, label_count + 1):
        cluster_size = ndimage.sum(labelling == n)
        result[n] = cluster_size

    return result


def save_csv(output_path, data, keys=None):
    import csv

    keys = data[0].keys() if keys is None else keys
    with open(output_path, "w") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)
    a_file = open(output_path, "w")
    dict_writer = csv.DictWriter(a_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(data)
    a_file.close()


def measure_dataset(MODE, task_name="Task505_SpleenTrauma"):

    if MODE == "train":
        name = "Tr"
    else:
        name = "Ts"

    home = "U:\\"
    train_images = sorted(
        glob.glob(
            os.path.join(
                home,
                "lauraalvarez",
                "nnunet",
                "nnUNet_raw_data",
                task_name,
                f"labels{name}",
                "*.nii.gz",
            )
        )
    )

    BASE_PATH = os.path.join(
        home,
        "lauraalvarez",
        "nnunet",
        "nnUNet_raw_data",
    )

    task_folder_name = os.path.join(BASE_PATH, task_name)

    equivalence_l = list()

    # load the csv file with the data
    total_keys = set()

    for i in tqdm(range(0, len(train_images))):

        print("\nMeasuring {}".format(train_images[i]))

        try:
            # read the original image
            img = sitk.ReadImage(train_images[i])
            # Get the array from the image and recreate it without any extra metadata
            img_array = sitk.GetArrayFromImage(img)

            sizes = get_connected_components_size(img_array, 2)
            sizes["name"] = os.path.basename(train_images[i])
            keys = sizes.keys()
            total_keys.update(keys)
            equivalence_l.append(sizes)

        # Save the csv file for each iteration in case of error
        except Exception as e:
            print(e)
            print("Error reading {}".format(train_images[i]))
            continue

    save_csv(
        os.path.join(task_folder_name, f"Injury_measures{MODE}.csv"), equivalence_l, total_keys
    )

    print(f"Finished measuring {MODE} to NII")


def main():
    task_name = "Task511_SpleenTraumaCV"
    MODES = ["train", "test"]
    for MODE in MODES:
        measure_dataset(MODE, task_name)


if __name__ == "__main__":
    main()
