import glob
import os
from imageio import save
from tqdm import tqdm

from convert_2_nii import main


def save_csv(output_path, data):
    import csv

    keys = data[0].keys()
    a_file = open(output_path, "w")
    dict_writer = csv.DictWriter(a_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(data)
    a_file.close()


def rename_files(mode, filepaths):
    conversions = []
    for i in tqdm(range(len(filepaths))):
        save_filename = "VI" + "_%03i_0000.nii.gz" % i
        file_mode = "imagesTr" if mode == "train" else "imagesTs"
        save_path = os.path.join(
            "U:\\",
            "lauraalvarez",
            "nnunet",
            "nnUNet_raw_data",
            "Task506_VITrauma",
            file_mode,
            save_filename,
        )
        os.rename(filepaths[i], save_path)
        conversions.append({"Original": filepaths[i], "NNUNET": save_path})

        save_filename = "VI" + "_%03i.nii.gz" % i
        file_mode = "labelsTr" if mode == "train" else "labelsTs"
        save_path = os.path.join(
            "U:\\",
            "lauraalvarez",
            "nnunet",
            "nnUNet_raw_data",
            "Task506_VITrauma",
            file_mode,
            save_filename,
        )

        label_path = filepaths[i].replace("imagesTr", "labelsTr") if mode == "train" else filepaths[i].replace("imagesTs", "labelsTs")
        os.rename(label_path, save_path)

    save_csv(
        os.path.join(
            "U:\\",
            "lauraalvarez",
            "nnunet",
            "nnUNet_raw_data",
            "Task506_VITrauma",
            f"equivalence_VI_{mode}.csv",
        ),
        conversions,
    )


def main():
    target_imagesTs = os.path.join(
        "U:\\",
        "lauraalvarez",
        "nnunet",
        "nnUNet_raw_data",
        "Task506_VITrauma",
        "imagesTs",
        "*.nii.gz",
    )

    target_imagesTr = os.path.join(
        "U:\\",
        "lauraalvarez",
        "nnunet",
        "nnUNet_raw_data",
        "Task506_VITrauma",
        "imagesTr",
        "*.nii.gz",
    )

    train_images = sorted(glob.glob(target_imagesTr))

    test_images = sorted(glob.glob(target_imagesTs))

    rename_files("train", train_images)
    rename_files("test", test_images)


if __name__ == "__main__":
    main()
