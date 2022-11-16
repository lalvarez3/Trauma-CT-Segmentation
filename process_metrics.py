"""
    Script to convert the JSON metrics files (from the nnunet) to CSV for easier extraction of data.
"""
import csv
import glob
import json
import os


def get_info(item, key):
    """
    Get info from json file
    """
    item = item[key]
    accuracy = item["Accuracy"]
    precision = item["Precision"]
    recall = item["Recall"]
    dice = item["Dice"]
    jaccard = item["Jaccard"]

    result = [key, accuracy, precision, recall, dice, jaccard]

    return result


HOME = "U://" # path to the dataset
TASK = "Task510_LiverTraumaDGX" # Task name

# path to the metrics files
paths_to_analyze = glob.glob(
    os.path.join(
        HOME, "lauraalvarez", "nnunet", "nnUNet_raw_data", TASK, "out_unet/*.json"
    )
)

# Define the structure of the data
csv_header = ["number", "accuracy", "precision", "recall", "dice", "jaccard"]


for i, path in enumerate(paths_to_analyze):
    # get filename
    filename = os.path.basename(path).split(".")[0] + ".csv"
    csv_path = os.path.join(
        HOME, "lauraalvarez", "nnunet", "nnUNet_raw_data/", TASK, "out_unet", filename
    )

    # Open the json file
    with open(path) as f:
        data = json.load(f)

    with open(csv_path, "w") as file:
        writer = csv.writer(file) # write data to the file
        writer.writerow(csv_header) # write header

        for items in data["results"]["all"]:
            new_row = get_info(items, "1")
            writer.writerow(new_row)
            new_row = get_info(items, "2")
            writer.writerow(new_row)
            # Untoggle for multiclass segmentation
            # new_row = get_info(items, "3")
            # writer.writerow(new_row)
            # new_row = get_info(items, "4")
            # writer.writerow(new_row)
