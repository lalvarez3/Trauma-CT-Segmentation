"""
File to convert the JSON files to CSV for easier extraction of data.
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


HOME = "U://"
TASK = "Task510_LiverTraumaDGX"

paths_to_analyze = glob.glob(
    os.path.join(
        HOME, "lauraalvarez", "nnunet", "nnUNet_raw_data", TASK, "out_unet/*.json"
    )
)

paths_to_analyze = paths_to_analyze

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
        writer = csv.writer(file)
        # 3. Write data to the file
        writer.writerow(csv_header)

        for items in data["results"]["all"]:
            new_row = get_info(items, "1")
            writer.writerow(new_row)
            new_row = get_info(items, "2")
            writer.writerow(new_row)
            # new_row = get_info(items, "3")
            # writer.writerow(new_row)
            # new_row = get_info(items, "4")
            # writer.writerow(new_row)
