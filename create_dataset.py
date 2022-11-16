"""

Snippet to create the datset file necessary to run the nnunet framework.

Generate_dataset_json function can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/aa53b3b87130ad78f0a28e6169a83215d708d659/nnunet/dataset_conversion/utils.py#L27

"""
import os
from batchgenerators.utilities.file_and_folder_operations import join
from nnunet.dataset_conversion.utils import generate_dataset_json

HOME = "/mnt/chansey/"

if __name__ == '__main__':


    # now start the conversion to nnU-Net:
    task_name = "Task514_VISpleenTrauma" # name of the task
    labels = ["Spleen_Vascular_Injure"] # name of the labels

    BASE_PATH = os.path.join(
        HOME,
        "lauraalvarez",
        "nnunet",
        "nnUNet_raw_data",
    )

 
    # path to the task folder
    target_base = join(BASE_PATH, task_name)

    # path to the images
    target_imagesTs = os.path.join(
                    HOME, "lauraalvarez", "nnunet", "nnUNet_raw_data", task_name, "imagesTs"
                )

    target_imagesTr = os.path.join(
                    HOME, "lauraalvarez", "nnunet", "nnUNet_raw_data", task_name, "imagesTr",
                )

    # function to generate the dataset.json file
    generate_dataset_json(join(HOME, "lauraalvarez", "nnunet", "nnUNet_raw_data", task_name, 'dataset.json'), target_imagesTr, target_imagesTs, (['CT']), labels={0: 'background', 1: labels[0]}, dataset_name=task_name, license='hands off!')