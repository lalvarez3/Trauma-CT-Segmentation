import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir
from nnunet.utilities.file_conversions import convert_2d_image_to_nifti
import glob

HOME = "/mnt/chansey/"

if __name__ == '__main__':
    """
    nnU-Net was originally built for 3D images. It is also strongest when applied to 3D segmentation problems because a
    large proportion of its design choices were built with 3D in mind. Also note that many 2D segmentation problems,
    especially in the non-biomedical domain, may benefit from pretrained network architectures which nnU-Net does not
    support.
    Still, there is certainly a need for an out of the box segmentation solution for 2D segmentation problems. And
    also on 2D segmentation tasks nnU-Net cam perform extremely well! We have, for example, won a 2D task in the cell
    tracking challenge with nnU-Net (see our Nature Methods paper) and we have also successfully applied nnU-Net to
    histopathological segmentation problems.
    Working with 2D data in nnU-Net requires a small workaround in the creation of the dataset. Essentially, all images
    must be converted to pseudo 3D images (so an image with shape (X, Y) needs to be converted to an image with shape
    (1, X, Y). The resulting image must be saved in nifti format. Hereby it is important to set the spacing of the
    first axis (the one with shape 1) to a value larger than the others. If you are working with niftis anyways, then
    doing this should be easy for you. This example here is intended for demonstrating how nnU-Net can be used with
    'regular' 2D images. We selected the massachusetts road segmentation dataset for this because it can be obtained
    easily, it comes with a good amount of training cases but is still not too large to be difficult to handle.
    """

    # download dataset from https://www.kaggle.com/insaff/massachusetts-roads-dataset
    # extract the zip file, then set the following path according to your system:
    # this folder should have the training and testing subfolders

    # now start the conversion to nnU-Net:
    task_name = "Task514_VISpleenTrauma"
    labels = ["Spleen_Vascular_Injure"]

    BASE_PATH = os.path.join(
        HOME,
        "lauraalvarez",
        "nnunet",
        "nnUNet_raw_data",
    )

    target_base = join(BASE_PATH, task_name)

    # create the dataset json file
    target_imagesTs = os.path.join(
                    HOME, "lauraalvarez", "nnunet", "nnUNet_raw_data", task_name, "imagesTs"
                )

    target_imagesTr = os.path.join(
                    HOME, "lauraalvarez", "nnunet", "nnUNet_raw_data", task_name, "imagesTr",
                )

    # finally we can call the utility for generating a dataset.json
    generate_dataset_json(join(HOME, "lauraalvarez", "nnunet", "nnUNet_raw_data", task_name, 'dataset.json'), target_imagesTr, target_imagesTs, (['CT']), labels={0: 'background', 1: labels[0]}, dataset_name=task_name, license='hands off!')