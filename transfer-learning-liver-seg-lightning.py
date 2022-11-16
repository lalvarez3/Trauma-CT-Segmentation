import glob
import os
import random
import time
from typing import Sequence, Union

import imageio
import nibabel as nib
import numpy as np
import pytorch_lightning
import torch
from monai.apps import load_from_mmar
from monai.apps.mmars import RemoteMMARKeys
from monai.config import print_config
from monai.data import CacheDataset, Dataset, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.networks.utils import copy_model_state
from monai.optimizers import generate_param_groups
from monai.transforms import (
    AddChanneld,
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureType,
    KeepLargestConnectedComponent,
    LoadImaged,
    Orientationd,
    Rand3DElasticd,
    RandCropByLabelClassesd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform
from monai.utils import InterpolateMode, set_determinism
from monai.visualize import blend_images
from pl_bolts.callbacks import PrintTableMetricsCallback
from pytorch_lightning.callbacks import LearningRateMonitor
from skimage.metrics import adapted_rand_error

import wandb


print_config()

InterpolateModeSequence = Union[
    Sequence[Union[InterpolateMode, str]], InterpolateMode, str
]


class RemoveDicts(MapTransform, InvertibleTransform):
    """Remove the specified keys from the input data"""

    def __init__(self, keys, allow_missing_keys=False, verbose=False):
        super().__init__(keys, allow_missing_keys)
        self.verbose = verbose

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            self.push_transform(d, key)
        # print(d["image_meta_dict"]["filename_or_obj"])
        a = {
            "image": d["image"],
            "label": d["label"],
            "path": d["image_meta_dict"]["filename_or_obj"],
        }
        if self.verbose:
            print(a["path"])
        d = a
        return d


SEED = 0  # Set seed for reproducibility
PRETRAINED = False  # Set to True to use pretrained model
TRANSFER_LEARNING = False  # Set to True to use transfer learning
N_WORKERS_LOADER = 0  # Set to 0 to use main process for data loading
N_WORKERS_CACHE = 0  # Set to 0 to use main process for caching
CACHE_RATE = 0  # Set to 0 to disable caching
BS = 16  # Batch size
MAX_EPOCHS = 500  # Maximum number of epochs
PATCH_SIZE = (128, 128, 128)  # Patch size
HOME = "/mnt/chansey"  # Home directory
IMG_SIZE = (128, 128, 128)  # Image size
VAL_SIZE = (256, 256, 256)  # Validation size
SAVE_PATH = os.path.join(
    HOME, "lauraalvarez", "traumaAI", "Liver_Segmentation", "lightning_logs"
)

run_idx = len(
    os.listdir(
        os.path.join(HOME, "lauraalvarez", "traumaAI", "Liver_Segmentation", "wandb")
    )
) # Get the run index for logging purposes

RUN_NAME = f"Focal_loss_L1_Final_new_spacing_val_loss_No_pretrain" # Set the run name for logging purposes
pytorch_lightning.seed_everything(SEED) # Set seed for reproducibility for everything


directory = os.environ.get("MONAI_DATA_DIRECTORY") # Get the MONAI_DATA_DIRECTORY environment variable
root_dir = os.path.join(
    HOME, "lauraalvarez", "traumaAI", "Liver_Segmentation", "spleen_data"
) # Set the root directory for the data

class Net(pytorch_lightning.LightningModule):
    """UNet model for liver segmentation using MONAI implementation"""
    def __init__(self, train_img_size, val_img_size, n_output):
        super().__init__()
        self.train_img_size = train_img_size
        self.val_img_size = val_img_size
        self.n_output = n_output
        if PRETRAINED:
            print("Using a pretrained model.")
            unet_model = load_from_mmar(
                item=mmar[RemoteMMARKeys.NAME],
                mmar_dir=root_dir,
                # map_location=device,
                pretrained=True,
            ) # Load the pretrained model from MMAR. This model is trained on Liver Cancer Segmentation
            self._model = unet_model # Set the model
            # copy all the pretrained weights except for variables whose name matches "model.0.conv.unit0"
            if TRANSFER_LEARNING:
                pretrained_dict, updated_keys, unchanged_keys = copy_model_state(
                    self._model,
                    unet_model,
                ) # Copy the pretrained weights to the new model

                self._model.load_state_dict(pretrained_dict) # Load the pretrained weights

                # stop gradients for the pretrained weights
                for x in self._model.named_parameters():
                    if x[0] in updated_keys:
                        x[1].requires_grad = True
                
                params = generate_param_groups(
                    network=self._model,
                    layer_matches=[lambda x: x[0] in updated_keys],
                    match_types=["filter"],
                    lr_values=[1e-4],
                    include_others=False,
                ) # Generate the parameter groups for the optimizer
                
                self.params = (
                    params  # Add if using CRF folder + list(self.crf.parameters())
                ) # Set the parameter groups for the optimizer

        else:
            # Create a UNet model not trained
            self._model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=3,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH,
            )

        # Create a loss function
        self.loss_function = DiceFocalLoss(
            to_onehot_y=True,
            softmax=True,
            jaccard=False,
            focal_weight=torch.FloatTensor([0.3, 0.5, 0.7]),
        )
        
        self.post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=3)]) # Post processing for the predictions
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=3)]) # Post processing for the labels
        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch") # Initialize a dice metric for validation
        self.train_dice_metric = DiceMetric(
            include_background=False, reduction="mean_batch"
        ) # Initialize a dice metric for training
        self.best_val_dice = 0 # Initialize the best validation dice score
        self.best_val_epoch = 0  # Initialize the best validation epoch    

        self.save_hyperparameters()  # save hyperparameters

    def prepare_data(self):
        """ Prepare the data for training and validation creating the datasets and dataloaders"""
        # set up the correct data path
        train_images = sorted(
            glob.glob(
                os.path.join(
                    HOME,
                    "lauraalvarez",
                    "nnunet",
                    "nnUNet_raw_data",
                    "Task510_LiverTraumaDGX",
                    "imagesTr",
                    "*.nii.gz",
                )
            )
        ) # Get the training images

        train_labels = [img.replace("imagesTr", "labelsTr") for img in train_images] 
        train_labels = [img.replace("_0000", "") for img in train_labels] # Get the labels for the training images

        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(train_images, train_labels)
        ] # Create a dictionary with the image and label paths

        test_images = sorted(
            glob.glob(
                os.path.join(
                    HOME,
                    "lauraalvarez",
                    "nnunet",
                    "nnUNet_raw_data",
                    "Task510_LiverTraumaDGX",
                    "imagesTs",
                    "*.nii.gz",
                )
            )
        ) # Get the test images

        test_labels = [img.replace("imagesTs", "labelsTs") for img in test_images]

        test_labels = [img.replace("_0000", "") for img in test_labels] # Get the labels for the test images

        data_dicts_test = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(test_images, test_labels)
        ] # Create a dictionary with the image and label paths

        random.shuffle(data_dicts) # Shuffle the training data

        validation_lim = int(len(data_dicts) * 0.9) # Get the limit for the validation data
        train_files, val_files, test_files = (
            data_dicts[:validation_lim],
            data_dicts[validation_lim:-1],
            data_dicts_test[:5] + data_dicts_test[6:],
        ) # Split the data into training, validation and test

        print("validation files", val_files)
        print("len(train_files)", len(train_files))
        print("len(validation files)", len(val_files))

        # set deterministic training for reproducibility
        set_determinism(seed=SEED)


        # Computed for the randomCropByLabel transformation based on outputs
        if self.n_output == 3:
            ratios = [1, 1, 2]
        else:
            ratios = [1, 1]
        
        # Create the training dataset
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                RemoveDicts(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                # Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2),
                    mode=("bilinear", "nearest"),
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                ToTensord(keys=["image", "label"]),
                RandCropByLabelClassesd(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=PATCH_SIZE,
                    ratios=ratios,
                    num_classes=self.n_output,
                    num_samples=2,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.40,
                ),
                Rand3DElasticd(
                    keys=["image", "label"],
                    prob=0.10,
                    sigma_range=(5, 10),
                    magnitude_range=(100, 250),
                    device=torch.device("cuda:0"),
                ),
            ]
        ) # Define the training transforms

        
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                RemoveDicts(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                # Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2),
                    mode=("bilinear", "nearest"),
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                ToTensord(keys=["image", "label"]),
            ]
        ) # define the validation transforms

        pred_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                # RemoveDicts(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                ToTensord(keys=["image", "label"]),
            ]
        ) # define the prediction transforms

        self.train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=0.4,
        ) # Create the training dataset

        self.val_ds = Dataset(data=val_files, transform=val_transforms) # Create the validation dataset

        self.test_ds = Dataset( 
            data=test_files,
            transform=pred_transforms,
        ) # Create the test dataset

    def configure_optimizers(self):
        """ Initialize the optimizer """
        optimizer = torch.optim.AdamW(self._model.parameters(), 1e-2, weight_decay=1e-4) # Define the optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", verbose=True, factor=0.05, patience=20, min_lr=1e-7
        ) # Define the scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
            "interval": "epoch",
        }

    def train_dataloader(self):
        """ Initialize the training dataloader """
        train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=BS,
            shuffle=True,
            num_workers=N_WORKERS_LOADER,
            collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        """ Initialize the validation dataloader """
        val_loader = torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=N_WORKERS_LOADER,
        )

        return val_loader

    def predict_dataloader(self):
        """ Initialize the prediction dataloader """
        predict_dataloader = torch.utils.data.DataLoader(
            self.test_ds, batch_size=1, shuffle=False, num_workers=2
        )
        return predict_dataloader

    def forward(self, x):
        """ Forward pass """
        return self._model(x)

    def training_step(self, batch, batch_idx):
        """ Training step
        
        Args:
            batch (dict): A batch of data from the training dataloader
            batch_idx (int): The index of the batch
         """
        images, labels = batch["image"], batch["label"] # Get the images and labels from the batch
        output = self.forward(images) # Forward pass get predictions
        outputs = [self.post_pred(i) for i in decollate_batch(output)] # Post process the predictions
        labels_1 = [self.post_label(i) for i in decollate_batch(labels)] # Post process the labels
        self.train_dice_metric(y_pred=outputs, y=labels_1) # Calculate the dice score
        loss = self.loss_function(output, labels) # Calculate the loss
        tensorboard_logs = {"batch_train_loss": loss.item()} # Log the loss
        return {"loss": loss, "log": tensorboard_logs} # Return the loss and logs

    def training_epoch_end(self, outputs):
        """ Training epoch end

        Args:
            outputs (list): A list of outputs from the training step
        """

        dice_liver, dice_injure = self.train_dice_metric.aggregate() # Aggregate the dice score
        avg_loss = torch.stack([x["loss"] for x in outputs if x != 0]).mean() # Calculate the average loss
        self.log("train_loss", avg_loss, prog_bar=True) # Log the average loss
        self.log("train_dice_liver", dice_liver, prog_bar=True) # Log the dice score
        self.log("train_dice_injury", dice_injure, prog_bar=True) # Log the dice score
        self.train_dice_metric.reset() # Reset the dice score

    def validation_step(self, batch, batch_idx):
        """ Validation step

        Args:
            batch (dict): A batch of data from the validation dataloader
            batch_idx (int): The index of the batch
        """

        images, labels = batch["image"], batch["label"] # Get the images and labels from the batch
        filenames = batch["path"] # Get the filenames   
        post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=3)]) # Post process the predictions
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=3)]) # Post process the labels

        roi_size = PATCH_SIZE # Define the patch size
        sw_batch_size = 4 # Define the sliding window batch size
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward
        ) # Perform sliding window inference

        with torch.cuda.amp.autocast(): # Perform mixed precision inference (Memory efficient)
            loss = self.loss_function(outputs, labels) # Calculate the loss
            predicition = {
                "output": torch.nn.Softmax()(outputs),
                "image": images,
                "label": labels,
                "filename": filenames,
            } # Create a dictionary of the predictions
            outputs = [post_pred(i) for i in decollate_batch(outputs)] # Post process the predictions
            labels = [post_label(i) for i in decollate_batch(labels)] # Post process the labels
            self.dice_metric(y_pred=outputs, y=labels) # Calculate the dice score

        return {
            "val_dice_metric": self.dice_metric,
            "val_number": len(outputs),
            "prediction": predicition,
            "val_loss": loss,
        }

    def validation_epoch_end(self, outputs):
        """ Validation epoch end
        
        Args:
            outputs (list): A list of outputs from the validation step
        """

        mean_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean() # Calculate the average loss
        post_pred_dice = Compose(
            [
                EnsureType(),
                AsDiscrete(argmax=True, to_onehot=3),
                KeepLargestConnectedComponent([1, 2], is_onehot=True, independent=True),
            ]
        ) # Post process the predictions
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=3)]) # Post process the labels
        dice_liver, dice_injury = self.dice_metric.aggregate() # Aggregate the dice score
        self.dice_metric.reset() # Reset the dice score
        tensorboard_logs = {
            "val_dice_metric": dice_injury,
        } # Log the dice score
        predictions = [x["prediction"] for x in outputs] # Get the predictions

        # Save the predictions every 25 epochs if the dice score has improved by 0.05 or if it is the last epoch 
        if (
            self.current_epoch % 25 == 0 
            or dice_injury - self.best_val_dice > 0.05
            or self.current_epoch == self.trainer.max_epochs - 1
        ): 
            test_dt = wandb.Table(
                columns=[
                    "epoch",
                    "filename",
                    "combined output",
                    "dice_value_liver",
                    "dice_value_injure",
                    "ground_truth",
                    "class predicted",
                ]
            )
            
            # Loop through the predictions and get gifs
            for i, prediction in enumerate(predictions):
                # Get the predictions, labels, filenames and images
                filename = os.path.basename(prediction["filename"][0]) # Get the filename
                output_one = [
                    post_pred_dice(i) for i in decollate_batch(prediction["output"])
                ] # Post process the predictions
                label_one = [
                    post_label(i) for i in decollate_batch(prediction["label"])
                ] # Post process the labels
                self.dice_metric(y_pred=output_one, y=label_one) # Calculate the dice score
                dice_value_liver, dice_value_injure = self.dice_metric.aggregate() # Aggregate the dice score
                self.dice_metric.reset() # Reset the dice score
                class_predicted, _, ground_truth = get_classification_info(prediction) # Get the classification info
                blended = make_gif(prediction, filename=i) # Create a gif of the predictions
                row = [
                    self.current_epoch,
                    filename,
                    wandb.Image(blended),
                    dice_value_liver,
                    dice_value_injure,
                    int(ground_truth[0]),
                    class_predicted,
                ] # Create a row for the table
                test_dt.add_data(*row) # Add the row to the table

            self.logger.experiment.log({f"SUMMARY_EPOCH_{self.current_epoch}": test_dt})

        # print the best model parameters
        if dice_injury > self.best_val_dice:
            self.best_val_dice = dice_injury.item()
            self.best_val_epoch = self.current_epoch
        lnp.lnp(
            f"current epoch: {self.current_epoch}, "
            f"current liver dice: {dice_liver:.4f}, "
            f"current injure  dice: {dice_injury:.4f} "
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.log("val_dice_metric_liver", dice_liver.item(), prog_bar=True)
        self.log("val_dice_metric_injury", dice_injury.item(), prog_bar=True)
        self.log("val_loss", mean_val_loss, prog_bar=True)
        return {"log": tensorboard_logs}

    def predict_step(self, batch, batch_idx):
        """ Predict step

        Args:
            batch (dict): A dictionary containing the input data
            batch_idx (int): The index of the current batch
        """

        print("predicting...")
        images, labels = batch["image"], batch["label"] # Get the images and labels

        post_pred = Compose(
            [
                EnsureType(),
                AsDiscrete(argmax=True, to_onehot=3),
                KeepLargestConnectedComponent(
                    [1, 2], is_onehot=True, independent=False
                ),
            ]
        ) # Post process the predictions

        post_pred_2 = Compose([EnsureType(), AsDiscrete(argmax=True)]) # Post process the predictions
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=3)]) # Post process the labels
        roi_size = PATCH_SIZE # Set the roi size
        sw_batch_size = 4 # Set the sliding window batch size
        output = sliding_window_inference(images, roi_size, sw_batch_size, self.forward) # Perform sliding window inference

        o = torch.nn.Softmax()(output) # Apply softmax to the output

        outputs = [post_pred(i.cpu()) for i in decollate_batch(output)] # Post process the predictions

        labels = [
            post_label(torch.unsqueeze(i, 0)).squeeze().cpu()
            for i in decollate_batch(labels)
        ] # Post process the labels

        dice_metric = DiceMetric(include_background=False, reduction="mean_batch") # Create a dice metric
        dice_metric(y_pred=outputs, y=labels) # Calculate the dice score
        dice_score_organ, dice_score_injury = dice_metric.aggregate() # Aggregate the dice score
        _, precision_organ, recall_organ = adapted_rand_error(
            labels[0][1, :, :, :].numpy().astype(int),
            outputs[0][1, :, :, :].numpy().astype(int),
        ) # Calculate the precision and recall for the liver
        _, precision_injury, recall_injury = adapted_rand_error(
            labels[0][2, :, :, :].numpy().astype(int),
            outputs[0][2, :, :, :].numpy().astype(int),
        ) # Calculate the precision and recall for the injury

        dict_data = {
            "dice_score_organ": dice_score_organ.numpy(),
            "dice_score_injury": dice_score_injury.numpy(),
            "precision_organ": precision_organ,
            "recall_organ": recall_organ,
            "precision_injury": precision_injury,
            "recall_injury": recall_injury,
        } # Create a dictionary containing the metrics

        outputs = [post_pred_2(i.cpu()) for i in decollate_batch(output)] # Post process the predictions
        

        save_filename = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[
            -1
        ]  # Get the filename
        save_filename = save_filename.replace("_0000.nii.gz", ".nii.gz") # Replace the filename

        output_directory = os.path.join(
            HOME,
            "lauraalvarez",
            "nnunet",
            "nnUNet_raw_data",
            "Task510_LiverTraumaDGX",
            "out_unet",
        ) # Set the output directory

        print(
            f"writing image to {os.path.join(output_directory, save_filename)}\n {dict_data}"
        )

        predicition = {
            "output": o,
            "image": batch["image"],
            "label": batch["label"],
        } # Create a dictionary containing the predictions

        affine = batch["image_meta_dict"]["original_affine"][0].cpu().numpy()   # Get the affine
        o = torch.nn.Softmax()(output).cpu().numpy() # Apply softmax to the output
        o = np.argmax(o, axis=1).astype(np.uint8)[0] # Get the argmax of the output
        nib.save(
            nib.Nifti1Image(o, affine), os.path.join(output_directory, save_filename)
        ) # Save the prediction


        blended = make_gif(predicition, filename=batch_idx) # Create a gif of the predictions
        print(f"saved at {blended}")
        return {"dice_metric": dice_metric}


def make_gif(prediction, filename):
    """ Creates a gif of the prediction and the label
    
    Args:
        prediction (dict): Dictionary with the prediction and the label
        filename (str): Name of the file to save the gif
    """
    def _save_gif(volume, filename):
        """ Saves a volume as a gif
        
        Args:
            volume (np.array): Volume to save
            filename (str): Name of the file to save the gif    
        
        """
        # normalize the data to 0 - 1
        volume = volume.astype(np.float64) / np.max(volume) 
        volume = 255 * volume  # Now scale by 255 
        volume = volume.astype(np.uint8)

        path_to_gif = os.path.join("gifs", f"{filename}.gif") # Set the path to save the gif
        # Save the volume as a gif
        if not os.path.exists("gifs"): 
            print("Creating gifs directory")
            os.mkdir("gifs")
        imageio.mimsave(path_to_gif, volume)
        return path_to_gif

    post_pred_blending = Compose(
        [
            EnsureType(),
            AsDiscrete(argmax=True),
            KeepLargestConnectedComponent([1, 2], is_onehot=False, independent=True),
        ]
    ) # Post process the predictions
    prediction["output"] = [
        post_pred_blending(i) for i in decollate_batch(prediction["output"])
    ] # Post process the predictions
    selected = {
        "output": prediction["output"][0],
        "image": prediction["image"][0],
        "label": prediction["label"][0],
    }   # Select the first element of the batch 

    selected = Resized(keys=["image", "label"], spatial_size=(160, 160, 160))(selected) # Resize the image and the label for memory
    selected = Resized(keys=["output"], spatial_size=(160, 160, 160))(selected) # Resize the prediction for memory

    selected = {
        "output": selected["output"],
        "image": selected["image"],
        "label": selected["label"],
    } 

    selected = Orientationd(keys=["output", "image", "label"], axcodes="LPS")(selected) # Set the orientation of the image, label and prediction

    pred = selected["output"][0].unsqueeze(0).detach().cpu().numpy() # Get the prediction
    true_label = selected["label"][0].unsqueeze(0).detach().cpu().numpy() # Get the label
    image = selected["image"][0].unsqueeze(0).cpu().numpy() # Get the image
 
    blended_true_label = blend_images(image, true_label, alpha=0.7)     # Blend the image and the label
    blended_final_true_label = torch.from_numpy(blended_true_label).permute(1, 2, 0, 3) # Permute the image and the label

    blended_prediction = blend_images(image, pred, alpha=0.7) # Blend the image and the prediction

    blended_final_prediction = torch.from_numpy(blended_prediction).permute(1, 2, 0, 3) # Permute the image and the prediction

    volume_pred = blended_final_prediction[:, :, :, :] # Get the prediction
    volume_label = blended_final_true_label[:, :, :, :] # Get the label
    volume_pred = blended_final_prediction[:, :, :, :].permute(3, 1, 0, 2).cpu() # Permute the prediction
    volume_label = (
        np.squeeze(blended_final_true_label[:, :, :, :]).permute(3, 1, 0, 2).cpu() 
    ) # Permute the label
    volume_img = torch.tensor(image).permute(3, 2, 1, 0).repeat(1, 1, 1, 3).cpu()   # Permute the image

    volume = torch.hstack((volume_img, volume_pred, volume_label)) # Stack the image, the prediction and the label

    volume_path = _save_gif(volume.numpy(), f"blended-{filename}") # Save the gif

    return volume_path


def get_classification_info(prediction):
    """ Gets the classification information from the prediction
    
    Args:
        prediction (dict): Dictionary with the prediction and the label
    """

    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=3)]) # Post process the label

    ground_truth = [
        1 if (post_label(i)[2, :, :, :].cpu() == 1).any() else 0
        for i in decollate_batch(prediction["label"])
    ] # Get the ground truth

    test = prediction["output"].cpu() # Get the prediction
    prediction_1 = torch.argmax(test, dim=1) # Get the argmax of the prediction

    class_2_mask = (prediction_1 == 2).cpu() # Get the mask of the class 2
    if class_2_mask.any(): # If there is a class 2
        prediction = torch.max(test[:, 2, :, :, :]).item() 
    else: # If there is no class 2
        prediction = np.max(np.ma.masked_array(test[:, 2, :, :, :], mask=class_2_mask))

    unique_values = torch.unique(prediction_1) # Get the unique values of the prediction
    predicted_class = 1 if 2 in unique_values else 0 # Get the predicted class

    return predicted_class, prediction, ground_truth


mmar = {
    RemoteMMARKeys.ID: "clara_pt_liver_and_tumor_ct_segmentation_1",
    RemoteMMARKeys.NAME: "clara_pt_liver_and_tumor_ct_segmentation",
    RemoteMMARKeys.FILE_TYPE: "zip",
    RemoteMMARKeys.HASH_TYPE: "md5",
    RemoteMMARKeys.HASH_VAL: None,
    RemoteMMARKeys.MODEL_FILE: os.path.join("models", "model.pt"),
    RemoteMMARKeys.CONFIG_FILE: os.path.join("config", "config_train.json"),
    RemoteMMARKeys.VERSION: 1,
}


def save_checkpoint(state, name):
    """Saves the model checkpoint to disk
    
    Args:
        state (dict): Dictionary with the model state
        name (str): Name of the checkpoint
    """
    file_path = "checkpoints/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    epoch = state["epoch"]
    save_dir = file_path + name + str(epoch)
    torch.save(state, save_dir)
    print(f"Saving checkpoint for epoch {epoch} in: {save_dir}")


def save_state_dict(state, name):
    """Saves the model state dict to disk

    Args:
        state (dict): Dictionary with the model state
        name (str): Name of the checkpoint
    """

    file_path = "checkpoints/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    save_dir = file_path + f"{name}_best"
    torch.save(state, save_dir)
    print(f"Best accuracy so far. Saving model to:{save_dir}")


class Log_and_print:
    """Class to log and print the training information"""
    def __init__(self, run_name):
        self.run_name = run_name
        self.str_log = "run_name" + "\n  \n"

    def lnp(self, tag):
        """Logs and prints the information

        Args:
            tag (str): Information to log and print
        """

        print(self.run_name, time.asctime(), tag)
        self.str_log += str(time.asctime()) + " " + str(tag) + "  \n"

lnp = Log_and_print(RUN_NAME) # Initialize the logging and printing class
lnp.lnp("Loggers start")
lnp.lnp("ts_script: " + str(time.time()))

wandb_logger = pytorch_lightning.loggers.WandbLogger(
    project="traumaIA",
    name=RUN_NAME,
    save_dir=SAVE_PATH,
) # Initialize the wandb logger

lnp.lnp("MAIN callbacks")
l_callbacks = []
cbEarlyStopping = pytorch_lightning.callbacks.early_stopping.EarlyStopping(
    monitor="val_loss", patience=500, mode="max"
) # Initialize the early stopping callback
l_callbacks.append(cbEarlyStopping) # Add the early stopping callback to the list of callbacks


checkpoint_dirpath = SAVE_PATH + "/checkpoints/" # Path to save the checkpoints
checkpoint_filename = SAVE_PATH[:-1] + "_" + RUN_NAME + "_Best" # Name of the checkpoint
lnp.lnp("checkpoint_dirpath: " + checkpoint_dirpath) 
lnp.lnp("checkpoint_filename: " + checkpoint_filename)
cbModelCheckpoint = pytorch_lightning.callbacks.ModelCheckpoint(
    monitor="val_dice_metric_injury",
    mode="max",
    dirpath=checkpoint_dirpath,
    filename=checkpoint_filename,
) # Initialize the model checkpoint callback
l_callbacks.append(cbModelCheckpoint) # Add the model checkpoint callback to the list of callbacks


checkpoint_dirpath = SAVE_PATH + "/checkpoints/"
checkpoint_filename = SAVE_PATH[:-1] + "_" + RUN_NAME + "_Last"
lnp.lnp("checkpoint_dirpath: " + checkpoint_dirpath)
lnp.lnp("checkpoint_filename: " + checkpoint_filename)
cbModelCheckpointLast = pytorch_lightning.callbacks.ModelCheckpoint(
    every_n_epochs=1,
    dirpath=checkpoint_dirpath,
    filename=checkpoint_filename,
) # Initialize the model checkpoint callback to log last epoch
l_callbacks.append(cbModelCheckpointLast) # Add the model checkpoint callback to the list of callbacks

l_callbacks.append(PrintTableMetricsCallback())
lr_monitor = LearningRateMonitor(logging_interval="epoch")
l_callbacks.append(lr_monitor)


if "__main__" == __name__:
    lnp.lnp(" Start Trainining process...")
    net = Net.load_from_checkpoint(
        os.path.join(
            HOME,
            "lauraalvarez",
            "traumaAI",
            "Liver_Segmentation",
            "lightning_log_Predict_Segmentation_UNETPRE_318_extended_Focal_loss_L1_Final_new_spacing_Best-v2.ckpt",
        )
    ) # Load the model from the checkpoint

    # set up loggers and checkpoints
    log_dir = os.path.join(root_dir, "logs")

    # initialise Lightning's trainer.
    trainer = pytorch_lightning.Trainer(
        default_root_dir=f"{SAVE_PATH}/checkpoints",
        gpus=[0],
        max_epochs=MAX_EPOCHS,
        fast_dev_run=False,
        auto_lr_find=False,
        logger=wandb_logger,
        enable_checkpointing=True,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        callbacks=l_callbacks,
        move_metrics_to_cpu=False,
        accumulate_grad_batches=16,
    )

    # train
    trainer.fit(net)


    wandb.alert(
        title="Train finished",
        text="The train has finished"
    )
    wandb.finish()
