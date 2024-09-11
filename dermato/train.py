import os
import sys
import argparse
import random
import pickle
import shutil
import yaml
from tqdm import tqdm
from glob import glob
from datetime import datetime

# 
import zarr
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import auc

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Torchvision
import torchvision.transforms as transforms
from torchvision.models import densenet121, DenseNet121_Weights


# W&B
import wandb

# Kornia 
import kornia as K

# Custom
from utils import gen_performance_metrics, compute_tp_fp_tn_fn
from stain_labeling import count_sox10_stain_blobs, count_mela_prop_section_from_ihc_dab

#---------------------------------------------------
def seed_evertyhing(seed, worker):
    """
    https://pytorch-lightning.readthedocs.io/en/1.6.5/_modules/pytorch_lightning/utilities/seed.html#seed_everything
    """
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(worker)}"

class StainTrainDataSet(Dataset):
    def __init__(self, is_df_ready=True, stain='sox10', list_sections='section_image', tile_size=256, stride_size=256, only_fg=True, df=None, yaml_file=None, split='train'):
        """
        pytorch dataset for the sox10 and melana/melpro stain dataset. If the dataframe is not ready, it will generate the dataframe and save it in ./csv_data
        Args:
            is_df_ready (bool): whether the dataframe is already created or not
            stain (str): sox10 or melana or melpro
            list_sections (list): list of section names to be used.
            tile_size (int): size of the tile
            stride_size (int): stride size of the window to generate tiles
            only_fg (bool): whether to only use foreground tiles
            yaml_file (str): path to the yaml file containing parameters for preprocessing IHC to create the labels. 
                       sox10_labeling.yaml or melana_melpro_labeling.yaml
            split (str): train, val, or test

        """
        if is_df_ready:
            self.df = df
        else:
            self.stain = stain
            # list of paths to the H&E, IHC, and DAB zarr files. ** Hard coded here **
            self.list_dab_sections_path = [glob(f'/srv/nas/mk1/users/mtada/paired_aligned_images/*/DAB/{section_name}.zarr')[0] for section_name in list_sections]
            self.list_ihc_sections_path = [glob(f'/srv/nas/mk1/users/mtada/paired_aligned_images/*/IHC/{section_name}.zarr')[0] for section_name in list_sections]
            self.list_he_sections_path = [glob(f'/srv/nas/mk1/users/mtada/paired_aligned_images/*/HE/{section_name}.zarr')[0] for section_name in list_sections]
            self._generate_df_labels()
        self.tile_size = tile_size
        self.stride_size = stride_size
        self.only_fg = only_fg
        self.split   = split
        self.yaml_file = yaml_file
        
        # get the current time for saving the csv file
        self.current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # compute the baseline, the ratio of positive to negative samples
        self._compute_baseline()

    def _generate_df_labels(self):
        """
        Create a dataframe containing the labels for each tile and save it in ./csv_data
        """
        # load the label config yaml file
        with open(self.yaml_file, 'r') as f:
            map_section2metadata = yaml.load(f, Loader=yaml.SafeLoader)
        # dict of section name to metadata
        map_section2metadata = map_section2metadata['sections']

        # list of rows to be saved in the csv file
        list_rows = []
        for he_path, ihc_path, dab_path in zip(self.list_he_sections_path, self.list_ihc_sections_path, self.list_dab_sections_path):
            section_name = os.path.basename(ihc_path).split('.')[0]
            print(f'Processing section: {section_name} ...')
            # get the metadata for this section
            meta_data = map_section2metadata[section_name]
            stain_color, dab_thred, is_artifact, min_size, first_kernel, second_kernel, threshold = meta_data['stain_color'], meta_data['dab_thred'], meta_data['artifact'], meta_data['min_artifact_size'], meta_data['first_kernel'], meta_data['second_kernel'], meta_data['threshold']
            
            # open the zarr files
            ihc = zarr.open(ihc_path, 'r')
            dab = zarr.open(dab_path, 'r')

            section_h, section_w = dab.shape

            # loop through the tiles
            for y in range(0, section_h, self.stride_size):
                for x in range(0, section_w, self.stride_size):
                    ihc_tile = ihc[y:y+self.tile_size, x:x+self.tile_size]
                    dab_tile = dab[y:y+self.tile_size, x:x+self.tile_size]

                    # Flag whether the tile is foreground or background
                    if self.only_fg and np.sum(ihc_tile) == 0:
                        continue # skip background tiles
                    
                    if self.stain == 'sox10':
                        # count number of SOX10 stain blobs
                        # Dab stain image is already preprocessed to remove artifacts such as dark green ink and blood
                        num_sox10_blobs = count_sox10_stain_blobs(ihc_tile, dab_tile, stain_color, dab_thred, first_kernel, second_kernel)
                        label = 1 if num_sox10_blobs >= threshold else 0
                        stain_quant = num_sox10_blobs
                    else: # melana or melpro
                        mela_prop, _ = count_mela_prop_section_from_ihc_dab(
                            ihc_rgb_tile=ihc_tile, dab_tile=dab_tile, 
                            stain_color='red', dab_thred=dab_thred, 
                            first_kernel=(None, None), second_kernel=(None, None)
                        )
                        label = 1 if mela_prop > threshold else 0
                        stain_quant = mela_prop


                    # store, H&E, IHC, xy coordinate, foreground flag,  for this tile
                    tile_row = [he_path, ihc_path, section_name, y, x, stain_quant, label, threshold]
                    list_rows.append(tile_row)

        # create the dataframe from list of lists
        self.df = pd.DataFrame(list_rows, columns=['he', 'ihc', 'section_name', 'y', 'x', 'stain_quant', 'label', 'threshold'])
        # the output directory is hard coded here
        self.df.to_csv(f'csv_data/{self.current_time}_{self.split}.csv', index=False) # save the csv file for future use.

    def _compute_baseline(self):
        """
        Compute the baseline accuracy
        """
        # sum the number of positive and negative samples   
        num_positives = np.sum(self.df['label']) # sum of the labels = number of positives since label=1
        num_negatives = self.df.shape[0] - num_positives
        # compute the ratio of positive to negative samples
        self.ratio_positives = round(num_positives / self.df.shape[0] * 100, 2)
        self.ratio_negatives = round(num_negatives / self.df.shape[0] * 100, 2)
        print(f'{self.split} Baseline accuracy: {self.ratio_positives}% positives, {self.ratio_negatives}% negatives')
        

    def __len__(self):
        # return the number of tiles, size of dataframe
        return self.df.shape[0]

    def __getitem__(self, idx):
        """
        
        Args:
            idx (int): Index to iterate through the dataframe
        """
        # get the paths to the H&E and IHC images
        he_section_path = self.df.iloc[idx]['he']
        ihc_section_path = self.df.iloc[idx]['ihc']

        # get the y, x coordinates of the tile
        y, x = self.df.iloc[idx]['y'], self.df.iloc[idx]['x']
        
        # get the label
        label = self.df.iloc[idx]['label']

        # TODO instead of loading the entire section, load only the tile by chunk for speed up
        # temporarily load the /scratch
        if os.path.splitext(he_section_path)[1] == '.zarr':
            he_section = zarr.open(he_section_path, 'r')  #[C,H,W] #
            # crop the tile from the section
            he_tile = he_section[y:y+self.tile_size, x:x+self.tile_size]
            h, w, c = he_tile.shape

            # pad the value 255 to bottom and right sides of the tile if it is smaller than the tile size
            if (h < self.tile_size) or (w < self.tile_size):
                he_tile = np.pad(he_tile, ((0, self.tile_size-h), (0, self.tile_size-w), (0,0)), 'constant', constant_values=(255.))
        
        elif os.path.splitext(he_section_path)[1] == '.jpg':
            he_tile = np.array(Image.open(he_section_path))


        # [H,W,C] -> [C,H,W] torch.uint8 then convert to float32 tensor in range [0,1]
        he_tile = K.image_to_tensor(he_tile, keepdim=True).float() / 255.0
        label = torch.tensor(label, dtype=torch.float)

        return he_tile, label

def initialize_pretrained_model():
    """
    Initializes a pre-trained DenseNet121 model with Imagenet1K_1V.

    Returns:
        model (torch.nn.Module): The initialized DenseNet121 model with a modified classifier.
    """

    weights = DenseNet121_Weights.IMAGENET1K_V1
    # generate the model 
    model = densenet121(weights=weights)
    model.classifier = nn.Linear(in_features=1024, out_features=1, bias=True)
    return model

def compute_mean_std_train(splitID):
    """
    Compute the mean and standard deviation of the training data for a specific split and save it to a file as pkl. 

    Args:
        splitID (int): The ID of the split.
    Returns:
        tuple: A tuple containing the mean and standard deviation of the training data.

    """
    # hard coded path 
    mean_std_file = f"/srv/home/mtada/dermato-paper/Sox10/5folds_csv/split{splitID}_w_tcga_outof_5folds_mean_std.pkl"
    if os.path.isfile(mean_std_file):
        with open(mean_std_file, "rb") as f:
            (train_mean, train_std) = pickle.load(f)
        print(f'Loaded mean and std {train_mean}, {train_std} from {mean_std_file} ...')

    return train_mean, train_std

def get_WRS(y_train, neg_weight:float = 0.5, pos_weight:float = 0.5):
    """
     Create a WeightedRandomSampler object for training data.
    Args:
        y_train (array-like): Vector representing labels of 0s and 1s for training set.
        neg_weight (float): Float weight representing likelihood that a given sample will be negative. Defaults to 0.5.
        pos_weight (float): Float weight representing likelihood that a given sample will be positive. Defaults to 0.5.
    Returns:
        WeightedRandomSampler: Weighted random sampler object for training data.
    """
    try:
        y_train = y_train.copy()
    except:
        y_train = y_train.clone()
        
    assert pos_weight + neg_weight == 1
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    
    y_train[np.where(y_train == 0)[0]] = neg_weight * class_sample_count[1]
    y_train[np.where(y_train == 1)[0]] = pos_weight * class_sample_count[0]
    
    return WeightedRandomSampler(torch.tensor(y_train).type('torch.DoubleTensor'), len(y_train), replacement = True)


def build_train_data(splitID, config):
    """
    Builds and returns a train DataLoader object for training data with an option to use WeightedRandomSampler.
    Args:
        splitID (int): The ID of the split.
        config (dict): A dictionary containing configuration parameters.
    Returns:
        DataLoader: A DataLoader object for training data.
    """
    stain = config['stain']
    # hard coded path
    df = pd.read_csv(f'/srv/home/mtada/dermato-paper/{stain}/5folds_csv/split{splitID}_w_tcga_train_outof_5folds.csv')

    batch_size = config['batch_size']
    num_workers = config['num_workers']
    train_ds = StainTrainDataSet(
        df=df,
        stride_size=128,
        tile_size=256,
        only_fg=True,
        split='train',
        )
    wrs = True
    if wrs:
        wrs_neg, wrs_pos = 0.5, 0.5
        WRS = get_WRS(train_ds.df['label'], neg_weight=wrs_neg , pos_weight = wrs_pos)
        shuffle = False
    else:
        WRS = None
        shuffle = True

    train_dl = DataLoader(
        train_ds, 
        batch_size=batch_size,
        num_workers=num_workers, 
        shuffle=shuffle, #mutually exclusive with sampler when WRS is on
        pin_memory=True, 
        drop_last=False,
        sampler = WRS,
        )
    return train_dl

def build_val_data(splitID, config):
    """
    Builds and returns a validation DataLoader object for training data.
    Args:
        splitID (int): The ID of the split.
        config (dict): A dictionary containing configuration parameters.
    Returns:
        DataLoader: A DataLoader object for validation data.
    """
    stain = config['stain']
    # hard coded path
    df = pd.read_csv(f'/srv/home/mtada/dermato-paper/{stain}/5folds_csv/split{splitID}_w_tcga_val_outof_5folds.csv')
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    val_ds = StainTrainDataSet(
            df=df,
            stride_size=256,
            tile_size=256,
            only_fg=True,
            split='val',
            )

    val_dl = DataLoader(
        val_ds, 
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False, 
        pin_memory=True, 
        drop_last=False)
    return val_dl


augmentation = K.augmentation.AugmentationSequential(
        K.augmentation.RandomHorizontalFlip(p=0.75),
        K.augmentation.RandomVerticalFlip(p=0.75),
        K.augmentation.RandomElasticTransform(p=0.5),
        K.augmentation.RandomGaussianBlur(p=0.5,
                                         kernel_size=(3, 3),
                                         sigma=(0.1, 4.0)),
        K.augmentation.RandomAffine(p=0.5,
                                    degrees=(45., 45.),
                                    scale=(0.75, 1.25),
                                    translate=(0.25, 0.25),
                                    shear=None
                                    ),
        K.augmentation.ColorJitter(p=0.75,
                                  brightness=(0.65, 1.35), 
                                  contrast=(0.5, 1.5), 
                                  saturation=(0.75, 1.75), 
                                  hue=(-0.04, 0.06)
        )
    )

def train_one_epoch(model, loader, optimizer, device, train_mean, train_std):
    """
    Trains the model for one epoch using the given data loader and optimizer.
    Args:
        model (nn.Module): The model to be trained.
        loader (DataLoader): The data loader containing the training data.
        optimizer (Optimizer): The optimizer used for updating the model's parameters.
        device (str): The device to be used for training (e.g., 'cpu', 'cuda').
        train_mean (float or list): The mean value(s) used for input normalization.
        train_std (float or list): The standard deviation value(s) used for input normalization.
    Returns:
        tuple: A tuple containing the following metrics:
            - loss_over_one_epoch (float): The average loss over the entire training set.
            - precision_over_one_epoch (float): The precision over the entire training set.
            - recall_over_one_epoch (float): The recall over the entire training set.
            - f1_over_one_epoch (float): The F1 score over the entire training set.
    """

    running_total_loss_over_one_epoch = 0.0
    running_total_tp, running_total_fp, running_total_tn, running_total_fn = 0, 0, 0, 0

    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    #loader_tqdm = tqdm(loader, total=int(len(loader)))
    for batch_idx, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # augmentation 
        inputs = augmentation(inputs)

        # mean and std normalization 
        inputs = K.enhance.normalize(data=inputs, mean=train_mean, std=train_std)

        # float32
        logits = model(inputs)
        # [B,1]->[B]
        logits = torch.squeeze(logits, dim=1)

        # average the minibatch loss
        avg_one_batch_loss = criterion(logits, labels)

        # backward
        avg_one_batch_loss.backward()
        # update the weights
        optimizer.step()
        # zero the gradients
        optimizer.zero_grad()

        # total minibatch loss
        total_one_batch_loss = avg_one_batch_loss.item() * logits.shape[0]
        # add the total loss to the running total loss
        running_total_loss_over_one_epoch += total_one_batch_loss

        # convert logits to probabilities
        prob_outputs = torch.sigmoid(logits.cpu())
        binary_outputs = prob_outputs > 0.5
        
        # compute the tp, fp, tn, fn for minibatch
        tp, fp, tn, fn = compute_tp_fp_tn_fn(binary_outputs.numpy().astype(np.uint8), labels.cpu().numpy().astype(np.uint8))
        running_total_tp += tp
        running_total_fp += fp
        running_total_tn += tn
        running_total_fn += fn

    # compute the precision, recall, and f1
    precision_over_one_epoch = running_total_tp / (running_total_tp + running_total_fp)
    recall_over_one_epoch = running_total_tp / (running_total_tp + running_total_fn)
    f1_over_one_epoch = 2 * (precision_over_one_epoch * recall_over_one_epoch) / (precision_over_one_epoch + recall_over_one_epoch)

    # compute the average loss over the entire training set
    loss_over_one_epoch = running_total_loss_over_one_epoch / len(loader.dataset)
    # print(f'Training')
    # print(f"    train loss:{round(loss_over_one_epoch,3)} | train Precision:{round(precision_over_one_epoch,3)} | train Recall:{round(recall_over_one_epoch,3)} | train F1:{round(f1_over_one_epoch,3)}")

    return loss_over_one_epoch, precision_over_one_epoch, recall_over_one_epoch, f1_over_one_epoch

# Evaluation
def save_checkpoint(state, filename='checkpoint.pt'):
    """
    Save the general checkpoint with the torch.save. 

    Args:
        state (dict): the state dictionary to save
        filename (str): the name of the file to save the checkpoint
    """
    torch.save(state, filename)

@torch.no_grad()
def evalute_validation(model, val_dl, device, train_mean, train_std, epoch):
    """
    Evalute the model on the validation set. Log the loss, precision, recall, and f1 on W&B.

    Args:
        model:      the model
        val_dl:     the validation dataloader
        criterion:  the loss function
        device:     the device to run the model on
        epoch:      the current epoch


    Returns:
        loss_over_one_epoch:     the average loss on the validation set
    """
    model.eval() # turn off dropout and batchnorm
    #val_dl_tqdm = tqdm(val_dl, total=int(len(val_dl)))

    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    running_total_val_loss = 0.0
    running_total_tp, running_total_fp, running_total_tn, running_total_fn = 0, 0, 0, 0

    all_labels = []
    all_prob_outputts = []

    for batch_idx, data in enumerate(val_dl):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        inputs = K.enhance.normalize(data=inputs, mean=train_mean, std=train_std)

        # forward pass. the output of the list linear layer
        logits = model(inputs)
        logits = torch.squeeze(logits, dim=1)
        # compute average the minibatch loss
        avg_one_batch_loss = criterion(logits, labels)

        # compute the total loss (sum of the minibatch)
        total_one_batch_loss = avg_one_batch_loss.item() * logits.shape[0]
        # add the total loss to the running total loss
        running_total_val_loss += total_one_batch_loss
        
        # Simgoid the logits to get the probability of the positive class
        prob_outputs = torch.sigmoid(logits.cpu())
        # convert the probability to binary class. 0.5 should be the same as argmax. 
        # it can be changed to 0.6 or 0.4 to be more strict or lenient
        binary_outputs = prob_outputs > 0.5
        # compute the tp, fp, tn, fn for minibatch
        tp, fp, tn, fn = compute_tp_fp_tn_fn(binary_outputs.numpy().astype(np.uint8), labels.cpu().numpy().astype(np.uint8))
        # add the tp, fp, tn, fn to the running total
        running_total_tp += tp
        running_total_fp += fp
        running_total_tn += tn
        running_total_fn += fn

        all_labels.extend(labels.cpu())
        all_prob_outputts.extend(prob_outputs)

    # compute the precision, recall, and f1
    precision_over_one_epoch = running_total_tp / (running_total_tp + running_total_fp)
    recall_over_one_epoch = running_total_tp / (running_total_tp + running_total_fn)
    f1_over_one_epoch = 2 * (precision_over_one_epoch * recall_over_one_epoch) / (precision_over_one_epoch + recall_over_one_epoch)

    # compute the average loss over the entire validation set
    loss_over_one_epoch = running_total_val_loss / len(val_dl.dataset)

    prevalence, all_TPRs, all_FPRs, all_PPVs = gen_performance_metrics(labels=all_labels, probs=all_prob_outputts, outfn=None, step_size=0.01)
    auroc = auc(x=all_FPRs, y=all_TPRs)
    auprc = auc(x=all_TPRs, y=all_PPVs)
    # print(f'Validation')
    # print(f"    val loss:{round(loss_over_one_epoch,3)} | val AUROC: {round(auroc,3)} | val AUPRC: {round(auprc,3)} | val Precision:{round(precision_over_one_epoch,3)} | val Recall:{round(recall_over_one_epoch,3)} | val F1:{round(f1_over_one_epoch,3)}")
    # loss_over_one_epoch, precision_over_one_epoch, recall_over_one_epoch, f1_over_one_epoch
    return loss_over_one_epoch, precision_over_one_epoch, recall_over_one_epoch, f1_over_one_epoch, auprc, auroc


def train_w_early_stopping(model, train_dl, val_dl, optimizer, device, train_mean, train_std, config, splitID, model_weights_dir):
    """
    Trains a model with early stopping based on validation performance.
    Args:
        model (torch.nn.Module): The model to be trained.
        train_dl (torch.utils.data.DataLoader): The data loader for the training set.
        val_dl (torch.utils.data.DataLoader): The data loader for the validation set.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        device (torch.device): The device to be used for training.
        train_mean (float): The mean value used for data normalization during training.
        train_std (float): The standard deviation value used for data normalization during training.
        config (dict): A dictionary containing configuration parameters.
        splitID (int): The ID of the current split.
        model_weights_dir (str): The directory to save the model weights.
    Returns:
        float: The best validation area under the precision-recall curve (val_auprc).
    """
    early_stopping_patients_counter = 0
    max_val_auprc = 0.0

    max_epochs = config['max_epochs']
    patients = config['patients']

    for epoch in range(max_epochs):
        model.train()
        # train one epoch
        loss_over_one_epoch, precision_over_one_epoch, recall_over_one_epoch, f1_over_one_epoch = train_one_epoch(model, train_dl, optimizer, device, train_mean, train_std)

        wandb.log({
            f"ID{splitID}_train_loss":loss_over_one_epoch,
            f"ID{splitID}_train_precision": precision_over_one_epoch,
            f'ID{splitID}_train_recall': recall_over_one_epoch,
            f'ID{splitID}_train_f1': f1_over_one_epoch,
            f"epoch": epoch+1}
            )

        # evalute on the validation set
        val_loss, val_precision, val_recall, val_f1, val_auprc, val_auroc = evalute_validation(model, val_dl, device, train_mean, train_std, epoch)

        wandb.log({
            f"ID{splitID}_val_loss":val_loss,
            f"ID{splitID}_val_precision": val_precision,
            f'ID{splitID}_val_recall': val_recall,
            f'ID{splitID}_val_f1': val_f1,
            f'ID{splitID}_val_auprc': val_auprc,
            f'ID{splitID}_val_auroc': val_auroc,
            f"epoch": epoch+1}
            )

        if val_auprc > max_val_auprc:
            max_val_auprc = val_auprc
            early_stopping_patients_counter = 0
            # Save the model on the hard corded path
            print(f'Saving model at epoch {epoch+1} with val_auprc {val_auprc}')
            save_checkpoint(
                state={
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss_over_one_epoch, 
                }, filename=os.path.join(model_weights_dir, f"5folds_splitID{splitID}-epoch-{epoch+1}-valauprc-{val_auprc}.pt")
            )

        else:
            early_stopping_patients_counter += 1

        print(f'    Epoch:{epoch} | {early_stopping_patients_counter}/{patients} | train_loss: {loss_over_one_epoch} | val_loss: {val_loss} | val_auprc: {val_auprc}')

        if early_stopping_patients_counter >= patients:
            print("Early stopping triggered")
            # Save the model on the hard corded path
            save_checkpoint(
                state={
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss_over_one_epoch, 
                }, filename=os.path.join(model_weights_dir, f"5folds_splitID{splitID}-epoch-{epoch+1}-valauprc-{val_auprc}.pt")
            )
            return val_auprc
    
    print(f'Reach max {max_epochs} epochs')
    return val_auprc

#---------------------------------------------------

def main():
    # set up for GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

    parser = argparse.ArgumentParser()

    # name of the experiment for wandb
    parser.add_argument('--entity',  type=str, default='team_path')
    parser.add_argument('--project', type=str, default='stain')
    parser.add_argument('--name',    type=str, required=True, default='stain_cross_validation')
    parser.add_argument('--train_config_yaml', type=str, default='train_config.yaml')

    args = parser.parse_args()

    # get current time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = args.name+'_'+current_time

    # set up wandb
    wandb.init(entity=args.entity, project=args.project, name=experiment_name)

    # parse yaml file
    with open(args.train_config_yaml, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # set up for seed 
    seed = config['seed']
    wandb.config.seed = seed
    seed_evertyhing(seed=seed, worker=True)

    # set up for GPU
    GPUs_LIST = config['gpu'] ##gpu ids to use
    wandb.config.GPUs_LIST = GPUs_LIST
    device = torch.device("cuda:" + str(GPUs_LIST[0]))

    # create the directory to save the model weights
    output_dir = config['output_dir']
    model_weights_dir = os.path.join(output_dir, f'{experiment_name}') 
    wandb.config.model_weights_dir = model_weights_dir
    if not os.path.isdir(model_weights_dir):
        os.makedirs(model_weights_dir, exist_ok=True)

    # loop over cross validation
    for splitID in [1,2,3,4,5]:
        print(f'Start training splitID:{splitID} using {device} GPU')

        # intialize the model 
        model = initialize_pretrained_model()
        if len(GPUs_LIST) > 1:
            model = nn.DataParallel(model, device_ids=GPUs_LIST).cuda()
            model = model.to(device)

        # generate the data 
        train_mean, train_std = compute_mean_std_train(splitID)
        wandb.log({f'splitID{splitID}_train_mean': train_mean, f'splitID{splitID}_train_std': train_std})
        train_dl = build_train_data(splitID=splitID, config=config)
        val_dl = build_val_data(splitID=splitID, config=config)

        # set up the optimizer
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], weight_decay=0.0, momentum=0.9)

        val_auprc = train_w_early_stopping(
            model=model, train_dl=train_dl, val_dl=val_dl, optimizer=optimizer, 
            device=device, train_mean=train_mean, train_std=train_std,
            config=config, splitID=splitID, model_weights_dir=model_weights_dir)
        print(f'Training {splitID} is done with the val AUPRC: {val_auprc}')

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()