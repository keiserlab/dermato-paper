import os
import sys
import argparse
from datetime import datetime
import yaml
import csv
import pickle  
from collections import OrderedDict

import numpy as np
import pandas as pd
import zarr 
from PIL import Image
from skimage import measure
from sklearn.metrics import auc
from tqdm import tqdm
import matplotlib.pyplot as plt

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# W&B
import wandb

# Kornia
import kornia as K

# Pytorch Lightning
import pytorch_lightning as pl

# custom modules
from utils import gen_performance_metrics, compute_tp_fp_tn_fn, initialize_trained_model
#----------------------------------------------------------
class StainDataSet(Dataset):
    """

    Initializes an instance of the class.
    Args:
        df (DataFrame): The input DataFrame.

    """
    def __init__(self, df):
        self.df = df
        self.tile_size = 256
        
        # compute the baseline, the ratio of positive to negative samples
        self._compute_baseline()

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
        print(f'Baseline accuracy: {self.ratio_positives}% positives, {self.ratio_negatives}% negatives')
        

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
    
    
CURVE_COLORS = ['darkorange', 'darkslategray', 'forestgreen', 'darkviolet', 'lightseagreen', 'gold', 'steelblue', 'sienna', 'olivedrab']




def get_FPRs_and_TPRs(perf_f):
    """
    Retrieves the False Positive Rates (FPRs) and True Positive Rates (TPRs) from a performance file.
    Args:
        perf_f (str): The path to the performance file.
    Returns:
        tuple: A tuple containing two lists: sFPRs (sorted FPRs) and sTPRs (sorted TPRs).
    
    """

    FPRs = []
    TPRs = []
    with open(perf_f, 'r') as fi:
        reader = csv.reader(fi)
        next(reader)
        for step_size,TPR,TNR,PPV,FPR,FNR,FDR in reader:
            FPRs.append(float(FPR))
            TPRs.append(float(TPR))
    zipped = zip(FPRs, TPRs)
    szipped = sorted(zipped, key=lambda x: (x[0],x[1]))
    sFPRs = [f for f,t in szipped]
    sTPRs = [t for f,t in szipped]
    return sFPRs, sTPRs

def plot_AUROC_curves(params,title='AUROC curves', colors=CURVE_COLORS):
    """
    Plots the AUROC curves for multiple sets of FPRs and TPRs.
    Args:
        params (list): A list of tuples containing FPRs, TPRs, AUROC, and label for each set of curves.
        title (str): The title of the plot (default: 'AUROC curves').
        colors (list): A list of colors for each set of curves (default: CURVE_COLORS).

    """

    plt.gcf()
    ax = plt.gca()
    for i,(FPRs, TPRs, auroc, label) in enumerate(params):
        label = '{0}  (AUROC={1:.3f})'.format(label, auroc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.plot(FPRs, TPRs, color=colors[i], alpha=0.7, lw=2, label=label)
        plt.xlim([0., 1.])
        plt.ylim([0., 1.])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    if title:
        plt.title('{}'.format(title))
    return

def get_TPRs_and_PPVs(perf_f):
    """
    
    Args:
        perf_f (str): performance metrics file where columns are step_size, TPR, TNR, PPV, FPR, FNR, FDR
    
    Returns:
        TPRs (list): list of TPRs
        PPVs (list): list of PPVs
    """
    TPRs = []
    PPVs = []
    with open(perf_f, 'r') as fi:
        reader = csv.reader(fi)
        next(reader)
        for step_size,TPR,TNR,PPV,FPR,FNR,FDR in reader:
            TPRs.append(float(TPR))
            PPVs.append(float(PPV))
    zipped = zip(TPRs, PPVs)
    szipped = sorted(zipped, key=lambda x: (x[0],x[1]))
    sTPRs = [t for t,p in szipped]
    sPPVs = [p for t,p in szipped]
    return sTPRs, sPPVs

def plot_AUPRC_curves(params, title='AUPRC curves', colors=CURVE_COLORS):
    """
    Plots the AUPRC curves.
    Args:
        params (list): A list of tuples containing TPRs, PPVs, AUPRC, and label for each curve.
        title (str, optional): The title of the plot. Defaults to 'AUPRC curves'.
        colors (list, optional): The colors to use for each curve. Defaults to CURVE_COLORS.
    Returns:
        None
    """
    plt.gcf()
    prevalence = []
    for i,(TPRs, PPVs, auprc, label) in enumerate(params):
        label = '{0}  (AUPRC={1:.3f})'.format(label, auprc)
        plt.plot(TPRs, PPVs, lw=2, color=colors[i], label=label)
        # Get prevalence
        prevs = np.array(PPVs)
        prev = np.min(prevs[prevs > 0])
        prevalence.append(prev)
    avg_prev = np.mean(prevalence)
    prevalence = [avg_prev] * len(TPRs)
    plt.plot(TPRs, prevalence, lw=2, color='navy', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower left")
    if title:
        plt.title('{}'.format(title))
    return
    
def gen_AUPRC_curves_from_performance_files(perf_files, labels, title='AUPRC curves', colors=CURVE_COLORS):
    """
    Generate AUPRC curves from performance files.
    Args:
        perf_files (list): List of performance files.
        labels (list): List of labels.
        title (str, optional): Title of the plot. Defaults to 'AUPRC curves'.
        colors (list, optional): List of colors for the curves. Defaults to CURVE_COLORS.
    Returns:
        None
    """
    params = []
    # loop through each performance file
    for i,fi in enumerate(perf_files):
        TPRs, PPVs = get_TPRs_and_PPVs(fi)
        auprc = auc(TPRs, PPVs)
        params.append((TPRs, PPVs, auprc, labels[i]))
    plot_AUPRC_curves(params=params, title=title, colors=colors)
    return

def gen_AUROC_curves_from_performance_files(perf_files, labels, title='AUROC curves', colors=CURVE_COLORS):
    """
    Generates plot of AUROC curves from performance metric files.
    Args:
        perf_files (iter; str): Iterable containing performance metric CSV file_path(s)
        labels (iter; str): Iterable containing labels for AUROC curves. Must contain the same
            number of elements as perf_files.
        outfn (str): Name of file path to write output image
        title (str, optional): Title of AUROC plot
    Returns: 
        None
    """
    params = []
    print(f'per_file:{len(perf_files)} | labels:{len(labels)}')
    assert len(perf_files) == len(labels), "Number of perf_files does not match labels"
    for i,fi in enumerate(perf_files):
        FPRs, TPRs = get_FPRs_and_TPRs(fi)
        auroc = auc(FPRs, TPRs)
        params.append((FPRs, TPRs, auroc, labels[i]))
    plot_AUROC_curves(params=params, title=title, colors=colors) # I need this params
    return

def plot_performance_metrics(fpaths, plot_labels, outfn, roc_title='AUROC curves', prc_title='AUPRC curves', figsize=(13,5), colors=CURVE_COLORS):
    """
    Plot AUROC and AUPRC curves for all performance metric files provided.
    Args:
        fpaths (list): List of file paths for performance metric files.
        plot_labels (list): List of labels for each performance metric file.
        outfn (str): Output file name to save the plot.
        roc_title (str, optional): Title for the AUROC curves plot. Defaults to 'AUROC curves'.
        prc_title (str, optional): Title for the AUPRC curves plot. Defaults to 'AUPRC curves'.
        figsize (tuple, optional): Figure size for the plot. Defaults to (13, 5).
        colors (list, optional): List of colors for the curves. Defaults to CURVE_COLORS.
    Returns:
        plt (matplotlib.pyplot): The generated plot.
    """
    
    #fpaths = load_perf_files(perf_f)
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1)
    gen_AUROC_curves_from_performance_files(perf_files=fpaths, labels=plot_labels, title=roc_title, colors=colors)
    plt.subplot(1,2,2)
    gen_AUPRC_curves_from_performance_files(perf_files=fpaths, labels=plot_labels, title=prc_title, colors=colors)
    if outfn:
        plt.savefig(outfn)
    return plt


def return_best_model(splitID, config):
    """
    Returns the path of the best trained model for a given split ID.
    Args:
        splitID (int): The ID of the split.
        config (dict): The configuration dictionary containing the trained model paths.
    Returns:
        str: The path of the best trained model for the given split ID.

    """
    
    model_paht_for_splitID = config["trained_model_path"][f'split_{splitID}'] 
    return model_paht_for_splitID

def compute_mean_std_train(splitID):
    """    
    Compute the mean and standard deviation of the training data.
    Args:
        splitID (int): The ID of the split.
    Returns:
        train_mean (float): The mean of the training data.
        train_std (float): The standard deviation of the training data.
    
    """

    # hard coded path 
    mean_std_file = f"/srv/home/mtada/dermato-paper/melanA/5folds_csv/split{splitID}_w_tcga_outof_5folds_mean_std.pkl"
    if os.path.isfile(mean_std_file):
        with open(mean_std_file, "rb") as f:
            (train_mean, train_std) = pickle.load(f)
        print(f'Loaded mean and std {train_mean}, {train_std} from {mean_std_file} ...')

    return train_mean, train_std

def build_test_data_with_defined_section(splitID, eval_config):
    """
    Build test data with defined section.
    Args:
        splitID (int): The ID of the split.
        eval_config (dict): The evaluation configuration.
    Returns:
        tuple: A tuple containing a list of test data loaders and a list of all sections.
    
    """
    batch_size = eval_config['batch_size']
    num_workers = eval_config['num_workers']
    stain = eval_config['stain']

    list_test_dls = []
    # Hard coded csv file 
    df = pd.read_csv(f'/srv/home/mtada/dermato-paper/{stain}/5folds_csv/split{splitID}_test_outof_5folds.csv')
    all_sections = list(df['section_name'].unique())
    for section in all_sections:
        df_sec = df.loc[df['section_name']==section].reset_index(drop=True, inplace=False)
        print(f'{section}: {len(df_sec)} rows')
        test_ds = StainDataSet(
            df=df_sec,
            )

        test_dl = DataLoader(
            test_ds, 
            batch_size=batch_size,
            num_workers=num_workers, 
            shuffle=False, #mutually exclusive with sampler when WRS is on
            pin_memory=True, 
            drop_last=False
            )
        list_test_dls.append(test_dl)

    return list_test_dls, all_sections

def inference(model, criterion, test_dl, device, train_mean, train_std):
    """
    
    Args:
        model (nn.Module): model to evaluate
        test_dl (DataLoader): test dataloader
        device (torch.device): GPU device to use

    Returns:
        all_labels (list) : 1D array of labels 
        all_prob_outputs (list): 1D array of predicted probabilities
    """
    all_labels = []
    all_prob_outputts = []

    running_total_loss_over_one_epoch = 0.0
    running_total_tp, running_total_fp, running_total_tn, running_total_fn = 0, 0, 0, 0
    test_dl_tqdm = tqdm(test_dl, total=int(len(test_dl)))

    with torch.set_grad_enabled(False):
        for i, data in enumerate(test_dl_tqdm):
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)

            # mean and std normalization 
            inputs = K.enhance.normalize(data=inputs, mean=train_mean, std=train_std)

            logits = model(inputs)
            # [B,1]->[B]
            logits = torch.squeeze(logits, dim=1)

            # loss
            avg_one_batch_loss = criterion(logits, labels)

            # summing all batch losses
            total_one_batch_loss = avg_one_batch_loss.item()*logits.shape[0]
            running_total_loss_over_one_epoch += total_one_batch_loss

            prob_outputs = torch.sigmoid(logits.cpu())

            # convert probabilities to binary outputs
            binary_outputs = prob_outputs > 0.5
            # current_over_one_batch = (binary_outputs == torch.tensor(labels.cpu(), dtype=torch.int8)).float().sum().item() 
            tp, fp, tn, fn = compute_tp_fp_tn_fn(binary_outputs.numpy().astype(np.uint8), labels.cpu().numpy().astype(np.uint8))
            running_total_tp += tp
            running_total_fp += fp
            running_total_tn += tn
            running_total_fn += fn

            all_labels.extend(labels.cpu())
            all_prob_outputts.extend(prob_outputs)
            # running_total_correct += current_over_one_batch
        
        print(f'TP:{running_total_tp} | FP:{running_total_fp} | TN:{running_total_tn} | FN:{running_total_fn} (tiles)')
        loss_over_one_epoch = running_total_loss_over_one_epoch / len(test_dl.dataset)

        precision_over_one_epoch = running_total_tp / (running_total_tp + running_total_fp)
        recall_over_one_epoch = running_total_tp / (running_total_tp + running_total_fn)
        f1_over_one_epoch = 2 * (precision_over_one_epoch * recall_over_one_epoch) / (precision_over_one_epoch + recall_over_one_epoch)

    wandb.log({
            "inference_loss":loss_over_one_epoch,
            'inference_precision': precision_over_one_epoch,
            'inference_recall': recall_over_one_epoch,
            'inference_f1': f1_over_one_epoch
            }
            )

    return all_labels, all_prob_outputts

#----------------------------------------------------------
def main():

    # set up for GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

    # agument parser
    parser = argparse.ArgumentParser()

    # name of the experiment for wandb
    parser.add_argument('--wandb_entity', type=str, default='team_path')
    parser.add_argument('--wandb_project', type=str, default='MelanA')
    parser.add_argument('--name',    type=str, required=True, default='eavl')
    parser.add_argument('--eval_config_yaml', type=str, default='eval_config.yaml')
    # parser.add_argument('--label_config_yaml', type=str, default='preprocessing/mela_label_preprocessing.yaml')

    args = parser.parse_args()

    # get current time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # W&B init
    experiment_name = args.name+'_'+current_time
    wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=experiment_name)

    # parse yaml files
    with open(args.eval_config_yaml, 'r') as f:
        eval_config = yaml.load(f, Loader=yaml.SafeLoader)

    # set the seed for reproducibility
    seed = eval_config['seed']
    pl.seed_everything(seed, workers=True)
    wandb.config.seed = seed

    # set up for GPU
    GPUs_LIST = eval_config['gpus'] ##gpu ids to use
    wandb.config.GPUs_LIST = GPUs_LIST
    device = torch.device("cuda:" + str(GPUs_LIST[0]))

    # create the directory to save the model weights
    output_dir = eval_config['output_dir']
    model_weights_dir = os.path.join(output_dir, f'{experiment_name}') 
    wandb.config.model_weights_dir = model_weights_dir
    if not os.path.isdir(model_weights_dir):
        os.makedirs(model_weights_dir, exist_ok=True)

    # Store AUROC and AUPRC for each split
    list_all_auroc_from_five_splits = {}
    list_all_auprc_from_five_splits = {}

    # loop over the models
    for splitID in [1,2,3,4,5]:
        print(f'------------------Split {splitID}------------------')
        
        # intialize the model 
        trained_model_path = return_best_model(splitID=splitID, config=eval_config)
        wandb.log({f'splitID{splitID}_trained_model_path': trained_model_path})
        model = initialize_trained_model(trained_model_path=trained_model_path)
        if len(GPUs_LIST) > 1:
            model = nn.DataParallel(model, device_ids=GPUs_LIST).cuda()
            model = model.to(device)

        model.to(device)
        model.eval() 
        
        list_auroc_split = []
        list_auprc_split = []
        # create data loaders
        list_test_dls, list_section_names = build_test_data_with_defined_section(splitID=splitID, eval_config=eval_config)
        print(f'{len(list_test_dls)}: {len(list_section_names)}')
        for test_dl, section_name in zip(list_test_dls, list_section_names):
            print(f'    ------------------Section {section_name}------------------')
            print(f'    test_dl: {len(test_dl.dataset)} tiles')
            # generate the data
            train_mean, train_std = compute_mean_std_train(splitID)
            wandb.log({f'splitID{splitID}_train_mean': train_mean, f'splitID{splitID}_train_std': train_std})

            criterion = nn.BCEWithLogitsLoss(reduction='mean')

            all_labels, all_prob_outputs = inference(model, criterion, test_dl, device, train_mean, train_std)

            output_file = os.path.join(output_dir, f'splitID{splitID}_{section_name}_{experiment_name}.csv')
            print(f'all_labels: {len(all_labels)} | all_prob_outputs: {len(all_prob_outputs)}')
            assert len(all_labels) == len(all_prob_outputs)
            prevalence, output_file, all_TPRs, all_FPRs, all_PPVs = gen_performance_metrics(labels=all_labels, probs=all_prob_outputs, outfn=output_file, step_size=0.01)

            # prevalence should be all the same
            wandb.log({f"splitID{splitID}_{section_name}_prevalence": prevalence})
            wandb.log({f"splitID{splitID}output_file": output_file})
    
            auroc = auc(x=all_FPRs, y=all_TPRs)
            auprc = auc(x=all_TPRs, y=all_PPVs)
            print(f'AUROC: {auroc} | AUPRC: {auprc} for split {splitID} {section_name}')
            wandb.log({f'splitID{splitID}_{section_name}_AUROC': auroc, f'splitID{splitID}_{section_name}AUPRC': auprc})

            list_auroc_split.append(auroc)
            list_auprc_split.append(auprc)
            list_all_auroc_from_five_splits[section_name] = auroc
            list_all_auprc_from_five_splits[section_name] = auprc

            outfn = os.path.join(output_dir, f'splitID{splitID}_{experiment_name}.png')
            # generate and save the ROC and PRC curves
            plot_lables=[f'{args.name}']
            fpaths = [output_file]
            plt = plot_performance_metrics(fpaths=fpaths, plot_labels=plot_lables, outfn=outfn, roc_title='AUROC curves', prc_title='AUPRC curves', 
                                                        figsize=(13,5), colors=CURVE_COLORS)
            wandb.log({f"splitID{splitID}_{section_name}_AUROC_AUPRC": plt})

        avg_auroc = np.mean(list_auroc_split)
        avg_auprc = np.mean(list_auprc_split)
        print(f'avg_auroc: {avg_auroc} avg_auprc: {avg_auprc} among {len(list_section_names)} sections for split {splitID}')
        wandb.log({f'avg_test_auroc_{splitID}fold': avg_auroc, f'avg_test_auprc_{splitID}fold': avg_auprc})

    sorted_auroc_items = sorted(list_all_auroc_from_five_splits.items(), key=lambda x: x[1], reverse=True)
    sorted_section_name_auroc = [item[0] for item in sorted_auroc_items]
    sorted_auroc = [item[1] for item in sorted_auroc_items]
    print(f'sorted_section_name_auroc: {sorted_section_name_auroc}')
    print(f'sorted_auroc: {sorted_auroc}')

    sorted_auprc_items = sorted(list_all_auprc_from_five_splits.items(), key=lambda x: x[1], reverse=True)
    sorted_section_name_auprc = [item[0] for item in sorted_auprc_items]
    sorted_auprc = [item[1] for item in sorted_auprc_items]
    print(f'sorted_section_name_auprc: {sorted_section_name_auprc}')
    print(f'sorted_auprc: {sorted_auprc}')

    wandb.finish()


#------------------------------------------------------------


if __name__ == '__main__':
    main()