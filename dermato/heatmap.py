import os
import shutil
from argparse import ArgumentParser
import yaml
import pickle

import zarr 
import numpy as np
import pandas as pd


from numcodecs import Blosc
#set this for multiple processing
Blosc.use_threads = False

# pytorch
import torch
import torch.nn as nn
import torch.utils.data as data

# torch vision 
import torch.utils.data as data

# pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# wandb 
import wandb

# Kornia
import kornia as K

# custom 
from utils import initialize_trained_model

#---------------------------------------------------

class MelanocytesModel(pl.LightningModule):
    """
    A PyTorch Lightning Module for the Melanocytes Model.

    """

    def __init__(self, model, mean_std_file, save_dir:str):#, stride:int = 256):
        super().__init__()

        #GASKIN APPROACH: use passed in model as-is
        self.model = model
        
        self.mean_std_file = mean_std_file

        # self.config= config

        #directory to dump prediction heatmap output
        self.save_dir = save_dir

        # 
        self.train_mean, self.train_std = self._compute_mean_std_train()
        
    def _compute_mean_std_train(self):
        """ retrieve the mean and std values from the training set for normalization """
        # mean_std_file = self.config['mean_std_file']
        # wandb.config.mean_std_file = mean_std_file

        if os.path.isfile(self.mean_std_file):
            with open(self.mean_std_file, "rb") as f:
                (train_mean, train_std) = pickle.load(f)
            print(f'Loaded mean and std {train_mean}, {train_std} from {self.mean_std_file} ...')

        return train_mean, train_std

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        #this is not fully fleshed out, as I'm not training this model
        imgs, data = batch #critical to unpack this!

        # mean and std normalization 
        imgs = K.enhance.normalize(data=imgs, mean=self.train_mean, std=self.train_std)
        
        pred = self.model(imgs) 
        out = torch.sigmoid(pred)
        # loss = self.loss_fn(out)
        # self.log('training_loss', loss)
        # return loss
        return out
    
    def predict_step(self, batch, batch_idx):
        imgs, labels = batch #critical to unpack this! (won't use label for inference)

        # mean and std normalization 
        imgs = K.enhance.normalize(data=imgs, mean=self.train_mean, std=self.train_std)

        pred = self.model(imgs) 
        out = torch.sigmoid(pred)
        #added rounding here for preds for stability of plotting
        out = torch.round(out, decimals = 4)

        return out

    def on_predict_batch_end(self, batch_preds:torch.tensor, batch, batch_idx):
        """
        Called in the predict loop after each batch.
        In this case, we will be using it to "batch write" tiles. Since we can read them into memory in order,
        we can utilize larger zarr chunks to minimize IO calls and allow for more efficient writing processes.
        Args:
            batch_preds (Optional[Any]) The outputs of predict_step_end(test_step(x))=
            batch (Any) The batched data as it is returned by the test DataLoader.
            batch_idx (int) the index of the batch
            # dataloader_idx (int) the index of the dataloader
        Returns:
            None
        """

        _, batch_data = batch #unpack original batch to get metadata

        #move coords to cpu to create a df which we can use to groupby 
        batch_data['save_x'] = batch_data['save_x'].cpu()
        batch_data['save_y'] = batch_data['save_y'].cpu()
        batch_df = pd.DataFrame(batch_data)

        #map of randID (name) to df
        wsi_dfs = {}

        #handle the case that there are multiple wsis in one batch (can only be 2?)
        if batch_df['Random ID'].nunique() > 1:
            for randID, randID_group in batch_df.groupby('Random ID'): #COULD BE THE SAME ANYWAYS
                wsi_dfs[randID] = randID_group
        else:
            #if just one wsi, add it to wsi_dfs so that it's included in for loop below
            wsi_dfs[batch_df['Random ID'].iloc[0]] = batch_df
        #go through each split df (should be max of 2)
        
        for randID, df in wsi_dfs.items():
            #group by column so we can batch write the correct span of each
            for x, col_group in df.groupby('save_x'):
                preds = batch_preds[col_group.index]
                #preds need to be on cpu to be saved to disk, fix x for single column indexing
                self._save_to_zarr(preds.cpu(), randID, [x, col_group['save_y'].values])

        return 
        #done
    

    def _save_to_zarr(self, batch_preds:torch.tensor, wsi_name:str, coords:list[list[int, int]]):
        #do filtering prior to this to prevent two separate wsis from being saved
        synchronizer = zarr.ProcessSynchronizer(self.save_dir + wsi_name + '.sync')
        out = zarr.open(self.save_dir + wsi_name, mode = 'a',
                        # shape = wsi_shape, 
                        # chunks = (self.tile_size*2, self.tile_size*2),
                        # write_empty_chunks=False, 
                        # compressor = self.compressor,
                        synchronizer = synchronizer) #use this to enable file locking
        #x should be a fixed number
        x, ys = coords

        #save to single column x, y_coords, and along last dim = (num_class = 4)
        out.oindex[x, ys, :] = batch_preds

    #add argparsing (but there are no params for this model)
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Model')
        #add additional params here
        # parser.add_argument(...)
        return 

#---------------------------------------------------

class TileDataset(data.Dataset):
    """
    Initializes the Heatmap class.

    Args:
        df (DataFrame): The input DataFrame containing information about the WSIs and CERAD labels.
        save_paths (str): The path to save the heatmap images.
        stride (int): The stride size for tiling the WSIs. Default is 256.
        num_classes (int): The number of classes within the dataset. Default is 4.
        save (bool): Whether to save the heatmap images. Default is True.
        wsi_filter (list[str]): A list of WSI IDs to filter the DataFrame. Default is None.

    """
    def __init__(self, df, save_paths:str, stride:int = 256, num_classes = 2, save = True, wsi_filter:list[str] = None):

        super().__init__()
        #define a static tile size (based on what model has been trained on)
        self.tile_size = 256
        #create class variables with init parameters
        self.stride = stride
        self.save_path = save_paths

        #compressor for writing to zarr files
        self.compressor = Blosc(cname='lz4', clevel=1, shuffle=1)

        #modify wsi_csv
        def __create_wsi_csv(df):
            #create Random ID column 
            df['Random ID'] = df['path2he'].apply(lambda x: x.split('/')[-1].split('.zarr')[0]) # it was split('.')[0] before
            
            #create a reference to each object's reader
            df['zarrReader'] = df['path2he'].apply(lambda path: zarr.open(path, mode='r'))
            df['shape'] = df['zarrReader'].apply(lambda z: z.shape) #consider when to insert padding
            print('Created zarr reader for each section')
            return df
        
        #load in wsi csv with path to WSIs and CERAD labels (and other info if necessary)
        self.df = __create_wsi_csv(df)

        #define the number of classes within the dataset, used for creating output shape dimension
        self.num_classes = num_classes

        #allow us to filter for specific wsis
        if wsi_filter:
            self.df = self.df[self.df['Random ID'].isin(wsi_filter)]

        #store number of tiles for easy access
        self.num_wsi = self.df.shape[0]

        #create save destination for each WSI depending on WSI size (could naiively make it same size as WSI, but empty)
        if save:
            self._create_save_destination()

        self.tiles = self._vectorized_tile_df()

    
    def __getitem__(self, idx):
        row = self.tiles.loc[idx]
        #load in tile with wsi_name reference and coords of top left point
        tile = self._load_zarr(row['Random ID'], (row['read_x'], row['read_y']))
        #create a dict, data, that has label inside of it as well as WSI_name and save_coords
        data = row[['Random ID', 'save_x', 'save_y']].to_dict() #might need to add label for downstream analysis?

        tile = K.image_to_tensor(tile, keepdim=True).float() / 255.0
        
        return tile, data

    def __len__(self):
        #returning total number of tiles, if need to know num_wsi, check self.num_wsi
        return self.tiles.shape[0]

    
    def _vectorized_tile_df(self):
        """
        Creates a single dataframe with each row representing a valid tile within a WSI.
        Encapsulates all tiles in the dataset, across an arbitrary number of WSIs.
        """
        dfs = []
        for _, seg in self.df.iterrows():
            #generate a meshgrid with shape of input
            x = np.arange(0, seg['shape'][0] - self.tile_size, self.stride)
            y = np.arange(0, seg['shape'][1] - self.tile_size, self.stride)
            i_coords, j_coords = np.meshgrid(x, y, indexing = 'ij')

            #construct dictionary with which to instantiate df
            out_dict = {
                        'read_x': i_coords.flatten(),
                        'read_y': j_coords.flatten(),
                        }
            df = pd.DataFrame(out_dict, dtype = np.int32)
            df['save_x'] = (df['read_x'] // self.stride).astype(np.int32)
            df['save_y'] = (df['read_y'] // self.stride).astype(np.int32)
            print(f"Section RANDOM ID: {seg['Random ID']}")
            df['Random ID'] = [seg['Random ID']] * df.shape[0]
            dfs.append(df)
        return pd.concat(dfs, ignore_index = True)
                    
                    
    def _load_zarr(self, randID, coords):
        """
        Use coordinate indexing to access a chunk of the zarr file. 
        The output will be fed into __getitem__ method, which will be how a dataloader accesses the data.
        We are given a set of coordinates from class variable self.tiles that possesses
        rows with a single (x, y) coordinate representing the top left corner of a fixed box.
        This box will navigate around our image via these coordinates.
        Args:
            randID (str) - 'Random ID' of the WSI 
            coords (tuple) - Coordinates of top left corner of chunk that we are loading in.
        Returns:
            img_chunk (zarr) - An image chunk of the WSI, represented in zarr array format for parallelized processing.
        """
        #select the correct Zarr reader from wsi_csv (to avoid repetitive zarr.open call)
        mask = self.df['Random ID'] == randID
        # print(f"Masked wsi_csv: {self.wsi_csv.loc[mask, 'zarrReader'].values}")
        z = self.df.loc[mask, 'zarrReader'].values[0]
        #do orthogonal indexing via slices
        x, y = coords
        img_chunk = z.oindex[slice(x, x + self.tile_size), slice(y, y + self.tile_size)]
        # print(img_chunk.shape, 'Shape of Tile')
        return img_chunk

    def _create_save_destination(self):
        """
            Creates an empty zarr file for each WSI to store the output heatmap. 
            Size will be the same as original WSI, and we will leave the rest of the zarr tensor unwritten.
        """
        #create a parent directory for storing all WSI heatmaps of this stride
        print(f'Creating {self.num_wsi} empty zarr files to save heatmaps')
        for _, wsi in self.df.iterrows():
#             wsi = row[1] #access non-index elements

            #calculate heatmap output size, assuming wsi_shape includes padding
            wsi_shape = wsi['shape']
            #output shape calculation could be wrong for stride < 256
            output_shape = ((wsi_shape[0] // self.stride), (wsi_shape[1] // self.stride), self.num_classes)
            #wsi_shape - 256?
            print(f'Section Shape:{wsi_shape}, Output shape: {output_shape}')

            #create save path for output wsi_heatmap
            wsi_path = self.save_path + wsi['Random ID']
            #create .sync file for multiprocessing
            synchronizer = zarr.ProcessSynchronizer(wsi_path + '.sync')
            #delete file if it already exists
            if os.path.isdir(wsi_path):
                shutil.rmtree(wsi_path)

            #create zarr file with read/write access, output_shape, etc.
            out = zarr.open(wsi_path, mode = 'a', shape = output_shape, #could be the issue
                            chunks = (self.tile_size*2, self.tile_size*2),
                            write_empty_chunks=False, 
                            compressor = self.compressor,
                            synchronizer = synchronizer) #use this to enable file locking
            #CONSIDER: adding a column to wsi_csv with each zarrWriter.
    
#---------------------------------------------------

def main(config_dict):

    # below code are for creating heatmaps in the loop
    seed = config_dict['seed']
    pl.seed_everything(seed)
    wandb.config.seed = seed

    STRIDE = int(config_dict['stride'])
    wandb.config.STRIDE = STRIDE

    EXPERIMENT = config_dict['experiemnt_name'] #os.path.basename(config_dict['model_weights']).split(".pt")[0]
    wandb.config.EXPERIMENT = EXPERIMENT

    SAVE_PATH = f"{config_dict['out_dir']}/{EXPERIMENT}/stride_{STRIDE}/"
    wandb.config.SAVE_PATH = SAVE_PATH
    os.makedirs(SAVE_PATH, exist_ok = True)

    # hard coded file path. base direcotry for all H&E sections
    dir_he = '/srv/ds/set-1/user/mtada/HE_sox10_melanA'

    # list of all H&E sections - a list of lists 
    list_sections_all = config_dict['sections']

    # list of all weights 
    list_weights = config_dict['list_model_weights']
    # list of all mean_std files
    list_mean_std = config_dict['list_mean_std_files']

    # batch size and num workers 
    BATCH_SIZE = config_dict['batch_size'] #param
    NUM_WORKERS = config_dict['num_workers']

    for idx in range(10):
        list_sections = list_sections_all[idx]
        print(f'Fold {idx}: processing {len(list_sections)} sections')
        print(f'{list_sections}')

        # get all the paths to the H&E sections
        all_he_sections_path = [f'{dir_he}/{section_name}.zarr'for section_name in list_sections]
        df_dict = {'path2he':all_he_sections_path}
        df = pd.DataFrame(df_dict)

        # intiliaze the trained model
        trained_model_path = list_weights[idx]
        print(f'Fold {idx}: loading model {trained_model_path}')
        model = initialize_trained_model(trained_model_path=trained_model_path)
        mean_std_file = list_mean_std[idx]
        lighting_model = MelanocytesModel(model=model, save_dir=SAVE_PATH, mean_std_file=mean_std_file)

        trainer = Trainer(
            accelerator='gpu',
            devices=config_dict['devices'],
            strategy='ddp',
        )

        dataset = TileDataset(
            df = df,
            save_paths=SAVE_PATH, #pass in paths dict
            stride = STRIDE,
            num_classes = 1,
            save = True,
            #this is a list input, remove to run on entire wsi list or pass as an arg --wsi_filter=False 
            wsi_filter = list_sections #config_dict['sections']
        )

        dataloader = data.DataLoader(dataset,
                                    shuffle = False,
                                    batch_size = BATCH_SIZE,
                                    num_workers = NUM_WORKERS,
                                #  worker_init_fn = WSIStreamDataset.worker_init_fn,
                                    )

        # print(f'datasize for {config_dict["sections"]}: {len(dataset)}')
        trainer.predict(lighting_model, dataloader) #currently rounding in predict_step
        print(f'Finished fold {idx}')

        torch.cuda.empty_cache()


#---------------------------------------------------

if __name__ == "__main__":
    # set up for GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

    parser = ArgumentParser()
    parser.add_argumetn('--entity', type=str, default='team_path')
    parser.add_argument('--project', type=str, default='dermato-predictionmaps')
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--config_yaml', type=str, default='heatmap_config.yaml')
    
    #begin program
    args = parser.parse_args()

    wandb.init(entity=args.entity, project=args.project, name=args.name)


    # load the label config yaml file
    with open(args.config_yaml, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    main(config_dict)