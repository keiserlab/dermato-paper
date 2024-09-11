# Machine-learning convergent melanocytic morphology despite noisy archival slide
## Requirements 
To use the code in this repository, you will need to set up two different environments: one for WSI preprocessing and another for model training. 

For WSI preprocessing, use the `wsi_pipeline.yml` environment. 

For model training and evaluation, use the `dermato.yml` environment.

## How to Use
As we are not sharing the whole slide images required to reproduce the results, please note that you cannot simply run the code provided in this repository to replicate the exact results. Certain files contain hard-coded paths that are specific to our setup. However, we have included essential components of the pipeline that can serve as a reference or starting point for your own research.

### WSI Segmentation
To perform WSI segmentation, refer to the `segmentation/TissueExtractionToolkit_demo.ipynb` notebook.

### Label Creation for Sox10 and MelanA/MelPro
The code for stain-specific labeling can be found in `dermato/stain_labeling.py`. After labeling, the dataset is created in CSV format using `dermato/train.py`. We also used `dermato/labelmap.py` to create the label map to visually check the labels.

### Training a Model for Sox10 and MelanA/MelPro
To train the model, use `dermato/train.py` along with the `dermato/train_config.yaml` file. Please note that there are several hard-coded paths in the code for loading CSV files and saving weights.

### Evaluation of the Model
To evaluate the model's performance, including metrics such as AUROC and AUPRC, use `dermato/eval.py` with the `dermato/eval_config.yaml` file. Similar to training, there are hard-coded paths in the code.

### Prediction Confidence Heatmap Generation
To generate prediction confidence heatmaps, use `dermato/heatmap.py` with the `dermato/heatmap_config.yaml` file. The script utilizes a sliding window approach and PyTorch Lightning for scalability.

### Saliency Map Generation
For saliency map generation, refer to the `saliency/generate_saliency_visualizations.ipynb` notebook.

## Citation

## License

