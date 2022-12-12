# CSE 597 Project: Weakly Supervised Temporal Sentence Grounding with Gaussian-based Contrastive Proposal Learning

This project is based on the following paper:

CPL: Weakly Supervised Temporal Sentence Grounding with Gaussian-based Contrastive Proposal Learning

[[Paper](https://minghangz.github.io/uploads/CPL/CPL_paper.pdf)] [[Project Page](https://minghangz.github.io/publication/cpl/)] [[GitHub](https://github.com/minghangz/cpl)] 

## Pipeline

![pipeline](imgs/pipeline.png)

## Main Results

### Charades-STA Dataset

| Method  | Rank1@0.3 | Rank1@0.5 | Rank1@0.7 | Rank5@0.3 | Rank5@0.5 | Rank5@0.7 |
| :-----: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
|   CPL   |   66.40   |   49.24   |   22.39   |   96.99   |   84.71   |   52.37   |
| CPL$^*$ |   65.99   |   49.05   |   22.61   |   96.99   |   84.71   |   52.37   |

The trained models are in the checkpoints directory. 

## Requiments

- pytorch
- h5py
- nltk
- fairseq

## Quick Start

### Data Preparation

We use the C3D feature for the ActivityNet Captions dataset. Please download from [here](http://activity-net.org/challenges/2016/download.html) and save as `data/activitynet/c3d_features.hdf5`. We use the I3D feature provided by [LGI](https://github.com/JonghwanMun/LGI4temporalgrounding) and use [this script](/data/convert_npy_to_hdf5.py) to convert the file format to HDF5. We also provide the converted I3D feature for the Charades-STA dataset, and can be downloaded from [here](https://pan.baidu.com/s/1WhWreaHIx8pI5hLK2uyCdw?pwd=4g9h). We expect the directory structure to be the following:

```
data
├── activitynet
│   ├── sub_activitynet_v1-3.c3d.hdf5
│   ├── glove.pkl
│   ├── train_data.json
│   ├── val_data.json
│   ├── test_data.json
├── charades
│   ├── i3d_features.hdf5
│   ├── glove.pkl
│   ├── train.json
│   ├── test.json
```


### Training

To train on the ActivityNet Captions dataset:
```bash
python train.py --config-path config/activitynet/main.json --log_dir LOG_DIR --tag TAG
```

To train on the Charades-STA dataset:
```bash
python train.py --config-path config/charades/main.json --log_dir LOG_DIR --tag TAG
```

Use `--log_dir` to specify the directory where the logs are saved, and use `--tag` to identify each experiment. They are both optional.

The model weights are saved in `checkpoints/` by default and can be modified in the configuration file.

### Inference

Our trained model are provided in [`checkpoints/`](/checkpoints/). Run the following commands for evaluation:

```bash
# Use loss-based strategy during inference
python train.py --config-path CONFIG_FILE --resume CHECKPOINT_FILE --eval
# Use vote-based strategy during inference
python train.py --config-path CONFIG_FILE --resume CHECKPOINT_FILE --eval --vote
```

The configuration file is the same as training.
