# CSE 597 Project: Weakly Supervised Temporal Sentence Grounding with Gaussian-based Contrastive Proposal Learning

This project is based on the following paper:

CPL: Weakly Supervised Temporal Sentence Grounding with Gaussian-based Contrastive Proposal Learning

[[Paper](https://minghangz.github.io/uploads/CPL/CPL_paper.pdf)] [[Project Page](https://minghangz.github.io/publication/cpl/)] [[GitHub](https://github.com/minghangz/cpl)] 

## Pipeline

![pipeline](imgs/pipeline.png)

## Main Replication Results

### Charades-STA Dataset (Trained model from scratch)

| Method  | Rank1@0.3 | Rank1@0.5 | Rank1@0.7 | Rank5@0.3 | Rank5@0.5 | Rank5@0.7 |
| :-----: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
|   CPL   |   67.13   |   49.72   |   23.43   |   96.49   |   84.39   |   51.33   |
|   CPL*  |   67.13   |   49.72   |   23.43   |   96.49   |   84.39   |   51.33   |

The trained models are in the [`checkpoints/charades`](/checkpoints/charades) directory. 

## Requiments

- pytorch
- h5py
- nltk
- fairseq

### Environment Setup

Create a conda environment and install the following versions for pytorch, torchaudio, torchvision and cuda (This is the cuda version for Nvidia RTX 3080).
```bash
conda create -n cpl python=3.8.12

conda activate cpl

pip3 install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -f https://download.pytorch.org/whl/cu11/torch_stable.html
```

Install the dependencies
```bash
pip3 install h5py nltk fairseq==0.10.0
```

## Quick Start

### Data Preparation

The converted I3D feature for the Charades-STA dataset can be downloaded from the following links:

[[Google Drive](https://drive.google.com/file/d/13TUxH41LE6aZmj9rvvX7fThkc9yb0DxO/view?usp=sharing)] [[Pan Baidu](https://pan.baidu.com/s/1WhWreaHIx8pI5hLK2uyCdw?pwd=4g9h)] 

The directory structure should be as follows:

```
data
├── activitynet
├── charades
│   ├── i3d_features.hdf5
│   ├── glove.pkl
│   ├── train.json
│   ├── test.json
```

### Training

To train on the Charades-STA dataset:
```bash
python train.py --config-path config/charades/main.json --log_dir logs --tag TAG
```

Use `--log_dir` to specify the directory where the logs are saved, and use `--tag` to identify each experiment. They are both optional.

The model weights are saved in [`checkpoints/charades`](/checkpoints/charades) by default and can be modified in the configuration file.

### Inference

The trained models are provided in [`checkpoints/charades`](/checkpoints/charades). Run the following commands for evaluation:

```bash
# Use loss-based strategy during inference
python train.py --config-path config/charades/main.json --resume CHECKPOINT_FILE --eval
# Use vote-based strategy during inference
python train.py --config-path config/charades/main.json --resume CHECKPOINT_FILE --eval --vote
```

The configuration file is the same as training.

### Other Considerations

This code has been modified to run on Windows 10. To ensure that the 'best model' checkpoint is saved when run on Linux or MacOS refer to the following code in line 47 of [`runners/main_runner.py`](/runners/main_runner.py)

```bash
# for Windows
os.system('copy "%s" "%s"'%(save_path, os.path.join(self.model_saved_path, 'model-best-max.pt')))
# for Linux and MacOS
os.system('cp %s %s'%(save_path, os.path.join(self.model_saved_path, 'model-best-max.pt')))
```
