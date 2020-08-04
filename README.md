# Stochastic Neural Architecture Search (SNAS) (Liu et al., 2019)

An unofficial Pytorch implementation of [Stochastic Neural Architecture Search (Liu et al., 2019)](https://arxiv.org/abs/1812.09926) based off of [Astrodyn94's implementation](https://github.com/Astrodyn94/SNAS-Stochastic-Neural-Architecture-Search-). This was developed to make the SNAS code easier to pick up and work with.

## Dataset
This implementation was created on a specific subset of the  [xView dataset](http://xviewdataset.org/), which is included in the repository, but is easily compatible with any other image dataset by changing the dataloader.

The dataset contains satellite images of land vehicles and the SNAS network was trained to classify these vehicles into a subset of the xView classes.

## Training
To begin training with `python >= 3.6.5`,

1. `pip install -r requirements.txt`
2. Configure options in `./options/default_option.py`
3.  `python main.py` or `python main_constraint.py` (depending on if resource constraints as in the paper are wanted)

To inspect a checkpoint,
`inspect_model.py --experiment=[experiment name] --epoch=[epoch number] --visualize=[bool]`

## Docs
### Directories
- `./dataset`: Dataset stored with a folder for each image class
- `./experiments`: Model checkpoints and accuracy data will be saved in subdirectories here as `.pt` files which can be loaded using `inspect_model.py`
- `./visualizations`: Cells learned during NAS can be visualized after loading a model checkpoing with `inspect_model.py` and will be saved into a subdirectory according to the experiment that the checkpoint belongs to

### Scripts
- `main.py` The main training code for unregularized SNAS, contains data loading (currently implementing weighted sampling to address class imbalance) and training across epochs.
- `main_constraint.py` Training code for regularized SNAS (resource constraints)
- `./option/default_option.py`: Training configurations including all **hyperparameters** specified in the paper.
- `model_search.py`: The model for unregularized SNAS.
- `inspect_model.py`: Used to inspect model checkpoints saved in `./experiments` by taking in the experiment name (`--experiment=`), checkpoint epoch (`--epoch=`), and whether you'd like to visualize a cell learned at this checkpoint and save it as a `.png` into `./visualizations` (`--visualize=`).
- `genotypes.py`: Defines the operation space for SNAS and has different operation spaces from other papers (NASNet and AmoebaNet) to try.
- `operations.py`: Pytorch implementation of operations.
- `utils.py`: Utility functions (accuracy, transforms, etc.)
- `visualize.py`: Helper file for visualizations
