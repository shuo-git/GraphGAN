## GraphGAN

In this repository, We implement [GraphGAN](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16611) ([arXiv](https://arxiv.org/abs/1711.08267)) using tensorflow.

> GraphGAN: Graph Representation Learning With Generative Adversarial Nets  
Hongwei Wang, Jia Wang, Jialin Wang, Miao Zhao, Weinan Zhang, Fuzheng Zhang, Xing Xie, Minyi Guo  
32nd AAAI Conference on Artificial Intelligence, 2018

### Requirements
The code has been tested running under Python 3.6.5, with the following packages installed (along with their dependencies):

- tensorflow == 1.8.0
- tqdm == 4.23.4 
- numpy == 1.14.3
- sklearn == 0.19.1

	


### Files in the folder
- `data/`: the preprocessed data and the pre-trained embedding for each experiment
- `exp`: the scripts used to perform experiments
- `graphGAN/`: source codes


### How to use this repository?
For simplicity, we use `CODE_DIR` to denote the path that you store this repository.

#### Data preprocess
We take the recommendation experiment using dataset [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/)
for example. The edges are stored in
`${CODE_DIR}/data/ml-1m/all_edges.txt` with the following format:

```0	1```  
```3	2```  
```...```

```
cd ${CODE_DIR}/data/ml-1m
python ${CODE_DIR}/graphGAN/utils/recommendation.py all_edges.txt
```

This step will divide the training set and the test set and also generate `train_trees.pkl`
used in the subsequent training step.

#### Training

```angular2
cd ${CODE_DIR}/exp/ml-1m
export PYTHONPATH=/data/private/ws/projects/GraphGAN/ShuoGraphGAN:$PYTHONPATH
mkdir -p ./train
python ${CODE_DIR}/bin/trainer.py \
--data_dir ${CODE_DIR}/data/ml-1m \
--log_dir ./train
```

The trained models and evaluation results are stored in `./train`.
You can also evaluate the trained model with `scorer.py`:

```angular2
python ${CODE_DIR}/bin/scorer.py \
--data_dir ${CODE_DIR}/data/ml-1m \
--log_dir ./train
```
