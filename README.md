# EGNAS: Efficient Graph Neural Architecture Search using Monte Carlo Tree Search and Prediction Network
Code for paper:  
> [Efficient Graph Neural Architecture Search using Monte Carlo Tree Search and Prediction Network](https://www.sciencedirect.com/science/article/abs/pii/S0957417422019340)  
> TianJin Deng, Jia Wu   
> *Expert Systems with Applications 2022*

### Abstract  
Graph Neural Networks (GNNs) have emerged recently as a powerful way of dealing with
non-Euclidean data on graphs, such as social networks and citation networks. 
Despite their success, obtaining optimal graph neural networks requires immense
manual work and domain knowledge. Inspired by the strong searching capability of 
neural architecture search in CNN, a few attempts automatically search optimal 
GNNs that rival the best human-invented architectures. However, existing Graph 
Neural Architecture Search (GNAS) approaches face two challenges: 1) Sampling 
GNNs across the entire search space results in low search efficiency, 
particularly in large search spaces. 2) It is pretty costly to evaluate GNNs by 
training architectures from scratch. To overcome these challenges, 
this paper proposes an Efficient Graph Neural Architecture Search (EGNAS) method based on 
Monte Carlo Tree Search (MCTS) and a prediction network. Specifically, 
EGNAS first uses MCTS to recursively partition the entire search space into 
good or bad search regions. Then, the reinforcement learning-based search 
strategy (also called the agent) is applied to sample GNNs in those good search 
regions, which prevents overly exploring complex architectures and 
bad-performance regions, thus improving sampling efficiency. 
To reduce the evaluation cost, we use a prediction network to estimate the 
performance of GNNs. We alternately use ground-truth accuracy (by training GNNs 
from scratch) and prediction accuracy (by the prediction network) to update the 
search strategy to avoid inaccuracies caused by long-term use of the prediction 
network. Furthermore, to improve the training efficiency and stability, 
the agent is trained by a variant of Proximal Policy Optimization.

### Requirements  
Ensuring that Pytorch 1.1.0 and cuda 9.0 are installed. Then run:
```bash
pip install -r requirements.txt
```

### Running the model
If you want to use the model EGNAS-NP, please run:
```bash
python egnas/main.py --dataset Cora --search_samples 2000 --is_predictor 0 --search_strategy PPO+MCTS --Cp 0.1 
python egnas/main.py --dataset Citeseer --search_samples 2000 --is_predictor 0 --search_strategy PPO+MCTS --Cp 0.1
python egnas/main.py --dataset Pubmed --search_samples 1000 --is_predictor 0 --search_strategy PPO+MCTS --Cp 1.0
python egnas/main.py --dataset Photo --search_samples 1000 --is_predictor 0 --search_strategy PPO+MCTS --Cp 0.1
``` 

If you want to use the model EGNAS, please run:
```bash
python egnas/main.py --dataset Cora --search_samples 2000 --is_predictor 1 --search_strategy PPO+MCTS --Cp 0.1 
python egnas/main.py --dataset Citeseer --search_samples 2000 --is_predictor 1 --search_strategy PPO+MCTS --Cp 0.1
python egnas/main.py --dataset Pubmed --search_samples 1000 --is_predictor 1 --search_strategy PPO+MCTS --Cp 1.0
python egnas/main.py --dataset Photo --search_samples 1000 --is_predictor 1 --search_strategy PPO+MCTS --Cp 0.1
```

### Acknowledgements
To implement this repo, we refer to the following code:
[GraphNAS](https://github.com/GraphNAS/GraphNAS) and
[LaNAS](https://github.com/facebookresearch/LaMCTS)

### Citation
If you find our work useful, please consider citing it:
```python
@article{deng2023efficient,
  title={Efficient graph neural architecture search using Monte Carlo Tree search and prediction network},
  author={Deng, TianJin and Wu, Jia},
  journal={Expert Systems with Applications},
  volume={213},
  pages={118916},
  year={2023},
  publisher={Elsevier}
}
```


