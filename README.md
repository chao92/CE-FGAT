
# Camouflaged Poisoning Attack on Graph Neural Networks with Limited Budget
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
## Abstract
Graph neural networks (GNNs) have enabled the automation of many web applications that entail node classifi-
cation on graphs, such as scam detection in social media, and event prediction in service networks. Nevertheless,
recent studies revealed that the GNNs are vulnerable to
adversarial attacks, where feeding GNNs with poisoned data at
training time can lead them to yield catastrophically devastative
test accuracy. This finding heats up the frontier of attacks and
defenses against GNNs. Despite considerable progress, the prior
studies mainly posit that the adversaries can enjoy free access
to manipulate the original graph, while obtaining such access
could be too costly in practice. To fill this gap, we propose
a novel attacking paradigm, named Cost-Effective Fake node
camouflaging with Generative-Adversarial Training (CE-FGAT),
which aims to craft a set of fake nodes in a generative-adversarial
training regime with a control of attack budget Three key ideas
of implementing CE-FGAT are as follows. First, we propose a
protocol to measure the budget of succeeding the poison attack
at certain rate. Second, these nodes carry malicious features
and can poison the victim GNN by passing their malicious
messages to the original graph via learned topological structures,
such that they 1) maximize the devastation of classification
accuracy (i.e., global attack) or 2) enforce the victim GNN
to misclassify a targeted node set into prescribed classes (i.e.,
target attack). Third, we use Generative Adversarial Networks to
camouflage the fake nodes, which helps to minimize the likelihood
of detection by off-the-shelf network intrusion detectors, such as
anomaly detection algorithms. We benchmark our experiments
on four real-world graph datasets, and the results substantiate the
viability, effectiveness, and stealthiness of our proposed poisoning
attack approach. Optimal attack budgets trading off for high
poisoning success rates are suggested for various threat model
setups
## Requirements
This code was tested on Linux(Ubuntu) and macOS
```
conda create --name CEFGAT python=3.7.10
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==2.0.6 -f https://data.pyg.org/whl/torch-1.6.0%2Bcpu.html
pip install torch-sparse==0.6.9 -f https://data.pyg.org/whl/torch-1.6.0%2Bcpu.html
pip install torch-geometric==1.7.0
pip install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.6.0%2Bcpu.html
pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.6.0%2Bcpu.html
conda install -c conda-forge pyod
```

## Run
```angular2html
conda activate CEFGAT
zsh run_global_attack_on_Cora.sh
zsh run_target_attack_on_Cora.sh
```
