# Provably Personalized and Robust Federated Learning
This repository contains the code for the paper [Provably Personalized and Robust Federated Learning](https://openreview.net/forum?id=B0uBSSUy0G) by Mariel Werner, Lie He, Michael Jordan, Martin Jaggi, Sai Praneeth Karimireddy.

## Abstract
Clustering clients with similar objectives and learning a model per cluster is an intuitive and interpretable approach to personalization in federated learning. However, doing so with provable and optimal guarantees has remained an open challenge. In this work, we formalize personalized federated learning as a stochastic optimization problem. We propose simple clustering-based algorithms which iteratively identify and train within clusters, using local client gradients. Our algorithms have optimal convergence rates which asymptotically match those obtained if we knew the true underlying clustering of the clients, and are provably robust in the Byzantine setting where some fraction of the clients are malicious.

## Get Started
To run the code, you'll need to install requirements: `pip install -r requirements.txt`. 

Code to reproduce the results in the paper:
- ./plot.ipynb
    - Code to generate Figure 2 & 3.
- mnist/
    - Code to generate Table 1.
- mnist_byz/
    - Code to generate Figure 6.
- cifar10/
    - Code to generate Figure 4.
- cifar100/
    - Code to generate Figure 5.
- synthetic/
    - Code to generate Figure 7. Visualization: ExploreL.ipynb.
 
Cached results for synthetic experiments can be found [here](https://drive.google.com/file/d/1-jkgcCVeZGnOXGaKnXB6VUZ0Hz_Dee_G/view?usp=sharing).

## Contact
If you have any questions or concerns about code, please contact us: `liam.he15@gmail.com`.


## Citation
If you find this repo useful in your research, please consider citing our paper as follows:
```
@article{
    werner2023provably,
    title={Provably Personalized and Robust Federated Learning},
    author={Mariel Werner and Lie He and Michael Jordan and Martin Jaggi and Sai Praneeth Karimireddy},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2023},
    url={https://openreview.net/forum?id=B0uBSSUy0G},
    note={}
}
```
