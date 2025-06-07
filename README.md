# Active Recommendation for Email Outreach Dynamics

Authors: anonymous (under a double-blind review)

## Installation
Pre-requisites: `Python 3.11`

`pip install -r requirements.txt`

## Running Experiments
1. Add data to `PROJECT_ROOT/data/subsets/sender_mails_{NAME_OF_DATASET}.parquet`
   - the parquet file should contain a `pd.DataFrame` with the following columns: `[user_id, mailshot_id, opened, time_to_open]`
2. Create a config for grid search
   - Option A: manually in `PROJECT_ROOT/experiments/{NAME_OF_DATASET}/{EXPERIMENT_METHOD}/{VERSION}/grid_search_config.json`
       (there is an example in [grid_search_config.json](experiments/1/contextual_bandit/20250606-111137/grid_search_config.json))
   - Option B: using script: e.g. `python3 ./experiment_utils/create_grid_search_config.py --model contextual_bandit --sender_id 1;`
3. Run a grid search:
   - Run the grid search script `run_grid_search.py` e.g. 
   `python3 ./experiment_utils/run_grid_search.py --model contextual_bandit --sender_id 1 --version 20250606-111137 --n_jobs 1 --n_samples 1 --split_sizes 5 10;`
4. Run experiments on test set:
   - Create a test config in `PROJECT_ROOT/experiments/test_set/{NAME_OF_DATASET}/{EXPERIMENT_METHOD}/{VERSION}/config.json`
   - Run test script:
   ```python3 -u experiment_utils/test_any_model.py --model contextual_bandit --sender_id 1 --experiment_version 20250524-152200 --repetitions 10 --split_sizes 10;```


## Our Method
Hyperparameters of the autoencoder We perform a grid search for the autoencoder validated 
with fixed parameters for the contextual model. Hyperparameters of the contextual model With a fixed model from Step 1, 
we perform another grid search focusing on hyperparameters of the contextual model with Thompson Sampling.
For our method, we search for $d \in \{8, 12, 16\}$, $b \in \{12,24, 48\}$, $\alpha \in \{0.1, 0.2, 0.3\}$, 
and the trade-off parameter $G \in \{10^0, \cdots, 10^4\}$. For autoencoder training, we use Adam optimizer, weight decay 
in $wd\in\{10^{-4}, 10^{-5}\}$, learning rate $lr \in \{0.001, 0.003, 0.01\}$, number of epochs $e \in \{20, 30\}$, 
and exponential learning rate decay of $0.97$.
In the active learning phase, we do not consider the exact time to open for the user. 
However, we sample an exponential distribution, simulating user behavior with 
$\lambda_{tto}=1/time\_to\_open$ for each combination of user-template separately. 

## Baseline Overview

Our experiments were performed on an
anonymized dataset (which we make publicly available to the community) containing 131,918 users, 160 templates, and 14,908,085
email sendings observed over the course of 12 months (with $T=48$).
The open rate is 9.1%, resulting in a matrix density of approximately
70.6%. For evaluation, we split the dataset into three disjoint parts:
the last 10 templates for the test set, the preceding 5 for validation,
and the remaining templates for training.
We compare our method with eight baselines, each with its own
hyperparameter tuning. **FMFC-DB** [1]: $\epsilon\in\{0.01, 0.03, 0.05, 0.08\}$,
learning rate in $lr\in\{0.01,0.05\}$, epochs in $e\in\{40,60\}$, feature
dimension $d\in\{4,6\}$, and sample rate $s=1.0$. **FMPSAL** and
**FMRSAL** [1]: latent dimension $k\in\{16, 32, 64\}$, learning rate
in$lr \in \{0.001, 0.005, 0.01\}$, $e \in \{20, 30, 40\}$, and bootstrap ratio $s \in \{0.3, 0.5, 0.7\}$. 
**FactorUCB** [2] latent dimension
$k \in \{16, 32, 64\}$, regularization $\lambda \in \{10^{-3}, 10^{-2}, 10^{-1}\}$, 
epochs in $\{10, 20, 30\}$, and exploration factor $\alpha \in \{0.1, 0.5, 1.0\}$.  
**MBRL** [3]: discount factor $\gamma \in \{0.7, 0.8, 0.9\}$ and parameter 
$\kappa \in \{0.2, 0.4, 0.6, 0.8\}$. **DDQN** [4]: Layer sizes 
$d_{\text{user}} \in \{32, 64, 128\}$, embedding size 
$d_{\text{template}} \in \{8, 16\}$, hidden dimensions 
$d_{\text{hidden}} \in \{16, 32\}$, epochs $e \in \{5, 10, 20\}$, and 
discount factor $\gamma \in \{0.25, 0.4, 0.7, 0.9\}$. Since side information is not
available, we use trainable user and template embeddings. As our
model is based on Thompson Sampling (TS) [5], we perform an
ablation by using the vanilla algorithm, setting 𝛼 and 𝛽 according
to the observed counts of openings. Random: an algorithm that
selects users uniformly at random.


## References
[1] Yu Zhu, Jinghao Lin, Shibi He, Beidou Wang, Ziyu Guan, Haifeng Liu, and Deng
Cai. 2019. Addressing the item cold-start problem by attribute-driven active
learning. IEEE Transactions on Knowledge and Data Engineering 32, 4 (2019),
631–644.

[2] Huazheng Wang, Qingyun Wu, and Hongning Wang. 2017. Factorization bandits
for interactive recommendation. In Proceedings of the AAAI conference on artificial
intelligence, Vol. 31.

[3] Conor O’Brien, Huasen Wu, Shaodan Zhai, Dalin Guo, Wenzhe Shi, and
Jonathan J Hunt. 2022. Should I send this notification? Optimizing push notifications 
decision making by modeling the future. arXiv preprint arXiv:2202.08812
(2022).

[4] Yiping Yuan, Ajith Muralidharan, Preetam Nandy, Miao Cheng, and Prakruthi
Prabhakar. 2022. Offline Reinforcement Learning for Mobile Notifications.
arXiv:2202.03867 [cs.LG] https://arxiv.org/abs/2202.03867

[5] Daniel J Russo, Benjamin Van Roy, Abbas Kazerouni, Ian Osband, Zheng Wen,
et al. 2018. A tutorial on thompson sampling. Foundations and Trends® in
Machine Learning 11, 1 (2018), 1–96.



