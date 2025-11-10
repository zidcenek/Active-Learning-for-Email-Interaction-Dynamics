[![License: CC-4](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

# Active Recommendation for Email Outreach Dynamics

Authors: Čeněk Žid, Rodrigo Alves, Pavel Kordík

Affiliation: FIT CTU, Czech Technical University in Prague

This is an official repository for the paper [*Active Recommendation for Email Outreach Dynamics*.](https://dl.acm.org/doi/10.1145/3746252.3760832)

Check the public dataset at [HuggingFace (XCampaign Dataset)](https://huggingface.co/datasets/zidcenek/XCampaignDataset)

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

---
# XCampaign Dataset
## Introduction
This document describes the Mailprofiler's **XCampaign Dataset** -- provided by Mailprofiler; 
[XCampaign](https://xcampaign.info/switzerland-en/) represents an email campaign management platform. The dataset was 
used in our CIKM 2025 paper *Active Recommendation for Email Outreach Dynamics*. The dataset captures user-level 
interactions with periodic marketing mailshots, including whether an email was opened and the time-to-open (TTO).

## How to Use and Cite

The XCampaign Dataset is made available under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

This license allows you to share and adapt the dataset for any purpose, **including commercial use**, as long as you provide appropriate credit.

If you use this dataset in your work, please **cite the following paper**, which introduced the dataset:

### Plain Text Citation
> Čeněk Žid, Rodrigo Alves, and Pavel Kordík. 2025. Active Recommendation for Email Outreach Dynamics. In *Proceedings 
> of the 34th ACM International Conference on Information and Knowledge Management (CIKM '25)}*. Association for 
> Computing Machinery, New York, NY, USA, 5540–5544. https://doi.org/10.1145/3746252.3760832

### BibTeX Citation
```bibtex
@inproceedings{10.1145/3746252.3760832,
  author = {\v{Z}id, \v{C}en\v{e}k and Kord\'{\i}k, Pavel and Alves, Rodrigo},
  title = {Active Recommendation for Email Outreach Dynamics},
  year = {2025},
  isbn = {9798400720406},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3746252.3760832},
  doi = {https://doi.org/10.1145/3746252.3760832},
  booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
  pages = {5540–5544},
  numpages = {5},
  keywords = {email outreach, reinforcement learning, shallow autoencoder},
  location = {Seoul, Republic of Korea},
  series = {CIKM '25}
}
```

You can download the dataset using Hugging Face's [datasets library](https://huggingface.co/datasets/xcampaign/xcampaign-dataset).
```python
dataset = load_dataset("zidcenek/XCampaignDataset")
```

## Dataset and Fields
The **XCampaign Dataset** includes the following fields:
- `mailshot_id`: (or template id) identifier of the mailshot campaign
- `user_id`: anonymized recipient identifier
- `opened`: binary label (\(1\) if opened, \(0\) otherwise)
- `time_to_open`: time delta between send and open (a parseable string of a timedelta `0 days 09:39:32`)

## Global Statistics
All statistics below are computed from the full dataset.
- Rows: 14,908,085; Users: 131,918; Mailshots: 160
- Global open rate: 9.09%
- Per-mailshot open rate: $9.13\% \pm 3.58\%$
- Per-user open rate: mean $12.33\% \pm 20.46\%$
- Time-to-open (opened only): mean 1d 17h 25m; median 6h 25m
- Fraction opened within 1h: 25.9%; within 24h: 71.2%; within 7d: 93.0%
- Sent to users at each mailshot: $93,175 \pm 19,162$
- Item \(\times\) User interaction matrix density: 70.63%

![Global open rate and distribution of per-user open rates.](./assets/user_open_rate_hist.png)
Global open rate and distribution of per-user open rates.

## Time to Open (TTO)
Time-to-open is heavy-tailed: while the median is about 6.4 hours, most opens occur within a week. Specifically, 
93.0\% of opens arrive within 7 days, so 7.0\% arrive later than 7 days. The plots below are truncated at 7 days to 
emphasize the main mass of the distribution. The CDF and histogram are shown in Figure~\ref{fig:tto}.

![Distribution of time-to-open for opened emails.](./assets/time_to_open_hist.png)
Distribution of time-to-open for opened emails.

![CDF of time-to-open for opened emails.](assets/time_to_open_cdf.png)
CDF of time-to-open for opened emails.

Distribution (left) and CDF (right) of time-to-open for opened emails.

The heavy-tailed TTO suggests robust objectives and appropriate censoring strategies. The two user segments motivate 
segment-aware priors and exploration strategies; mailshot-level heterogeneity motivates per-mailshot features or random effects.

## Acknowledgements
Čeněk Žid's research was supported by the Grant Agency of the Czech Technical University (SGS20/213/OHK3/3T/18). 
We warmly thank *Mailprofiler* for providing the dataset for this research.

<p align="center">
  <a href="https://fit.cvut.cz/en" target="_blank">
    <img src="assets/logo-fit-en-modra.jpg" alt="FIT CTU" height="60"/>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://xcampaign.info/switzerland-en/" target="_blank">
    <img src="assets/Xcampaign_logo.svg" alt="XCampaign" height="60"/>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.recombee.com/" target="_blank">
    <img src="assets/recombee_logo.png" alt="Recombee" height="60"/>
  </a>
</p>

## License
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

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



