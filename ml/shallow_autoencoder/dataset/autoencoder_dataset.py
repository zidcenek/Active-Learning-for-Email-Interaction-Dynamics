import logging
from typing import Dict, Tuple, List, Optional, Self

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.data_loader_utils import load_data_for_sender

logger = logging.getLogger(__name__)

class AutoencoderDataset(Dataset):
    validation_size = 1
    test_size = 1

    def __init__(self, mails: pd.DataFrame, user_id_mapper: Dict[int, int] = None, remove_users_below_n_opens: int = 0, sender_id: int = None):
        assert all(column in mails.columns for column in ['mailshot_id', 'user_id', 'opened', 'time_to_open'])
        self.mails = mails
        self.sender_id = sender_id
        self._mailshot_embeddings: np.ndarray = np.array([])
        self._time_to_open: Dict[int, Dict[int, float]] = {}  # mailshot_id -> user_id -> time_to_open
        if user_id_mapper:
            self._user_id_mapper = user_id_mapper
        else:
            self._user_id_mapper: Dict[int, int] = {}
        self._validation_mask = None
        self._training_mask = None
        self._remove_users_with_no_opens(remove_users_below_n_opens)
        self._reverse_user_id_mapper: Dict[int, int] = {}
        self._mails_to_data()
        self._mails_to_time_to_open()
        self._user_clusters: Optional[pd.DataFrame] = None
        self._training_mask_recalc = 0
        logger.info(f"Dataset created with {len(self)} mailshots and {self.num_users} users")

    @property
    def num_users(self) -> int:
        return len(self._user_id_mapper)

    @property
    def average_opens(self) -> float:
        return self.mails['opened'].sum() / len(self)

    @property
    def reversed_user_id_mapper(self) -> Dict[int, int]:
        if not self._reverse_user_id_mapper:
            self._reverse_user_id_mapper = {v: k for k, v in self._user_id_mapper.items()}
        return self._reverse_user_id_mapper

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._mailshot_embeddings.shape

    @property
    def validation_mask(self) -> torch.tensor:
        if self._validation_mask is None:
            torch.manual_seed(42)
            mask = torch.zeros(self.shape)
            mask = mask + torch.bernoulli(torch.ones(self.shape) * 0.98)
            mask = mask.bool()
            self._validation_mask = mask
        return self._validation_mask

    @property
    def training_mask(self) -> torch.tensor:
        if self._training_mask is None:
            torch.manual_seed(42)
            mask = torch.zeros(self.shape)
            mask = mask + torch.bernoulli(torch.ones(self.shape) * 0.85)
            mask = mask.bool()
            # Combine with the validation mask
            mask = mask & self.validation_mask
            self._training_mask = mask
        return self._training_mask

    @property
    def mailshot_ids(self) -> List[int]:
        return sorted(self.mails['mailshot_id'].unique())

    @property
    def popularity_tensor(self) -> torch.Tensor:
        opens = self.mails.groupby('user_id')['opened'].sum().sort_index()
        count = self.mails.groupby('user_id')['opened'].count().sort_index()
        return torch.tensor(opens / count)

    @property
    def user_clusters(self) -> pd.DataFrame:
        """
        DEPRECATED
        Return the precalculated user clusters.
        @return: pd.DataFrame with user clusters
        """
        if self._user_clusters is None:
            try:
                raise FileNotFoundError
            except FileNotFoundError:
                self._user_clusters = self.mails[['user_id']].copy()
                self._user_clusters['cluster'] = 0
        return self._user_clusters


    def mailshot_user_indices(self, mailshot_id: int) -> List[int]:
        return self.mails[self.mails['mailshot_id'] == mailshot_id]['user_id'].unique()

    @classmethod
    def from_disk(cls, sender_id: int) -> Self:
        mails: pd.DataFrame = load_data_for_sender(sender_id)
        return cls(mails)

    @classmethod
    def from_disk_data_split(cls,
                             sender_id: int,
                             split_sizes: List[int],
                             remove_users_below_n_opens: int = 0
                             ) -> List[Self]:
        logger.info(f"Loading data for sender {sender_id}")
        mails: pd.DataFrame = load_data_for_sender(sender_id)
        # Drop mailshots with less mails than 100
        mails = mails.groupby('mailshot_id').filter(lambda x: len(x) >= 100)
        mails = mails.reset_index(drop=True)
        mails = mails[['mailshot_id', 'user_id', 'opened', 'time_to_open']]
        mails = mails.drop_duplicates(subset=['mailshot_id', 'user_id'])
        mails = mails.reset_index(drop=True)
        assert all(column in mails.columns for column in ['mailshot_id', 'user_id', 'opened', 'time_to_open'])
        mailshot_ids = mails['mailshot_id'].unique()
        mailshot_ids = sorted(mailshot_ids)
        # remap mailshot ids to autoincrementing integers
        mailshot_id_map = {mailshot_id: i for i, mailshot_id in enumerate(mailshot_ids)}
        mails['mailshot_id'] = mails['mailshot_id'].map(mailshot_id_map)
        mailshot_ids = sorted(mails['mailshot_id'].unique())
        first_size = len(mailshot_ids) - np.sum(split_sizes)
        split_sizes = [first_size] + split_sizes

        # Calculate the indices for the splits
        split_indices = np.cumsum(split_sizes)
        split_indices = [0] + split_indices.tolist()

        datasets = []
        for i in range(len(split_sizes)):
            split_mailshot_ids = mailshot_ids[split_indices[i]:split_indices[i+1]]
            split_mails = mails[mails['mailshot_id'].isin(split_mailshot_ids)].copy()
            if i == 0:
                train = cls(split_mails, remove_users_below_n_opens=remove_users_below_n_opens, sender_id=sender_id)
                datasets.append(train)
            else:
                datasets.append(cls(split_mails, train._user_id_mapper, sender_id=sender_id))

        return datasets

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the training, validation and test tensor for the given index.
        :param index:
        :return: masked data by the training mask, masked data by the training mask, full data
        """
        x = torch.tensor(self._mailshot_embeddings[index])
        return x * self.training_mask[index], x * self.validation_mask[index], x

    def __len__(self):
        return len(self._mailshot_embeddings)

    def _mails_to_data(self):
        # Mapper for user_id
        user_ids = self.mails['user_id'].unique()
        if self._user_id_mapper:
            # If a user mapper is provided -> remove users that are not in the mapper
            # the users that are not present in the current dataset will be treated with 0s
            logger.info("Using existing user id mapper")
            extra_user_ids = set(user_ids) - set(self._user_id_mapper.keys())
            reduced_user_ids = [user_id for user_id in user_ids if user_id not in extra_user_ids]
            logger.info(f"Reduced user ids: {len(reduced_user_ids)}, old user ids: {len(user_ids)}")
            user_ids = reduced_user_ids
        else:
            # If a user mapper is not provided -> create a new one
            user_ids = sorted(user_ids)
            user_id_map = {user_id: i for i, user_id in enumerate(user_ids)}
            self._user_id_mapper = user_id_map

        self.mails = self.mails[self.mails['user_id'].isin(user_ids)]  # TODO: resolve this in a better way
        self.mails['user_id'] = self.mails['user_id'].map(self._user_id_mapper)

        # For each mailshot we have to create a sequence of mails and their opened binary vector
        # The shape resembles the shape of the user_id_mapper (either the one provided or the one created)
        mailshot_embeddings = []
        for mailshot_id in self.mails['mailshot_id'].unique():
            feature_vector = np.zeros(len(self._user_id_mapper))
            # Fill in the opened mails
            opened = self.mails[(self.mails['mailshot_id'] == mailshot_id) & (self.mails['opened'] == 1) & (self.mails['user_id'].isin(self._user_id_mapper.values()))]
            feature_vector[opened['user_id']] = 1
            mailshot_embeddings.append(feature_vector)
        self._mailshot_embeddings = np.array(mailshot_embeddings).astype(np.float32)

    def _mails_to_time_to_open(self):
        # Calculate the average time to open for each user
        for mailshot_id in self.mails.mailshot_id.unique():
            mailshot_mails = self.mails[self.mails['mailshot_id'] == mailshot_id].copy()
            # Mean does not do anything here, there is only one value per user
            time_to_open = mailshot_mails.set_index('user_id')['time_to_open'].to_dict()
            # Time to float representing number of hours
            # Sample from exponential distribution to simulate time to open
            mailshot_tto = {
                user_id: np.random.exponential(abs(time_to_open.total_seconds() / 60))
                if time_to_open
                else np.inf
                for user_id, time_to_open in time_to_open.items()
            }
            self._time_to_open[mailshot_id] = mailshot_tto

    def number_of_opens_for_user(self, user_id: int) -> int:
        return self.mails[self.mails['user_id'] == user_id]['opened'].sum()

    def _remove_users_with_no_opens(self, threshold: int = 0):
        logger.info(f"Removing users with less than {threshold} opens, current number of users: {len(self.mails['user_id'].unique())}")
        users_with_opens = self.mails.groupby('user_id')['opened'].sum()
        users_with_opens = users_with_opens[users_with_opens >= threshold].index
        self.mails = self.mails[self.mails['user_id'].isin(users_with_opens)]
        logger.info(f"Removed users with no opens, new number of users: {len(self.mails['user_id'].unique())}")


    def select_opened_indices(self, mailshot_id: int, indices: List[int], time_to_open: float) -> List[int]:
        """
        Selects indices of users that opened the mail within the given time frame.
        """
        tto_users: Dict[int, float] = self._time_to_open[mailshot_id]
        selected = [i for i in indices if i in tto_users and tto_users[i] <= time_to_open]
        return selected

    def tensor_of_opens_of_mailshot(self, mailshot_id: int) -> torch.Tensor:
        """
        Returns a tensor of opens for the given mailshot_id.
        """
        return torch.tensor(self._mailshot_embeddings[mailshot_id])
