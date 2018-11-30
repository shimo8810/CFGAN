"""
dataset
"""
from pathlib import Path
import pandas as pd
import numpy as np
from chainer.dataset import dataset_mixin


def get_movie_lens_100k(train_path, test_path, info_path, user_path, item_path):

    info_df = pd.read_csv(info_path, delim_whitespace=True, header=None)
    train_df = pd.read_csv(train_path, delim_whitespace=True, header=None)
    test_df = pd.read_csv(test_path, delim_whitespace=True, header=None)
    user_df = pd.read_csv(user_path, delimiter='|', header=None)
    item_df = pd.read_csv(item_path, delimiter='|', header=None)

    train = MovieLensDataset(train_df, info_df, user_df)
    test = MovieLensDataset(test_df, info_df, user_df)

    return train, test

class MovieLensDataset(dataset_mixin.DatasetMixin):
    """
    Dataset of MovieLens
    """
    def __init__(self, data_df, info_df, user_df):
        num_user, num_item, _ = info_df[0].values
        self.user_data = pd.get_dummies(user_df[[1, 2, 3]]).values.astype(np.float32)

        self.data = np.zeros((num_user, num_item)).astype(np.float32)
        for _, (user_id, item_id, rating, _) in data_df.iterrows():
            self.data[user_id-1, item_id-1] = rating

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        return self.data[i], self.user_data[i]


if __name__ == '__main__':
    _, test = get_movie_lens_100k('./dataset/u1.base', './dataset/u1.test',
                    './dataset/u.info', './dataset/u.user', './dataset/u.item')
    u, c = _[0]
    print(u.shape)
    print(c.shape)
