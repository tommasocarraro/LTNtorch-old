import numpy as np
import torch
import pandas as pd
from scipy.sparse import csr_matrix
import time
import datetime
import ltn

# TODO l'indice 266 degli item e' poco informativo, non sappiamo nulla di quel item


def prepare_dataset():
    ratings = pd.read_csv("datasets/ml-100k/u.data", sep='\t', header=None)
    users_info = pd.read_csv("datasets/ml-100k/u.user", sep='|', header=None)
    items_info = pd.read_csv("datasets/ml-100k/u.item", sep='|', encoding='iso-8859-1', header=None)
    users_info.columns = ['user', 'age', 'gender', 'occupation', 'zip']
    items_info.columns = ['item', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown', 'Action',
                          'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    ratings.columns = ['user', 'item', 'rating', 'timestamp']
    # remove item 267 from the dataset since it is not informative
    ratings = ratings[ratings['item'] != 267]
    items_info = items_info[items_info['item'] != 267]
    # shifting indexes above 267 of one position to the left in order to not have holes in the structure
    ratings['item'] = ratings['item'].apply(lambda x: x - 1 if x > 267 else x)
    items_info['item'] = items_info['item'].apply(lambda x: x - 1 if x > 267 else x)
    n_users = np.max(list(ratings['user']))
    n_items = np.max(list(ratings['item']))
    ratings['user'] -= 1
    ratings['item'] -= 1
    users_info['user'] -= 1
    items_info['item'] -= 1
    ratings['rating'][ratings['rating'] < 4] = 0
    ratings['rating'][ratings['rating'] >= 4] = 1

    # build sparse user-item matrix, the sparse matrix has a 1 if user has interacted with the item, 0 otherwise
    # even if the user has rated badly an item, there will be a 1 in the sparse matrix for that user-item pair
    g = ratings.groupby('user')
    rows, cols = [], []
    for _, group in g:
        # code to avoid to count for zeros
        # non_zeros = group['rating'].to_numpy().nonzero()[0]
        # rows.extend(group['user'].to_numpy()[non_zeros])
        # cols.extend(group['item'].to_numpy()[non_zeros])
        cols.extend(list(group['item']))
        rows.extend(list(group['user']))

    values = np.ones(len(rows))
    user_item_matrix = csr_matrix((values, (rows, cols)), shape=(n_users, n_items))

    # pre-processing items
    items_info = items_info.drop(columns=['item', 'imdb_url', 'video_release_date', 'title'])
    # get only the year from the date
    items_info['release_date'] = items_info['release_date'].apply(lambda x: int(x.split('-')[2]))
    # group years in decades
    items_info['release_date'] = items_info['release_date'].apply(lambda x: 0 if x < 1950 else (1 if x < 1960 else (
        2 if x < 1970 else (3 if x < 1980 else (4 if x < 1990 else 5))
    )))
    # Get one hot encoding of column 'realease_date'
    one_hot = pd.get_dummies(items_info['release_date'], prefix='year')
    # Drop column as it is now encoded
    items_info = items_info.drop('release_date', axis=1)
    # Join the encoded df
    items_info = items_info.join(one_hot)

    # pre-processing users
    # group age in significant groups
    users_info['age'] = users_info['age'].apply(lambda x: 0 if x <= 10 else (1 if x <= 20 else (
        2 if x <= 30 else (3 if x <= 40 else (4 if x <= 50 else (5 if x <= 60 else 6)))
    )))
    # Get one hot encoding of column 'age'
    one_hot = pd.get_dummies(users_info['age'], prefix='age')
    # Drop column as it is now encoded
    users_info = users_info.drop('age', axis=1)
    # Join the encoded df
    users_info = users_info.join(one_hot)
    # one-hot of gender
    users_info['gender'] = users_info['gender'].apply(lambda x: 0 if x == 'M' else 1)
    # one-hot occupation
    one_hot = pd.get_dummies(users_info['occupation'], prefix='occupation')
    # Drop column as it is now encoded
    users_info = users_info.drop('occupation', axis=1)
    # Join the encoded df
    users_info = users_info.join(one_hot)
    # remove 'user' column
    users_info = users_info.drop(columns=['user', 'zip'])

def main():
    prepare_dataset()
    # features = torch.tensor([9, 3, 4, 3])
    # u = ltn.variable('u', [0, 3, 1, 0])
    # f = ltn.Function(lambda_func=lambda x: features[x[0].long()])
    # print(f(u).free_variables)


if __name__ == "__main__":
    main()