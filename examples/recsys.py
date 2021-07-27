import numpy as np
import torch
import pandas as pd
from scipy.sparse import csr_matrix
import ltn
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# TODO risolvere il problema della incompatibilita' tra matrici sparse e PyTorch
# TODO vedere come tenere la matrice sparsa e poi costruirsi a batch il tensor normale


def prepare_dataset():
    ratings = pd.read_csv("datasets/ml-100k/u.data", sep='\t', header=None)
    users_info = pd.read_csv("datasets/ml-100k/u.user", sep='|', header=None)
    items_info = pd.read_csv("datasets/ml-100k/u.item", sep='|', encoding='iso-8859-1', header=None)
    users_info.columns = ['user', 'age', 'gender', 'occupation', 'zip']
    items_info.columns = ['item', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown', 'Action',
                          'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    ratings.columns = ['user', 'item', 'rating', 'timestamp']
    ratings = ratings.drop('timestamp', axis=1)
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
    user_item_matrix = csr_matrix((values, (rows, cols)), (n_users, n_items))

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

    return ratings.to_numpy(), user_item_matrix, items_info.to_numpy(), users_info.to_numpy()


class Likes(torch.nn.Module):
    """
    This is the Likes predicate of the recommender system. It takes as input one user-item pair and returns a value
    in [0., 1.]. The higher the value the better the compatibility between the user and the item. Users and items are
    given to this predicate in the form of features. For the user the features are the demographic information, while
    for the items are the content information.
    """
    def __init__(self):
        super(Likes, self).__init__()
        self.elu = torch.nn.ELU()
        self.sigmoid = torch.nn.Sigmoid()
        self.first_layer = torch.nn.Linear(54, 128)
        self.second_layer = torch.nn.Linear(128, 64)
        self.third_layer = torch.nn.Linear(64, 32)
        self.fourth_layer = torch.nn.Linear(32, 16)
        self.fifth_layer = torch.nn.Linear(16, 1)

    def forward(self, inputs):
        x = torch.cat(inputs, dim=1)
        x = self.elu(self.first_layer(x))
        x = self.elu(self.second_layer(x))
        x = self.elu(self.third_layer(x))
        x = self.elu(self.fourth_layer(x))
        return self.sigmoid(self.fifth_layer(x))


# this is a standard PyTorch DataLoader to load the dataset for the training and testing of the model
# for each batch, this loader returns the positive and negative examples contained in the batch
# a positive example is a user-item pair for which the user has given an high rating to the item, while a negative
# example is a user-item pair for which the user has given a low rating to the item
class DataLoader(object):
    def __init__(self,
                 data,
                 batch_size=1,
                 shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data = self.data[idxlist[start_idx:end_idx]]
            index_positives = np.where(data[:, 2] == 1)[0]
            index_negatives = np.where(data[:, 2] == 0)[0]
            positives = data[index_positives, :2]
            negatives = data[index_negatives, :2]
            batch_users = data[:, 0]
            batch_items = data[:, 1]
            yield positives, negatives, batch_users, batch_items


def main():
    # prepare dataset for recommendation
    ratings, u_i_matrix, items, users = prepare_dataset()

    # create DataLoader for the training of the model
    train_loader = DataLoader(ratings, 128, True)

    # convert dataset to tensors to properly work with LTN
    u_i_matrix = torch.tensor(u_i_matrix.todense()).to(ltn.device)  # this problem has to be solved, for the moment it could work since
    # data is small, the problem is that csr matrix does not work well with PyTorch tensors and PyTorch sparse is still
    # in beta
    items = torch.tensor(items).to(ltn.device)
    users = torch.tensor(users).to(ltn.device)

    # create Likes and Sim predicates
    likes = ltn.Predicate(Likes()).to(ltn.device)
    # this measures the similarity between two users' behavioral vectors (vectors containing users' preferences)
    sim = ltn.Predicate(lambda_func=lambda args: torch.nn.functional.cosine_similarity(args[0], args[1], dim=1, eps=1e-8)).to(ltn.device)

    # create functions that return users and items information given the indexes
    # given a user index, it returns the features of the user (demographic information)
    get_u_features = ltn.Function(lambda_func=lambda u: users[u[0]]).to(ltn.device)
    # given an item index, it returns the features of the item (movie's year and genres)
    get_i_features = ltn.Function(lambda_func=lambda i: items[i[0]]).to(ltn.device)
    # given a user index, it returns his/her historical behaviour (rating vector)
    get_u_ratings = ltn.Function(lambda_func=lambda u: u_i_matrix[u[0]]).to(ltn.device)

    # Operators to build logical axioms
    Forall = ltn.WrapperQuantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantification_type="forall")
    And = ltn.WrapperConnective(ltn.fuzzy_ops.AndProd())
    Implies = ltn.WrapperConnective(ltn.fuzzy_ops.ImpliesGoguenStrong())
    Not = ltn.WrapperConnective(ltn.fuzzy_ops.NotStandard())

    formula_aggregator = ltn.fuzzy_ops.AggregPMeanError(p=2)

    # here, we define the knowledge base for the recommendation task
    def axioms(positive_pairs, negative_pairs, batch_users, batch_items):
        u1 = ltn.variable('u1', batch_users, add_batch_dim=False)
        u2 = ltn.variable('u2', batch_users, add_batch_dim=False)
        i = ltn.variable('i', batch_items, add_batch_dim=False)
        u_pos = ltn.variable('u_pos', positive_pairs[:, 0], add_batch_dim=False)
        i_pos = ltn.variable('i_pos', positive_pairs[:, 1], add_batch_dim=False)
        u_neg = ltn.variable('u_neg', negative_pairs[:, 0], add_batch_dim=False)
        i_neg = ltn.variable('i_neg', negative_pairs[:, 1], add_batch_dim=False)

        axioms = [
            Forall(ltn.diag([u_pos, i_pos]), likes([get_u_features(u_pos), get_i_features(i_pos)])),
            Forall(ltn.diag([u_neg, i_neg]), Not(likes([get_u_features(u_neg), get_i_features(i_neg)]))),
            Forall([u1, u2, i], Implies(
                                        And(
                                            sim([get_u_ratings(u1), get_u_ratings(u2)]),
                                            likes([get_u_features(u1), get_i_features(i)])
                                        ),
                                        likes([get_u_features(u2), get_i_features(i)])
                                        ))
        ]

        axioms = torch.stack(axioms)

        sat_level = formula_aggregator(axioms, dim=0)
        return sat_level

    # training of the LTN model for recommendation
    optimizer = torch.optim.Adam(likes.parameters(), lr=0.001)

    for epoch in range(100):
        train_loss = 0.0
        mean_sat = 0.0
        for batch_idx, (positive_pairs, negative_pairs, batch_users, batch_items) in enumerate(train_loader):
            optimizer.zero_grad()
            sat_agg = axioms(positive_pairs, negative_pairs, batch_users, batch_items)
            mean_sat += sat_agg.item()
            loss = 1. - sat_agg
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        mean_sat = mean_sat / len(train_loader)

        # we print metrics every epoch of training
        logger.info(" epoch %d | loss %.4f | Train Sat %.3f ", epoch, train_loss, mean_sat)



if __name__ == "__main__":
    main()