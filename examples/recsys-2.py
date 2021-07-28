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
# TODO lasciare i rating come sono e creare una copia dei rating tra 0 e 1 che rispecchi la fuzzy logic
# TODO la matrice dei rating sparsa ora deve contenere i rating in fuzzi e anche la similarita' tiene conto di quei ratings


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
    # convert ratings to fuzzy logic ratings 1 -> 0.2, 2 -> 0.4, 3 -> 0.6, 4 -> 0.8, 5 -> 1.0
    ratings['rating'] /= 5

    # create test_set
    # TODO in una versione successiva bisogna assicurarsi che che se un rating e' unico nel dataset, deve comparire solo durante il training e non solo durante il test
    # 20% of ratings of every user are put in test set
    g = ratings.groupby('user')
    ratings_test = pd.DataFrame(columns=['user', 'item', 'rating'])
    for _, group in g:
        n_ratings_to_remove = np.round(len(group) * 20 / 100)
        ratings_to_remove = group.sample(n=int(n_ratings_to_remove))
        ratings_test = ratings_test.append(ratings_to_remove)
        ratings = ratings.drop(ratings_to_remove.index)

    ratings = ratings.reset_index(drop=True)
    # random shuffle of test ratings
    ratings_test = ratings_test.sample(frac=1)
    ratings_test = ratings_test.reset_index(drop=True)

    # build sparse user-item matrix, the sparse matrix has a 1 if user has interacted with the item, 0 otherwise
    # even if the user has rated badly an item, there will be a 1 in the sparse matrix for that user-item pair
    g = ratings.groupby('user')
    rows, cols, values = [], [], []
    for _, group in g:
        # code to avoid to count for zeros
        # non_zeros = group['rating'].to_numpy().nonzero()[0]
        # rows.extend(group['user'].to_numpy()[non_zeros])
        # cols.extend(group['item'].to_numpy()[non_zeros])
        cols.extend(list(group['item']))
        rows.extend(list(group['user']))
        values.extend(list(group['rating']))

    # the user-item matrix does not contain test interactions
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

    return ratings.to_numpy(), ratings_test.to_numpy(), user_item_matrix, items_info.to_numpy(), users_info.to_numpy()


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
        self.first_layer = torch.nn.Linear(54, 32)
        self.second_layer = torch.nn.Linear(32, 16)
        self.third_layer = torch.nn.Linear(16, 1)

    def forward(self, inputs):
        x = torch.cat(inputs, dim=1)
        x = self.elu(self.first_layer(x))
        x = self.elu(self.second_layer(x))
        return self.sigmoid(self.third_layer(x))


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
            yield data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(np.float32)


def main():
    # prepare dataset for recommendation
    ratings, ratings_test, u_i_matrix, items, users = prepare_dataset()

    # create DataLoader for the training and testing of the model
    train_loader = DataLoader(ratings, 256, True)
    test_loader = DataLoader(ratings_test, 256)

    # convert dataset to tensors to properly work with LTN
    u_i_matrix = torch.tensor(u_i_matrix.todense()).to(ltn.device)  # this problem has to be solved, for the moment it could work since
    # data is small, the problem is that csr matrix does not work well with PyTorch tensors and PyTorch sparse is still
    # in beta
    items = torch.tensor(items).to(ltn.device)
    users = torch.tensor(users).to(ltn.device)

    # create Likes, Sim and Eq predicates
    likes = ltn.Predicate(Likes()).to(ltn.device)
    # this measures the similarity between two users' behavioral vectors (vectors containing users' preferences)
    sim = ltn.Predicate(lambda_func=lambda args: torch.nn.functional.cosine_similarity(args[0], args[1], dim=1, eps=1e-8)).to(ltn.device)
    # this measures the similarity between two truth values in [0., 1.]
    eq = ltn.Predicate(lambda_func=lambda args: torch.exp(-torch.norm(args[0] - args[1], dim=1)))

    # create functions that return users and items information given the indexes
    # given a user index, it returns the features of the user (demographic information)
    get_u_features = ltn.Function(lambda_func=lambda u: torch.flatten(users[u[0]], start_dim=1)).to(ltn.device)
    # given an item index, it returns the features of the item (movie's year and genres)
    get_i_features = ltn.Function(lambda_func=lambda i: torch.flatten(items[i[0]], start_dim=1)).to(ltn.device)
    # given a user index, it returns his/her historical behaviour (rating vector)
    get_u_ratings = ltn.Function(lambda_func=lambda u: torch.flatten(u_i_matrix[u[0]], start_dim=1)).to(ltn.device)

    # Operators to build logical axioms
    Forall = ltn.WrapperQuantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantification_type="forall")
    And = ltn.WrapperConnective(ltn.fuzzy_ops.AndProd())
    Implies = ltn.WrapperConnective(ltn.fuzzy_ops.ImpliesGoguenStrong())
    Not = ltn.WrapperConnective(ltn.fuzzy_ops.NotStandard())

    formula_aggregator = ltn.fuzzy_ops.AggregPMeanError(p=2)

    # here, we define the knowledge base for the recommendation task
    def axioms(uid, iid, rate):
        u1 = ltn.variable('u1', uid)
        u2 = ltn.variable('u2', uid)
        i = ltn.variable('i', iid)
        r = ltn.variable('r', rate)

        f1 = Forall(ltn.diag([u1, i, r]), eq([likes([get_u_features(u1), get_i_features(i)]), r]))
        f2 = Forall([u1, u2, i], Implies(
                                        And(
                                            sim([get_u_ratings(u1), get_u_ratings(u2)]),
                                            likes([get_u_features(u1), get_i_features(i)])
                                        ),
                                        likes([get_u_features(u2), get_i_features(i)])
                                        ))
        '''
        f4 = Forall([u1, u2, i1], Implies(
            And(
                sim([get_u_features(u1), get_u_features(u2)]),
                likes([get_u_features(u1), get_i_features(i1)])
            ),
            likes([get_u_features(u2), get_i_features(i1)])
        ))
        f5 = Forall([i1, i2, u1], Implies(
            And(
                sim([get_i_features(i1), get_i_features(i2)]),
                likes([get_u_features(u1), get_i_features(i1)])
            ),
            likes([get_u_features(u1), get_i_features(i2)])
        ))'''

        axioms = torch.stack([f1, f2])

        sat_level = formula_aggregator(axioms, dim=0)

        return sat_level

    # training of the LTN model for recommendation
    optimizer = torch.optim.Adam(likes.parameters(), lr=0.01)

    for epoch in range(100):
        train_loss = 0.0
        mean_sat = 0.0
        for batch_idx, (uid, iid, rate) in enumerate(train_loader):
            optimizer.zero_grad()
            sat_agg = axioms(uid, iid, rate)
            mean_sat += sat_agg.item()
            loss = 1. - sat_agg
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        mean_sat = mean_sat / len(train_loader)

        # test step
        mean_sat_test = 0.0
        for batch_idx, (uid, iid, rate) in enumerate(test_loader):
            sat_agg = axioms(uid, iid, rate)
            mean_sat_test += sat_agg.item()

        mean_sat_test = mean_sat_test / len(test_loader)

        logger.info(
            " epoch %d | loss %.4f | Train Sat %.3f | Mean Sat Test %s ",
            epoch, train_loss, mean_sat, mean_sat_test)


if __name__ == "__main__":
    main()