import numpy as np
import torch
import pandas as pd
from scipy.sparse import csr_matrix
import ltn
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


def prepare_dataset():
    ratings = pd.read_csv("datasets/ml-100k/u.data", sep='\t', header=None)
    users_info = pd.read_csv("datasets/ml-100k/u.user", sep='|', header=None)
    items_info = pd.read_csv("datasets/ml-100k/u.item", sep='|', encoding='iso-8859-1', header=None)
    users_info.columns = ['user', 'age', 'gender', 'occupation', 'zip']
    items_info.columns = ['item', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown', 'Action',
                          'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                          'Western']
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
    # convert ratings to fuzzy logic ratings 1 -> 0.0, 2 -> 0.25, 3 -> 0.5, 4 -> 0.75, 5 -> 1.0
    ratings['rating'] -= 1
    ratings['rating'] /= 4

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

    # build sparse user-item matrices
    # the `seen` matrix has a 1 if user `u` has interacted with the item `i`, 0 otherwise
    # the `rate` matrix contains the ratings given by the users to the items of the dataset. Note that a 0 in this
    # matrix could be a missing rating or a bad rating given by the user
    g = ratings.groupby('user')
    rows, cols, values_rate = [], [], []
    for _, group in g:
        cols.extend(list(group['item']))
        rows.extend(list(group['user']))
        values_rate.extend(list(group['rating']))

    # create vector of values for the `seen` matrix
    values_seen = np.ones(len(rows))
    # note that the user-item matrices do not contain test interactions
    seen_matrix = torch.tensor(csr_matrix((values_seen, (rows, cols)), (n_users, n_items)).todense())
    rate_matrix = torch.tensor(csr_matrix((values_rate, (rows, cols)), (n_users, n_items)).todense())

    # here, we compute the similarities between each pair of users in the dataset
    # the result is a square matrix of similarities
    # similarity computation between two users:
    # 1. the cosine similarity between the `seen` vectors of the two users is computed. The `seen` vector contains 1
    # if the user has seen the item, 0 otherwise. This similarity should capture how much two users are similar based
    # on which items they have interacted with in the past;
    # 2. the `seen` vectors are multiplied to have ones where the users have interacted with the same items;
    # 3. this output vector is used as a mask to fetch the ratings of the two users for the items that they have both
    # interacted with in the past;
    # 4. the cosine similarity between the two masked rating vectors is computed. This similarity should capture how
    # much to users are similar based on the ratings that they have given to the items that both have interacted with;
    # 5. the final similarity is computed by a linear combination of the two similarities, with alpha parameter set to
    # 0.3 to give less weight to the similarity based on the interactions.
    alpha = 0.50
    sim_users = torch.ones((n_users, n_users))
    for i in range(seen_matrix.shape[0]):
        expanded_user_interactions = seen_matrix[i].expand(n_users, n_items)
        expanded_user_ratings = rate_matrix[i].expand(n_users, n_items)
        cos_sim_user_interactions = torch.nn.functional.cosine_similarity(expanded_user_interactions, seen_matrix,
                                                                          dim=1)
        user_shared_items = expanded_user_interactions * seen_matrix
        user_shared_ratings = expanded_user_ratings * user_shared_items
        other_users_shared_ratings = rate_matrix * user_shared_items
        cos_sim_user_ratings = torch.nn.functional.cosine_similarity(other_users_shared_ratings, user_shared_ratings,
                                                                     dim=1)
        final_cos_sim = alpha * cos_sim_user_interactions + (1 - alpha) * cos_sim_user_ratings
        sim_users[i] = final_cos_sim

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

    return ratings.to_numpy(), ratings_test.to_numpy(), seen_matrix, rate_matrix, sim_users, items_info.to_numpy(), \
           users_info.to_numpy()


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
    ratings, ratings_test, seen_matrix, rate_matrix, sim_users, items, users = prepare_dataset()

    like_embedding = torch.randn((users.shape[0], items.shape[0]), requires_grad=True).to(ltn.device)

    # create DataLoader for the training and testing of the model
    train_loader = DataLoader(ratings, 256, True)
    test_loader = DataLoader(ratings_test, 256)

    # convert dataset to tensors to properly work with LTN
    # seen_matrix = torch.tensor(seen_matrix.todense()).to(ltn.device)
    # rate_matrix = torch.tensor(rate_matrix.todense()).to(ltn.device)  # this problem has to be solved, for the moment it
    # could work since data is small, the problem is that csr matrix does not work well with PyTorch tensors and
    # PyTorch sparse is still in beta
    sim_users = sim_users.to(ltn.device)
    items = torch.tensor(items).to(ltn.device)
    users = torch.tensor(users).to(ltn.device)

    # create Likes, Sim and Eq predicates
    likes_nn = ltn.Predicate(Likes()).to(ltn.device)
    likes = ltn.Predicate(lambda_func=lambda args: torch.sigmoid(like_embedding[args[0], args[1]])).to(ltn.device)
    # this measures the similarity between two users' behavioral vectors (vectors containing users' preferences)
    sim = ltn.Predicate(lambda_func=lambda args: sim_users[args[0], args[1]]).to(ltn.device)
    # this measures the similarity between two truth values in [0., 1.]
    eq = ltn.Predicate(lambda_func=lambda args: torch.exp(-torch.norm(torch.unsqueeze(args[0] - args[1], 1), dim=1)))

    # create functions that return users and items information given the indexes
    # given a user index, it returns the features of the user (demographic information)
    get_u_features = ltn.Function(lambda_func=lambda u: users[u[0]]).to(ltn.device)
    # given an item index, it returns the features of the item (movie's year and genres)
    get_i_features = ltn.Function(lambda_func=lambda i: items[i[0]]).to(ltn.device)
    # given a user index, it returns his/her historical behaviour (rating vector)
    get_u_ratings = ltn.Function(lambda_func=lambda u: rate_matrix[u[0]]).to(ltn.device)
    # given a user index, it returns his/her historical behaviour (seen/unseen vector)
    get_u_interactions = ltn.Function(lambda_func=lambda u: seen_matrix[u[0]]).to(ltn.device)

    # Operators to build logical axioms
    Forall = ltn.WrapperQuantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantification_type="forall")
    And = ltn.WrapperConnective(ltn.fuzzy_ops.AndProd())
    Implies = ltn.WrapperConnective(ltn.fuzzy_ops.ImpliesGoguenStrong())
    Equiv = ltn.WrapperConnective(ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.AndProd(), ltn.fuzzy_ops.ImpliesGoguenStrong()))
    Not = ltn.WrapperConnective(ltn.fuzzy_ops.NotStandard())

    formula_aggregator = ltn.fuzzy_ops.AggregPMeanError(p=2)

    # here, we define the knowledge base for the recommendation task
    def axioms(uid, iid, rate):
        u1 = ltn.variable('u1', uid, add_batch_dim=False)
        u2 = ltn.variable('u2', uid, add_batch_dim=False)
        i = ltn.variable('i', iid, add_batch_dim=False)
        r = ltn.variable('r', rate, add_batch_dim=False)
        # TODO capire se queste similarita' sono organizzate bene all'interno del tensore

        axioms = [
            Forall(ltn.diag([u1, i, r]), Equiv(likes_nn([get_u_features(u1), get_i_features(i)]), r)),
            Forall([u1, i], Implies(likes_nn([get_u_features(u1), get_i_features(i)]), likes([u1, i]))),
            Forall([u1, u2, i],
                   Implies(
                       And(
                           sim([u1, u2]),
                           likes([u1, i])
                       ),
                       likes([u2, i])
                   ),
                   mask_vars=[u1, u2],
                   mask_fn=lambda mask_vars: ~torch.eq(mask_vars[0], mask_vars[1])
                   )
            ]

        '''
        f2 = Forall([u1, u2, i],
                    Implies(
                        And(
                            sim([u1, u2]),
                            likes([get_u_features(u1), get_i_features(i)])
                        ),
                        likes([get_u_features(u2), get_i_features(i)])
                    ),
                    mask_vars=[u1, u2],
                    mask_fn=lambda mask_vars: ~torch.eq(mask_vars[0], mask_vars[1])
                    )
        '''
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

        axioms = torch.stack(axioms)

        sat_level = formula_aggregator(axioms, dim=0)

        return sat_level

    # test metrics
    def compute_mae(predictions, ground_truth):
        return np.mean(np.abs(predictions - ground_truth))

    def compute_rmse(predictions, ground_truth):
        return np.sqrt(np.mean(np.square(predictions - ground_truth)))

    # training of the LTN model for recommendation
    params = list(likes_nn.parameters())
    params.append(like_embedding)
    optimizer = torch.optim.Adam(params, lr=0.001)

    '''
    train_loss = 0.0
    mean_sat, tr_mean_mae, tr_mean_rmse = 0.0, 0.0, 0.0
    for batch_idx, (uid, iid, rate) in enumerate(train_loader):
        sat_agg = axioms(uid, iid, rate)
        mean_sat += sat_agg.item()
        loss = 1. - sat_agg
        print(like_embedding[0])
        train_loss += loss.item()
        predictions = likes_nn.model([users[uid].float(), items[iid].float()]).detach().cpu().numpy().flatten()
        tr_mean_mae += compute_mae(predictions * 4 + 1, rate * 4 + 1)
        tr_mean_rmse += compute_rmse(predictions * 4 + 1, rate * 4 + 1)
    train_loss = train_loss / len(train_loader)
    mean_sat = mean_sat / len(train_loader)
    tr_mean_mae = tr_mean_mae / len(train_loader)
    tr_mean_rmse = tr_mean_rmse / len(train_loader)
    

    logger.info(
        " epoch %d | loss %.4f | Train Sat %.3f | Train RMSE %.3f | Train MAE %.3f ",
        -1, train_loss, mean_sat, tr_mean_rmse, tr_mean_mae)
    '''

    for epoch in range(100):
        train_loss = 0.0
        mean_sat, tr_mean_mae, tr_mean_rmse = 0.0, 0.0, 0.0
        for batch_idx, (uid, iid, rate) in enumerate(train_loader):
            optimizer.zero_grad()
            sat_agg = axioms(uid, iid, rate)
            mean_sat += sat_agg.item()
            loss = 1. - sat_agg
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            predictions = likes_nn.model([users[uid].float(), items[iid].float()]).detach().cpu().numpy().flatten()
            tr_mean_mae += compute_mae(predictions * 4 + 1, rate * 4 + 1)
            tr_mean_rmse += compute_rmse(predictions * 4 + 1, rate * 4 + 1)
        train_loss = train_loss / len(train_loader)
        mean_sat = mean_sat / len(train_loader)
        tr_mean_mae = tr_mean_mae / len(train_loader)
        tr_mean_rmse = tr_mean_rmse / len(train_loader)

        # test step
        mean_sat_test, mean_mae, mean_rmse = 0.0, 0.0, 0.0
        for batch_idx, (uid, iid, rate) in enumerate(test_loader):
            sat_agg = axioms(uid, iid, rate)
            mean_sat_test += sat_agg.item()
            # compute rating prediction metrics
            predictions = likes.model([users[uid].float(), items[iid].float()]).detach().cpu().numpy().flatten()
            mean_mae += compute_mae(predictions * 4 + 1, rate * 4 + 1)
            mean_rmse += compute_rmse(predictions * 4 + 1, rate * 4 + 1)

        mean_rmse = mean_rmse / len(test_loader)
        mean_mae = mean_mae / len(test_loader)
        mean_sat_test = mean_sat_test / len(test_loader)

        logger.info(
            " epoch %d | loss %.4f | Train Sat %.3f | Test Sat %.3f | Train RMSE %.3f | Train MAE %.3f | Test RMSE %.3f | Test MAE %.3f ",
            epoch, train_loss, mean_sat, mean_sat_test, tr_mean_rmse, tr_mean_mae, mean_rmse, mean_mae)


if __name__ == "__main__":
    main()
