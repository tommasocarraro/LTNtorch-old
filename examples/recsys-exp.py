import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import torch
import os
import ltn

# seed
np.random.seed(2021)
torch.manual_seed(2021)

# Create directory
dirName = 'saved-models'
try:
    # Create target Directory
    os.mkdir(dirName)
    print("Directory ", dirName,  " Created ")
except FileExistsError:
    print("Directory ", dirName,  " already exists")

# read dataset
data = pd.read_csv("datasets/ml-100k/u.data", sep="\t")

# entire dataset
data.columns = ['uid', 'iid', 'rate', 'timestamp']
# training set
data_tr = pd.read_csv("datasets/ml-100k/u1.base", sep="\t")
data_tr.columns = data.columns
# test set
data_te = pd.read_csv("datasets/ml-100k/u1.test", sep="\t")
data_te.columns = data.columns

# prepare dataset for Matrix Factorization

# get number of users and items
n_users = data['uid'].nunique()
n_items = data['iid'].nunique()
# make indexes for users and items starting from 0 (in the dataset they begin with 0)
data_tr['uid'] -= 1
data_tr['iid'] -= 1
data_te['uid'] -= 1
data_te['iid'] -= 1
# create validation fold from training data (10% of the ratings of each user are used as validation set)
groups = data_tr.groupby('uid')
data_val = pd.DataFrame(columns=data_tr.columns).astype(np.int64)
for _, user_group in groups:
    n_ratings_to_remove = np.round(len(user_group) * 10 / 100)  # 10% of ratings
    ratings_to_remove = user_group.sample(n=int(n_ratings_to_remove))
    data_val = data_val.append(ratings_to_remove)
    data_tr = data_tr.drop(ratings_to_remove.index)

data_tr = data_tr.reset_index(drop=True)

# create train, validation and test user-item sparse matrices
data_tr = csr_matrix((data_tr['rate'], (data_tr['uid'], data_tr['iid'])), (n_users, n_items))
data_te = csr_matrix((data_te['rate'], (data_te['uid'], data_te['iid'])), (n_users, n_items))
data_val = csr_matrix((data_val['rate'], (data_val['uid'], data_val['iid'])), (n_users, n_items))


class DataSampler:
    """
    This sampler takes as input one data fold containing a sequence of user-item pairs and returns batches of user-item pairs taken randomly from the fold.
    The fold must be a user-item sparse matrix containing the users on the rows and the items on the columns.
    """
    def __init__(self,
                 data,
                 batch_size=1,
                 shuffle=True):
        self.data = data
        nonzeros = data.nonzero()
        self.rows = nonzeros[0]
        self.cols = nonzeros[1]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.rows.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.rows.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            rows = self.rows[idxlist[start_idx:end_idx]]
            cols = self.cols[idxlist[start_idx:end_idx]]
            ratings = self.data[rows, cols]

            yield torch.LongTensor(rows), torch.LongTensor(cols), torch.FloatTensor(ratings).squeeze_()


def compute_rmse(errors):
    """
    It takes as input a list of errors (tensors), namely a list of preds - groung_truths, and returns the RMSE based on predictions and ground truths given.
    """
    return torch.sqrt(torch.mean(torch.square(torch.cat(errors)))).item()


def compute_mae(errors):
    """
    It takes as input a list of errors (tensors), namely a list of preds - groung_truths, and returns the MAE based on predictions and ground truths given.
    """
    return torch.mean(torch.abs(torch.cat(errors))).item()


class MatrixFactorizationLTN(torch.nn.Module):
    """
    This is a standard matrix factorization model where two matrices of factors are created, one for the users and one for the items.
    The model takes as input a user-item pair and returns the dot product of their factors.
    """
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)
        self.user_biases = torch.nn.Embedding(n_users, 1)
        self.item_biases = torch.nn.Embedding(n_items, 1)

    def forward(self, args):
        users, items = args
        return torch.sigmoid(torch.sum(torch.cat([self.user_factors(users) * self.item_factors(items), self.user_biases(users), self.item_biases(items)], dim=1), dim=1))

Likes = ltn.Predicate(MatrixFactorizationLTN(n_users, n_items, n_factors=3))
Eq = ltn.Predicate(lambda_func=lambda args: torch.exp(-torch.abs(args[0] - args[1])))
Forall = ltn.WrapperQuantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantification_type="forall")


# create cosine similarity matrix
# seen matrix is the user-item matrix where there is a 1 if a user has interacted with an item, 0 otherwise
seen_matrix = torch.tensor(data_tr.toarray())
seen_matrix = torch.where(seen_matrix != 0, 1., 0.)

# compute similarities between each pair of users in the dataset
cos_sim_users = []
for u in range(n_users):
    # expand the user behavior #n_users times
    exp_u_interactions = seen_matrix[u].expand(n_users, n_items)
    # compute similarity between user u and all other users in the datasets
    cos_sim_user_interactions = torch.nn.functional.cosine_similarity(exp_u_interactions, seen_matrix, dim=1)
    cos_sim_users.append(cos_sim_user_interactions)

cos_sim_users = torch.stack(cos_sim_users)

Sim = ltn.Predicate(lambda_func=lambda args: cos_sim_users[args[0], args[1]])

desc_sim_order = torch.argsort(cos_sim_users, dim=1, descending=True) # we order the similarity matrix by similarity in descending order. By doing so, the first n positions of each row will be the most similar users to the user in that row.


def sample_similar_user(user, size=30):
    u = desc_sim_order[user] # this is a vector containing the similarity of the input user with all the other users of the system ordered by similarity in descending order
    # notice that the first position will be taken by the user himself since the user himself is the most similar user to the user himself
    # now, we create a random permutation of indexes to fetch the most similar users. The number of indexes will be equal to the size parameter
    rand_indexes = torch.randperm(size) + 1 # 1 is added because the first position is taken by the user himself
    most_similar_users = u[:, rand_indexes] # this is a random permutation of the size most similar users to the input user
    return most_similar_users[:, 0] # we need only one similar user so we return only the first column of the matrix we have obtained

# IL PROBLEMA DI QUESTA FUNZIONE E' CHE ALL'INTERNO DELLO STESSO BATCH, LO STESSO UTENTE SI BECCA SEMPRE LO STESSO INDICE


Implies = ltn.WrapperConnective(ltn.fuzzy_ops.ImpliesGoguenStrong())
And = ltn.WrapperConnective(ltn.fuzzy_ops.AndProd())
formula_agg = ltn.fuzzy_ops.AggregPMeanError(p=2)


train_loader = DataSampler(data_tr, batch_size=512)
val_loader = DataSampler(data_val, batch_size=512, shuffle=False)
test_loader = DataSampler(data_te, batch_size=512, shuffle=False)


# print satisfaction level of formulas and metrics before the training
Likes = ltn.Predicate(MatrixFactorizationLTN(n_users, n_items, n_factors=3))
train_loss = 0.0
train_error = []
a, b, c, d, f3_l, f4_l, f5_l, f6_l, f7_l, f4_l_bis, f7_l_bis, f4_l_tris, l_w_g_l, l2_w_g_l = [], [], [], [], [], [], [], [], [], [], [], [], [], []
frac = 0
ones = torch.ones((n_users, n_users))
p = ltn.Predicate(lambda_func=lambda args: ones[args[0], args[1]])
param = 1
for (batch_idx, (rows, cols, gt)) in enumerate(train_loader):
    with torch.no_grad():
        # create users, items and ratings LTN variables
        users = ltn.variable('users', rows, add_batch_dim=False)
        sim_users = ltn.variable('sim_users', sample_similar_user(rows), add_batch_dim=False)
        #users2 = ltn.variable('users2', np.random.permutation(rows), add_batch_dim=False)
        users2 = ltn.variable('users2', sample_similar_user(rows), add_batch_dim=False)
        items = ltn.variable('items', cols, add_batch_dim=False)
        ratings = ltn.variable('ratings', (gt - 1) / 4, add_batch_dim=False)

        # Predict and calculate loss
        '''f1 = Forall(ltn.diag([users, items, ratings]), Eq([Likes([users, items]), ratings]))
        f2 = Forall(ltn.diag([users, sim_users, items]), Implies(And(Sim([users, sim_users]), Likes([users, items])), Likes([sim_users, items])),
                    mask_vars=[users, sim_users],
                    mask_fn=lambda mask_vars: torch.gt(Sim([mask_vars[0], mask_vars[1]]), torch.tensor(0.0)))'''
        #f3 = Forall(ltn.diag([users, users2, items]), Implies(And(Sim([users, users2]), Likes([users, items])), Likes([users2, items])),
        #            mask_vars=[users, users2],
        #            mask_fn=lambda mask_vars: torch.gt(Sim([mask_vars[0], mask_vars[1]]), torch.tensor(0.0)))
        f3 = Forall(ltn.diag([users, users2]),
                    Implies(p([users, users2]), Sim([users, users2])),
                    mask_vars=[users, users2],
                    mask_fn=lambda mask_vars: torch.gt(Sim([mask_vars[0], mask_vars[1]]), torch.tensor(0.0)), p=param)
        f4_bis = Forall(ltn.diag([users, users2, items]),
                    Implies(And(Sim([users, users2]), Likes([users, items])), Likes([users2, items])),
                    mask_vars=[users, users2],
                    mask_fn=lambda mask_vars: torch.gt(Sim([mask_vars[0], mask_vars[1]]), torch.tensor(0.4)), p=param)
        f4_tris = Forall(ltn.diag([users, users2, items]),
                        Implies(And(Sim([users, users2]), Likes([users, items])), Likes([users2, items])),
                        mask_vars=[users, users2],
                        mask_fn=lambda mask_vars: torch.gt(Sim([mask_vars[0], mask_vars[1]]), torch.tensor(0.0)),
                        p=param)
        f4 = Forall(ltn.diag([users, users2, items]),
                    Implies(And(Sim([users, users2]), Likes([users, items])), Likes([users2, items])),
                    mask_vars=[users, users2],
                    mask_fn=lambda mask_vars: torch.gt(Sim([mask_vars[0], mask_vars[1]]), torch.tensor(0.5)), p=param)
        l_w_g = Forall(ltn.diag([users, users2, items]),
                    Likes([users, items]),
                    mask_vars=[users, users2],
                    mask_fn=lambda mask_vars: torch.gt(Sim([mask_vars[0], mask_vars[1]]), torch.tensor(0.5)), p=param)
        l2_w_g = Forall(ltn.diag([users, users2, items]),
                       Likes([users2, items]),
                       mask_vars=[users, users2],
                       mask_fn=lambda mask_vars: torch.gt(Sim([mask_vars[0], mask_vars[1]]), torch.tensor(0.5)),
                       p=param)
        f5 = Forall(ltn.diag([users, items]),
                    Likes([users, items]), p=param)
        f6 = Forall(ltn.diag([users2, items]),
                    Likes([users2, items]), p=param)
        f7 = Forall(ltn.diag([users, users2]), Sim([users, users2]), mask_vars=[users, users2],
                    mask_fn=lambda mask_vars: torch.gt(Sim([mask_vars[0], mask_vars[1]]), torch.tensor(0.5)), p=param)
        f7_bis = Forall(ltn.diag([users, users2]), Sim([users, users2]), mask_vars=[users, users2],
                    mask_fn=lambda mask_vars: torch.gt(Sim([mask_vars[0], mask_vars[1]]), torch.tensor(0.0)), p=param)
        #f4 = Forall(ltn.diag([users, items]), Likes([users, items]))
        #formulas_dict['f1'].append(f1)
        #formulas_dict['f2'].append(f2)
        #formulas_dict['f3'].append(f3)
        #formulas_dict['f4'].append(f4)
        f3_l.append(f3)
        f4_l.append(f4)
        f5_l.append(f5)
        f6_l.append(f6)
        f7_l.append(f7)
        f4_l_bis.append(f4_bis)
        f7_l_bis.append(f7_bis)
        f4_l_tris.append(f4_tris)
        l_w_g_l.append(l_w_g)
        l2_w_g_l.append(l2_w_g)
        ltn.diag([users, users2, items])
        a.append(Sim([users, users2]))
        b.append(Likes([users, items]))
        c.append(Likes([users2, items]))
        d.append(Implies(And(Sim([users, users2]), Likes([users, items])), Likes([users2, items])))
        ltn.undiag([users, users2, items])
        axioms = torch.stack([f3])
        sat_agg = formula_agg(axioms, dim=0)
        loss = 1. - sat_agg
        train_loss += loss.item()

        # Compute RMSE on training set
        pred = Likes.model([rows, cols])
        train_error.append((pred * 4 + 1) - gt)

train_loss = train_loss / len(train_loader)
train_rmse = compute_rmse(train_error)
#print("Epoch %d | Train loss %.3f | Train RMSE %.3f | f1 SAT %.3f | f2 SAT %.3f | f3 SAT %.3f | f4 SAT %.3f" % (-1, train_loss, train_rmse, torch.mean(torch.stack(formulas_dict['f1'])), torch.mean(torch.stack(formulas_dict['f2'])), torch.mean(torch.stack(formulas_dict['f3'])), torch.mean(torch.stack(formulas_dict['f4']))))
print("The rule is: Sim(u1, u2) and Likes(u1, i) -> Likes(u2, i) and the p used for quantification is %d" % param)
print("-----")
print("Mean value of Sim(u1, u2): %.3f" % torch.mean(torch.cat(a)))
print("Mean value of Forall (u1, u2) Sim(u1, u2) (no guarded): %.3f" % torch.mean(torch.stack(f7_l_bis)))
print("Mean value of Forall (u1, u2) Sim(u1, u2) (guarded 0.5): %.3f" % torch.mean(torch.stack(f7_l)))
print("Mean value of Forall(u1, u2) 1 -> Sim(u1, u2): %.3f" % torch.mean(torch.stack(f3_l)))
print("-----")
print("Mean value of Likes(u1, i): %.3f" % torch.mean(torch.cat(b)))
print("Mean value of Forall(u1, i) Likes(u1, i): %.3f" % torch.mean(torch.stack(f5_l)))
print("Mean value of Forall(u1, i) Likes(u1, i) (guarded 0.5): %.3f" % torch.mean(torch.stack(l_w_g_l)))
print("-----")
print("Mean value of Likes(u2, i): %.3f" % torch.mean(torch.cat(c)))
print("Mean value of Forall(u2, i) Likes(u2, i): %.3f" % torch.mean(torch.stack(f6_l)))
print("Mean value of Forall(u2, i) Likes(u2, i) (guarded 0.5): %.3f" % torch.mean(torch.stack(l2_w_g_l)))
print("-----")
print("Mean value of Sim(u1, u2) and Likes(u1, i) -> Likes(u2, i): %.3f" % torch.mean(torch.cat(d)))
print("Mean value of Forall(u1, u2, i) Sim(u1, u2) and Likes(u1, i) -> Likes(u2, i) (no guarded): %.3f" % torch.mean(torch.stack(f4_l_tris)))
print("Mean value of Forall(u1, u2, i) Sim(u1, u2) and Likes(u1, i) -> Likes(u2, i) (guarded 0.4): %.3f" % torch.mean(torch.stack(f4_l_bis)))
print("Mean value of Forall(u1, u2, i) Sim(u1, u2) and Likes(u1, i) -> Likes(u2, i) (guarded 0.5): %.3f" % torch.mean(torch.stack(f4_l)))