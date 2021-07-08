"""
Here, you can find the the regression example of the LTN paper. Please, carefully read the example on the paper before
going through the PyTorch example.
"""
import logging
import torch
import numpy as np
import pandas as pd
import ltn
from sklearn.metrics import mean_squared_error
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


# this is a standard PyTorch DataLoader to load the dataset for the training and testing of the model
class DataLoader(object):
    def __init__(self,
                 x,
                 y,
                 batch_size=1,
                 shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.x.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.x.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            x = self.x[idxlist[start_idx:end_idx]]
            y = self.y[idxlist[start_idx:end_idx]]

            yield x, y


def main():
    np.random.seed(12)
    torch.manual_seed(12)

    # # Data
    #
    # Load the real estate dataset

    df = pd.read_csv("datasets/real-estate.csv")
    df = df.sample(frac=1)  # shuffle

    x = np.array(df[['X1 transaction date', 'X2 house age',
                     'X3 distance to the nearest MRT station',
                     'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']]).astype(np.float32)

    y = np.array(df[['Y house price of unit area']]).astype(np.float32)

    # load dataset for training and testing the model
    train_loader = DataLoader(x[:330], y[:330], 64, shuffle=True)
    test_loader = DataLoader(x[330:], y[330:], 64)

    # # LTN
    #
    # Regressor (trainable)

    f = ltn.Function(layers_size=(6, 8, 8, 1))

    # Equality Predicate - not trainable
    alpha = 0.05
    eq = ltn.Predicate(lambda_func=lambda args: torch.exp(
        -alpha * torch.sqrt(torch.sum(torch.square(args[0] - args[1]), dim=1)))
    )

    # Operators and axioms
    Forall = ltn.WrapperQuantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantification_type="forall")

    # this function defines the variables and the axioms that need to be used to train the function f
    # it returns the satisfaction level of the given knowledge base (axioms)
    def axioms(x_data, y_data):
        x = ltn.variable("x", x_data)
        y = ltn.variable("y", y_data)
        return Forall(ltn.diag([x, y]), eq([f(x), y]))

    # define the metrics

    # it computes the overall satisfaction level on the knowledge base using the given data loader (train or test)
    def compute_sat_level(loader):
        mean_sat = 0
        for x_data, y_data in loader:
            mean_sat += axioms(x_data, y_data)
        mean_sat /= len(loader)
        return mean_sat

    # it computes the overall RMSE of the predictions of the trained model using the given data loader (train or test)
    def compute_rmse(loader):
        mean_rmse = 0.0
        for x_data, y_data in loader:
            predictions = f.model(torch.tensor(x_data)).detach().numpy()
            mean_rmse += mean_squared_error(y_data, predictions, squared=False)
        return mean_rmse / len(loader)

    # # Training
    #
    # Define the metrics. While training, we measure:
    # 1. The level of satisfiability of the Knowledge Base of the training data.
    # 1. The level of satisfiability of the Knowledge Base of the test data.
    # 3. The training accuracy.
    # 4. The test accuracy.

    optimizer = torch.optim.Adam(f.parameters(), lr=0.0005)

    # training of the function f using a loss containing the satisfaction level of the knowledge base
    # the objective it to maximize the satisfaction level of the knowledge base
    for epoch in range(500):
        train_loss = 0.0
        for batch_idx, (x_data, y_data) in enumerate(train_loader):
            optimizer.zero_grad()
            sat_agg = axioms(x_data, y_data)
            loss = 1. - sat_agg
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)

        # we print metrics every 50 epochs of training
        if epoch % 50 == 0:
            # | Train Acc %.3f | Test Acc %.3f compute_accuracy(train_loader), compute_accuracy(test_loader)
            logger.info(" epoch %d | loss %.4f | Train Sat %.3f | Test Sat %.3f | Train RMSE %.3f | Test RMSE %.3f ",
                        epoch, train_loss, compute_sat_level(train_loader), compute_sat_level(test_loader),
                        compute_rmse(train_loader), compute_rmse(test_loader))


if __name__ == "__main__":
    main()
