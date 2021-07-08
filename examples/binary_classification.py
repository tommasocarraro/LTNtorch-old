"""
Here, you can find the the binary classification example of the LTN paper. Please, carefully read the example on the
paper before going through the PyTorch example.
"""
import numpy as np
import ltn
import torch
import logging
from sklearn.metrics import accuracy_score
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


# this is a standard PyTorch DataLoader to load the dataset for the training and testing of the model
class DataLoader(object):
    def __init__(self,
                 data,
                 labels,
                 batch_size=1,
                 shuffle=True):
        self.data = data
        self.labels = labels
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
            labels = self.labels[idxlist[start_idx:end_idx]]

            yield data, labels


def main():
    # # Data
    # Sample data from [0,1]^2.
    # The ground truth of a data point is positive when the data point is close to the center (.5,.5) (given a threshold)
    # All the other data is considered as negative examples
    np.random.seed(12)
    torch.manual_seed(12)
    nr_samples = 100
    data = np.random.uniform([0, 0], [1, 1], (nr_samples, 2))
    labels = np.sum(np.square(data - [.5, .5]), axis=1) < .09
    data = data.astype(np.float32)
    labels = labels.astype(np.float32)

    # 50 examples are selected for training; 50 examples are selected for testing
    # batch size is set to 64 meaning that all the examples are included in one single batch since we have only 50 points
    train_loader = DataLoader(data[:50], labels[:50], 64, True)
    test_loader = DataLoader(data[50:], labels[50:], 64, False)

    # # LTN

    A = ltn.Predicate(layers_size=(2, 16, 16, 1))

    # # Axioms
    #
    # ```
    # forall x_A: A(x_A)
    # forall x_not_A: ~A(x_not_A)
    # ```

    Not = ltn.WrapperConnective(ltn.fuzzy_ops.NotStandard())
    Forall = ltn.WrapperQuantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantification_type="forall")

    formula_aggregator = ltn.fuzzy_ops.AggregPMeanError(p=2)

    # this function defines the variables and the axioms that need to be used to train the predicate A
    # it returns the satisfaction level of the given knowledge base (axioms)
    def axioms(data, labels):
        # NB, here A is simply a neural network used for doing binary classification, while x_A and x_not_A are the
        # positive and negative examples that have to be fed to the network
        x_A = ltn.variable("x_A", data[np.nonzero(labels)])  # these are the positive examples
        x_not_A = ltn.variable("x_not_A", data[np.nonzero(np.logical_not(labels))])  # these are the negative examples
        axioms = [
            Forall(x_A, A(x_A)),  # we force the predicate A to be true for each positive example
            Forall(x_not_A, Not(A(x_not_A)))  # we force the negation of the predicate A to be true for every negative
            # example
        ]
        axioms = torch.stack(axioms)  # we stack the results of the axioms and use the formula aggregator to aggregate
        # the results
        sat_level = formula_aggregator(axioms, dim=0)  # the aggregation of the formulas in the knowledge base returns a
        # value in [0, 1] which can be seen as a satisfaction level of the knowledge base
        return sat_level

    # it computes the overall satisfaction level on the knowledge base using the given data loader (train or test)
    def compute_sat_level(loader):
        mean_sat = 0
        for data, labels in loader:
            mean_sat += axioms(data, labels)
        mean_sat /= len(loader)
        return mean_sat

    # it computes the overall accuracy of the predictions of the trained model using the given data loader (train or test)
    def compute_accuracy(loader):
        mean_accuracy = 0.0
        for data, labels in loader:
            predictions = A.model(torch.tensor(data)).detach().numpy()
            predictions = np.where(predictions > 0.5, 1., 0.).flatten()
            mean_accuracy += accuracy_score(labels, predictions)

        return mean_accuracy / len(loader)

    optimizer = torch.optim.Adam(A.parameters(), lr=0.001)

    # training of the predicate A using a loss containing the satisfaction level of the knowledge base
    # the objective it to maximize the satisfaction level of the knowledge base
    for epoch in range(1000):
        train_loss = 0.0
        for batch_idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = axioms(data, labels)
            loss = 1. - output
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)

        # we print metrics every 20 epochs of training
        if epoch % 20 == 0:
            logger.info(" epoch %d | loss %.4f | Train Sat %.3f | Test Sat %.3f | Train Acc %.3f | Test Acc %.3f",
                        epoch, train_loss, compute_sat_level(train_loader), compute_sat_level(test_loader),
                        compute_accuracy(train_loader), compute_accuracy(test_loader))


if __name__ == "__main__":
    main()