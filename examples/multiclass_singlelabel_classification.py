import logging
import torch
import numpy as np
import pandas as pd
import ltn
# TODO usare numpy per calcolare le metriche
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


class MLP(torch.nn.Module):
    def __init__(self, layer_sizes=(4, 16, 16, 8, 3)):
        super(MLP, self).__init__()
        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax()
        self.dropout = torch.nn.Dropout(0.2)
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i])
                                                  for i in range(1, len(layer_sizes))])

    def forward(self, inputs, training=False):
        x = inputs[0]
        class_label = inputs[1]
        for layer in self.linear_layers[:-1]:
            x = layer(x)
            if training:
                x = self.dropout(x)
        outputs = self.softmax(self.linear_layers[-1](x))
        return outputs[:, class_label[0].long()]


def main():
    torch.manual_seed(12)
    np.random.seed(12)
    # # Data
    #
    # Load the iris dataset:
    # 50 samples from each of three species of iris flowers (setosa, virginica, versicolor), measured with four features.

    train_data = pd.read_csv("datasets/iris_training.csv")
    test_data = pd.read_csv("datasets/iris_test.csv")

    train_labels = train_data.pop("species")
    test_labels = test_data.pop("species")

    train_data = np.array(train_data)
    train_data = train_data.astype(np.float32)
    test_data = np.array(test_data)
    test_data = test_data.astype(np.float32)
    train_labels = np.array(train_labels)
    train_labels = train_labels.astype(np.longlong)
    test_labels = np.array(test_labels)
    test_labels = test_labels.astype(np.longlong)

    train_loader = DataLoader(train_data, train_labels, 64)
    test_loader = DataLoader(test_data, test_labels, 64, shuffle=False)

    model = MLP()
    p = ltn.Predicate(model)

    def compute_accuracy(loader):
        mean_accuracy = 0.0
        for data, labels in loader:
            predictions = model([torch.tensor(data), torch.tensor(labels)])
            print(predictions)
            print(torch.argmax(predictions, dim=1))
            predictions = torch.where(predictions > 0.5, 1., 0.)
            labels = torch.tensor(labels)
            labels = labels.view(labels.shape[0], 1)
            accuracy = torch.where(predictions == labels, 1., 0.)
            mean_accuracy += torch.sum(accuracy) / data.shape[0]

        return mean_accuracy / len(loader)

    # Constants to index/iterate on the classes
    class_A = ltn.constant(0)
    class_B = ltn.constant(1)
    class_C = ltn.constant(2)

    # Operators and axioms
    Forall = ltn.WrapperQuantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantification_type="forall")

    formula_aggregator = ltn.fuzzy_ops.AggregPMeanError(p=2)

    def axioms(features, labels, training=False):
        x_A = ltn.variable("x_A", features[labels == 0])
        x_B = ltn.variable("x_B", features[labels == 1])
        x_C = ltn.variable("x_C", features[labels == 2])
        axioms = [
            Forall(x_A, p([x_A, class_A], training=training)),
            Forall(x_B, p([x_B, class_B], training=training)),
            Forall(x_C, p([x_C, class_C], training=training))
        ]
        axioms = torch.stack(axioms)
        sat_level = formula_aggregator(axioms, dim=0)
        return sat_level

    # # Training
    #
    # Define the metrics. While training, we measure:
    # 1. The level of satisfiability of the Knowledge Base of the training data.
    # 1. The level of satisfiability of the Knowledge Base of the test data.
    # 3. The training accuracy.
    # 4. The test accuracy.

    # Define the training and test step

    optimizer = torch.optim.Adam(params=p.parameters(), lr=0.001)

    for epoch in range(500):
        train_loss = 0.0
        for batch_idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = axioms(data, labels, training=True)
            loss = 1. - output
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)

        # we print metrics every 20 epochs of training
        if epoch % 20 == 0:
            mean_sat_test = 0
            for data, labels in test_loader:
                mean_sat_test += axioms(data, labels, training=False)
            mean_sat_train = 0
            for data, labels in train_loader:
                mean_sat_train += axioms(data, labels, training=False)
            # | Train Acc %.3f | Test Acc %.3f compute_accuracy(train_loader), compute_accuracy(test_loader)
            logger.info(" epoch %d | loss %.4f | Train Sat %.3f | Test Sat %.3f ",
                        epoch, train_loss, mean_sat_train / len(train_loader), mean_sat_test / len(test_loader))
            print(compute_accuracy(train_loader))


if __name__ == "__main__":
    main()