"""
Here, you can find the the multi digits addition example of the LTN paper. Please, carefully read the example on the
paper before going through the PyTorch example.
"""
import logging
import torch
import ltn
import numpy as np
from sklearn.metrics import accuracy_score
from torch.nn.init import xavier_uniform_, normal_
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


# this is a standard PyTorch DataLoader to load the dataset for the training and testing of the model
class DataLoader(object):
    def __init__(self,
                 fold,
                 batch_size=1,
                 shuffle=True):
        self.fold = fold
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.fold[0].shape[0] / self.batch_size))

    def __iter__(self):
        n = self.fold[0].shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            digits = self.fold[0][idxlist[start_idx:end_idx]]
            addition_labels = self.fold[1][idxlist[start_idx:end_idx]]

            yield digits, addition_labels


class SingleDigitClassifier(torch.nn.Module):
    """
    Model classifying one MNIST digit image into 10 possible classes. It outputs the logits, so it is not a normalized
    output. It has a convolutional part in the initial layers of the architecture and a linear part in the last
    layers of the architecture. To build the convolutional part a pre-configured convolutional model contained
    in the `utils.py` file is used.
    Args:
        layers_sizes: tuple containing the sizes of the linear layers used as the final layers of the
        architecture. The first element of the tuple must be the number of features in input to the first linear layer,
        while the last element of the tuple must be the number of output features of the last linear layer. Specifically,
        the number of layers constructed is equal to `len(layers_sizes) - 1`.
    """
    def __init__(self, layers_sizes=(100, 84, 10)):
        super(SingleDigitClassifier, self).__init__()
        self.mnistconv = ltn.utils.MNISTConv()  # this is the convolutional part of the architecture
        self.tanh = torch.nn.Tanh()  # tanh is used as activation for the linear layers
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(layers_sizes[i - 1], layers_sizes[i])
                                                  for i in range(1, len(layers_sizes))])
        self.batch_norm_layers = torch.nn.ModuleList([torch.nn.BatchNorm1d(layers_sizes[i])
                                                  for i in range(1, len(layers_sizes))])
        self.init_weights()

    def forward(self, inputs, training=False):  # the parameter training is not used in this example
        x = inputs
        x = self.mnistconv(x)
        for i in range(len(self.linear_layers) - 1):
            x = self.tanh(self.batch_norm_layers[i](self.linear_layers[i](x)))
        return self.linear_layers[-1](x)  # in the last layer a sigmoid or a softmax has to be applied

    def init_weights(self):
        r"""Initialize the weights of the linear layers of the network.
        Weights are initialized with the :py:func:`torch.nn.init.xavier_uniform_` initializer,
        while biases are initialized with the :py:func:`torch.nn.init.normal_` initializer.
        """
        for layer in self.linear_layers:
            xavier_uniform_(layer.weight)
            normal_(layer.bias)


def main():
    # DATASET
    train_set, test_set = ltn.utils.get_mnist_dataset_for_digits_addition(single_digit=False)

    # create the DataLoader to prepare the dataset for training and testing
    train_loader = DataLoader(train_set, 32)
    test_loader = DataLoader(test_set, 32, False)

    # LTN model

    # Predicate Digit of the LTN example on the paper
    logits_model = SingleDigitClassifier()
    Digit = ltn.Predicate(ltn.utils.LogitsToPredicateModel(logits_model)).to(ltn.device)

    # Fixed variables (they do not change their values during training)
    # These variables represent the 10 labels that a MNIST digit can belong to
    d1 = ltn.variable("digits1", range(10))
    d2 = ltn.variable("digits2", range(10))
    d3 = ltn.variable("digits3", range(10))
    d4 = ltn.variable("digits4", range(10))

    # Operators
    And = ltn.WrapperConnective(ltn.fuzzy_ops.AndProd())
    Forall = ltn.WrapperQuantifier(ltn.fuzzy_ops.AggregPMeanError(), quantification_type="forall")
    Exists = ltn.WrapperQuantifier(ltn.fuzzy_ops.AggregPMean(), quantification_type="exists")

    # this function defines the variables and the axioms that need to be used to train the predicate Digit
    # it returns the satisfaction level of the given knowledge base (axioms)
    # see the example in the paper to understand the axiom used in this knowledge base
    def axioms(operand_images, addition_label, p_schedule=2):
        images_x1 = ltn.variable("x1", operand_images[:, 0])
        images_y1 = ltn.variable("x2", operand_images[:, 1])
        images_x2 = ltn.variable("y1", operand_images[:, 2])
        images_y2 = ltn.variable("y2", operand_images[:, 3])
        labels_z = ltn.variable("z", addition_label)
        return Forall(
            ltn.diag([images_x1, images_x2, images_y1, images_y2, labels_z]),
            Exists(
                [d1, d2, d3, d4],
                And(
                    And(Digit([images_x1, d1]), Digit([images_x2, d2])),
                    And(Digit([images_y1, d3]), Digit([images_y2, d4]))
                ),
                mask_vars=[d1, d2, d3, d4, labels_z],
                mask_fn=lambda vars: torch.eq(10 * vars[0] + vars[1] + 10 * vars[2] + vars[3], vars[4]),
                p=p_schedule
            ),
            p=1
        )

    # define metrics

    # it computes the overall accuracy and satisfaction level of the trained model using the given data loader
    # (train or test)
    def compute_metrics(loader):
        mean_accuracy = 0.0
        mean_sat = 0
        for operand_images, addition_label in loader:
            predictions_x1 = logits_model(operand_images[:, 0].to(ltn.device)).detach().cpu().numpy()
            predictions_x2 = logits_model(operand_images[:, 1].to(ltn.device)).detach().cpu().numpy()
            predictions_x1 = np.argmax(predictions_x1, axis=1)
            predictions_x2 = np.argmax(predictions_x2, axis=1)
            predictions_y1 = logits_model(operand_images[:, 2].to(ltn.device)).detach().cpu().numpy()
            predictions_y2 = logits_model(operand_images[:, 3].to(ltn.device)).detach().cpu().numpy()
            predictions_y1 = np.argmax(predictions_y1, axis=1)
            predictions_y2 = np.argmax(predictions_y2, axis=1)
            predictions = 10 * predictions_x1 + predictions_x2 + 10 * predictions_y1 + predictions_y2
            mean_accuracy += accuracy_score(addition_label, predictions)
            mean_sat += axioms(operand_images, addition_label).item()

        return mean_accuracy / len(loader), mean_sat / len(loader)

    # # Training
    #
    # While training, we measure:
    # 1. The level of satisfiability of the Knowledge Base of the training data.
    # 1. The level of satisfiability of the Knowledge Base of the test data.
    # 3. The training accuracy.
    # 4. The test accuracy.

    optimizer = torch.optim.Adam(params=Digit.parameters(), lr=0.001)

    # training of the predicate Digit using a loss containing the satisfaction level of the knowledge base
    # the objective is to maximize the satisfaction level of the knowledge base
    for epoch in range(30):
        # scheduling of the parameter p for the existential quantifier as described in the LTN paper
        if epoch in range(0, 6):
            p = 1
        if epoch in range(6, 12):
            p = 2
        if epoch in range(12, 18):
            p = 4
        if epoch in range(18, 30):
            p = 6
        train_loss = 0.0
        for batch_idx, (operand_images, addition_label) in enumerate(train_loader):
            optimizer.zero_grad()
            sat_agg = axioms(operand_images, addition_label, p_schedule=p)
            loss = 1. - sat_agg
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)

        # we print metrics every epoch of training
        mean_accuracy_train, mean_sat_train = compute_metrics(train_loader)
        mean_accuracy_test, mean_sat_test = compute_metrics(test_loader)

        logger.info(" epoch %d | loss %.4f | Train Sat %.3f | Test Sat %.3f | Train Acc %.3f | Test Acc %.3f ",
                    epoch, train_loss, mean_sat_train, mean_sat_test,
                    mean_accuracy_train, mean_accuracy_test)


if __name__ == "__main__":
    main()