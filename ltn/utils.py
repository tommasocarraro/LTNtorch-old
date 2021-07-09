"""
This package contains some utility functions and models that are used in the implementation of the examples of the
LTN paper.
"""
import torch
import torchvision
import numpy as np


class LogitsToPredicateModel(torch.nn.Module):
    """
    Given a model C that outputs k logits (for k classes)
        e.g. C(x) returns k values, not bounded in [0,1]
    `Cp = LogitsToPredicateModel(C)` is a corresponding model that returns
    probabilities for the class at the given index.
        e.g. Cp([x,i]) where i=0,1,...,k-1, returns one value in [0,1] for class i
    """

    def __init__(self, logits_model, single_label=True):
        """
        logits_model: a tf Model that outputs logits
        single_label: True for exclusive classes (logits are translated into probabilities using softmax),
                False for non-exclusive classes (logits are translated into probabilities using sigmoid)
        """
        super(LogitsToPredicateModel, self).__init__()
        self.logits_model = logits_model
        self.to_probs = torch.nn.Softmax(dim=1) if single_label else torch.nn.Sigmoid()

    def forward(self, inputs, training=False):
        """
        Args:
            inputs: the inputs of the model
                inputs[0] are the inputs to the logits_model, for which we have then to compute probabilities.
                    probs[i] = to_probs(logits_model(inputs[0,i]))
                inputs[1] are the classes to index the result such that:
                    results[i] = probs[i,inputs[1][i]]

            training: boolean flag indicating whether the dropout units of the logits model have to be used or not.
            Yes if training is True, No otherwise.
        """
        x = inputs[0]
        logits = self.logits_model(x, training)
        probs = self.to_probs(logits)
        indices = torch.stack(inputs[1:], dim=1).long()
        return torch.gather(probs, 1, indices)


def get_mnist_dataset_for_digits_addition(single_digit=True):
    """
    It prepares the dataset for the MNIST single digit or multi digits addition example of the LTN paper.

    :param single_digit: whether the dataset has to be generated for the single digit or multi digits example (please,
    carefully read the examples in the paper to understand the differences between the two).
    :return: a tuple of two elements. The first element is the training set, while the second element is the test set.
    Both training set and test set are lists that contain the following information:
        1. a list [left_operands, right_operands], where left_operands is a list of MNIST images that are used as the
        left operand of the addition, while right_operands is a list of MNIST images that are used as the right operand
        of the addition;
        2. a list [left_operands_labels, right_operands_labels], where left_operands_labels contains the labels of the
        MNIST images contained in left_operands, while right_operands_labels contains the labels of the MNIST images
        contained in right_operands;
        3. a list `summation` containing the summation of the labels contained in
        labels = [left_operands_labels, right_operands_labels], s.t. summation[i] = labels[0][i] + labels[1][i].
    Note that this is the output of the process for the single digit case. In the multi digits case the lists at points
    1. and 2. will have 4 elements each since in the multi digits case 4 digits are involved in each addition.
    """
    n_train_examples, n_test_examples, n_operands, label_generator_function = None, None, None, None
    if single_digit:
        n_train_examples = 30000
        n_test_examples = 5000
        n_operands = 2
        label_generator_function = lambda labels: labels[0] + labels[1]
    else:
        n_train_examples = 15000
        n_test_examples = 2500
        n_operands = 4
        label_generator_function = lambda labels: 10 * labels[0] + labels[1] + 10 * labels[2] + labels[3]

    mnist_train = torchvision.datasets.MNIST("./examples/datasets/", train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())
    mnist_test = torchvision.datasets.MNIST("./examples/datasets/", train=False, download=True,
                                            transform=torchvision.transforms.ToTensor())

    train_imgs, train_labels, test_imgs, test_labels = mnist_train.data.numpy(), mnist_train.targets.numpy(), \
                                                       mnist_test.data.numpy(), mnist_test.targets.numpy()

    imgs_operand_train = [train_imgs[i * n_train_examples:i * n_train_examples + n_train_examples]
                          for i in range(n_operands)]
    labels_operand_train = [train_labels[i * n_train_examples:i * n_train_examples + n_train_examples]
                            for i in range(n_operands)]

    label_addition_train = np.apply_along_axis(label_generator_function, 0, labels_operand_train)

    imgs_operand_test = [test_imgs[i * n_test_examples:i * n_test_examples + n_test_examples]
                         for i in range(n_operands)]
    labels_operand_test = [test_imgs[i * n_test_examples:i * n_test_examples + n_test_examples]
                           for i in range(n_operands)]
    label_addition_test = np.apply_along_axis(label_generator_function, 0, labels_operand_test)

    train_set = [imgs_operand_train, labels_operand_train, label_addition_train]
    test_set = [imgs_operand_test, labels_operand_test, label_addition_test]

    return train_set, test_set