"""
This package contains some utility functions and models that are used in the implementation of the examples of the
LTN paper.
"""
import torch
import torchvision
from torch.nn.init import kaiming_uniform_, normal_, xavier_uniform_


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
        logits_model: a PyTorch model that outputs logits
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
        inputs[1] = torch.flatten(inputs[1])
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
        2. a list containing the summation of the labels of the images contained in the list at point 1. The label of
        the left operand is added to the label of the right operand, and the target label is generated. This represents
        the target of the digits addition task.
    Note that this is the output of the process for the single digit case. In the multi digits case the list at point
    1. will have 4 elements since in the multi digits case 4 digits are involved in each addition (two digits
    represent the first operand and two digits the second operand).
    """
    if single_digit:
        n_train_examples = 30000
        n_test_examples = 5000
        n_operands = 2
    else:
        n_train_examples = 15000
        n_test_examples = 2500
        n_operands = 4

    mnist_train = torchvision.datasets.MNIST("./examples/datasets/", train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())
    mnist_test = torchvision.datasets.MNIST("./examples/datasets/", train=False, download=True,
                                            transform=torchvision.transforms.ToTensor())

    train_imgs, train_labels, test_imgs, test_labels = mnist_train.data, mnist_train.targets, \
                                                       mnist_test.data, mnist_test.targets

    train_imgs, test_imgs = train_imgs / 255.0, test_imgs / 255.0

    train_imgs, test_imgs = torch.unsqueeze(train_imgs, 1), torch.unsqueeze(test_imgs, 1)

    imgs_operand_train = [train_imgs[i * n_train_examples:i * n_train_examples + n_train_examples]
                          for i in range(n_operands)]
    labels_operand_train = [train_labels[i * n_train_examples:i * n_train_examples + n_train_examples]
                            for i in range(n_operands)]

    imgs_operand_test = [test_imgs[i * n_test_examples:i * n_test_examples + n_test_examples]
                         for i in range(n_operands)]
    labels_operand_test = [test_labels[i * n_test_examples:i * n_test_examples + n_test_examples]
                           for i in range(n_operands)]

    if single_digit:
        label_addition_train = labels_operand_train[0] + labels_operand_train[1]
        label_addition_test = labels_operand_test[0] + labels_operand_test[1]
    else:
        label_addition_train = 10 * labels_operand_train[0] + labels_operand_train[1] + \
                               10 * labels_operand_train[2] + labels_operand_train[3]
        label_addition_test = 10 * labels_operand_test[0] + labels_operand_test[1] + \
                              10 * labels_operand_test[2] + labels_operand_test[3]

    train_set = [torch.stack(imgs_operand_train, dim=1), label_addition_train]
    test_set = [torch.stack(imgs_operand_test, dim=1), label_addition_test]

    return train_set, test_set


class MNISTConv(torch.nn.Module):
    """
    CNN that returns linear embeddings for MNIST images. It is used in the single digit and multi digits addition
    examples of the LTN paper.
    Args:
        conv_channels_sizes: tuple containing the number of channels of the convolutional layers of the model. The first
        element of the tuple must be the number of input channels of the first conv layer, while the last element
        of the tuple must be the number of output channels of the last conv layer. Specifically, the number of conv
        layers constructed is equal to `len(conv_channels_sizes) - 1`;
        kernel_sizes: tuple containing the sizes of the kernels used in the conv layers of the architecture;
        linear_layers_sizes: tuple containing the sizes of the linear layers used as the final layers of the
        architecture. The first element of the tuple must be the number of features in input to the first linear layer,
        while the last element of the tuple must be the number of output features of the last linear layer. Specifically,
        the number of layers constructed is equal to `len(linear_layers_sizes) - 1`.
    """
    def __init__(self, conv_channels_sizes=(1, 6, 16), kernel_sizes=(5, 5), linear_layers_sizes=(256, 100)):
        super(MNISTConv, self).__init__()
        self.conv_layers = torch.nn.ModuleList([torch.nn.Conv2d(conv_channels_sizes[i - 1], conv_channels_sizes[i],
                                                                kernel_sizes[i - 1])
                                                  for i in range(1, len(conv_channels_sizes))])
        self.relu = torch.nn.ReLU()  # relu is used as activation for the conv layers
        self.tanh = torch.nn.Tanh()  # tanh is used as activation for the linear layers
        self.maxpool = torch.nn.MaxPool2d((2, 2))
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(linear_layers_sizes[i - 1], linear_layers_sizes[i])
                                                  for i in range(1, len(linear_layers_sizes))])

        self.init_weights()

    def forward(self, x):
        for conv in self.conv_layers:
            x = self.relu(conv(x))
            x = self.maxpool(x)
        x = torch.flatten(x, start_dim=1)
        for linear_layer in self.linear_layers:
            x = self.tanh(linear_layer(x))
        return x

    def init_weights(self):
        r"""Initialize the weights of the network.
        Weights of conv layers are initialized with the :py:func:`torch.nn.init.kaiming_uniform_` initializer,
        weights of linear layers are initialized with the :py:func:`torch.nn.init.xavier_uniform_` initializer,
        while biases are initialized with the :py:func:`torch.nn.init.normal_` initializer.
        """
        for layer in self.conv_layers:
            kaiming_uniform_(layer.weight)
            normal_(layer.bias)

        for layer in self.linear_layers:
            xavier_uniform_(layer.weight)
            normal_(layer.bias)
