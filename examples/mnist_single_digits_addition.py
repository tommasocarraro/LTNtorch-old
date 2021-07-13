import logging
import torch
import ltn
import numpy as np
from sklearn.metrics import accuracy_score
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
# TODO capire come mettere tutto in GPU perche' non funziona questo esperimento
# TODO sistemare il problema di memoria della GPU

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
    layers of the architecture. To build the convolutional part it uses a pre-configured convolutional model contained
    in the `utils.py` file.
    Args:
        layers_sizes: tuple containing the sizes of the linear layers used as the final layers of the
        architecture. The first element of the tuple must be the number of features in input to the first linear layer,
        while the last element of the tuple must be the number of output features of the last linear layer.
    """
    def __init__(self, layers_sizes=(100, 84, 10)):
        super(SingleDigitClassifier, self).__init__()
        self.mnistconv = ltn.utils.MNISTConv()
        self.elu = torch.nn.ELU()
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(layers_sizes[i - 1], layers_sizes[i])
                                                  for i in range(1, len(layers_sizes))])

    def forward(self, inputs, training=False):
        x = inputs
        x = self.mnistconv(x)
        for layer in self.linear_layers[:-1]:
            x = self.elu(layer(x))
        return self.linear_layers[-1](x)


def main():
    # DATASET
    train_set, test_set = ltn.utils.get_mnist_dataset_for_digits_addition(single_digit=True)

    # create the DataLoader to prepare the dataset for training and testing
    train_loader = DataLoader(train_set, 32)
    test_loader = DataLoader(test_set, 32)

    # LTN model

    # Predicates
    logits_model = SingleDigitClassifier()
    Digit = ltn.Predicate(ltn.utils.LogitsToPredicateModel(logits_model)).to(ltn.device)

    # Fixed variables (they do not change their values during training)
    d1 = ltn.variable("digits1", range(10))
    d2 = ltn.variable("digits2", range(10))

    # Operators
    And = ltn.WrapperConnective(ltn.fuzzy_ops.AndProd())
    Forall = ltn.WrapperQuantifier(ltn.fuzzy_ops.AggregPMeanError(), quantification_type="forall")
    Exists = ltn.WrapperQuantifier(ltn.fuzzy_ops.AggregPMean(), quantification_type="exists")

    # this function defines the variables and the axioms that need to be used to train the predicate Digit
    # it returns the satisfaction level of the given knowledge base (axioms)
    def axioms(operand_images, addition_label, p_schedule=2):
        images_x = ltn.variable("x", torch.unsqueeze(operand_images[:, 0], 1))
        images_y = ltn.variable("y", torch.unsqueeze(operand_images[:, 1], 1))
        labels_z = ltn.variable("z", addition_label)
        return Forall(
            ltn.diag([images_x, images_y, labels_z]),
            Exists(
                [d1, d2],
                And(Digit([images_x, d1]), Digit([images_y, d2])),
                mask_vars=[d1, d2, labels_z],
                mask_fn=lambda vars: torch.eq(vars[0] + vars[1], vars[2]),
                p=p_schedule
            ),
            p=2
        )

    # define metrics

    # it computes the overall accuracy of the predictions of the trained model using the given data loader (train or test)
    def compute_accuracy(loader):
        mean_accuracy = 0.0
        for operand_images, addition_label in loader:
            predictions_x = logits_model(torch.unsqueeze(operand_images[:, 0], 1)).detach().cpu().numpy()
            predictions_y = logits_model(torch.unsqueeze(operand_images[:, 1], 1)).detach().cpu().numpy()
            predictions_x = np.argmax(predictions_x, axis=1)
            predictions_y = np.argmax(predictions_y, axis=1)
            predictions = predictions_x + predictions_y
            mean_accuracy += accuracy_score(addition_label, predictions)

        return mean_accuracy / len(loader)

    # it computes the overall satisfaction level on the knowledge base using the given data loader (train or test)
    def compute_sat_level(loader):
        mean_sat = 0
        for operand_images, addition_label in loader:
            mean_sat += axioms(operand_images, addition_label)
            print(mean_sat)
        mean_sat /= len(loader)
        return mean_sat

    # # Training
    #
    # While training, we measure:
    # 1. The level of satisfiability of the Knowledge Base of the training data.
    # 1. The level of satisfiability of the Knowledge Base of the test data.
    # 3. The training accuracy.
    # 4. The test accuracy.

    optimizer = torch.optim.Adam(params=Digit.parameters(), lr=0.001)
    # TODO fare lo scheduling del p
    # training of the predicate Digit using a loss containing the satisfaction level of the knowledge base
    # the objective is to maximize the satisfaction level of the knowledge base
    for epoch in range(20):
        train_loss = 0.0
        for batch_idx, (operand_images, addition_label) in enumerate(train_loader):
            optimizer.zero_grad()
            sat_agg = axioms(operand_images, addition_label)
            loss = 1. - sat_agg
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)

        print("fine training")

        # we print metrics every epoch of training
        # | Train Acc %.3f | Test Acc %.3f compute_accuracy(train_loader), compute_accuracy(test_loader)
        logger.info(" epoch %d | loss %.4f | Train Sat %.3f | Test Sat %.3f | Train Acc %.3f | Test Acc %.3f ",
                    epoch, train_loss, compute_sat_level(train_loader), compute_sat_level(test_loader),
                    compute_accuracy(train_loader), compute_accuracy(test_loader))



if __name__ == "__main__":
    main()