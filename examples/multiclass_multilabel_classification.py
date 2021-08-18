"""
Here, you can find the the multiclass multi label classification example of the LTN paper. Please, carefully read the
example on the paper before going through the PyTorch example.
"""
import logging
import torch
import numpy as np
import pandas as pd
import ltn
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


# this is a standard PyTorch DataLoader to load the dataset for the training and testing of the model
class DataLoader(object):
    def __init__(self,
                 data,
                 labels_sex,
                 labels_color,
                 batch_size=1,
                 shuffle=True):
        self.data = data
        self.labels_sex = labels_sex
        self.labels_color = labels_color
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
            labels_sex = self.labels_sex[idxlist[start_idx:end_idx]]
            labels_color = self.labels_color[idxlist[start_idx:end_idx]]

            yield data, labels_sex, labels_color


class MLP(torch.nn.Module):
    """
    This model returns the logits given an input. It does not compute the softmax. The output are not normalized.
    This is done to separate the accuracy computation from the satisfaction level computation. Go through the example
    to understand it.
    """
    def __init__(self, layer_sizes=(5, 16, 16, 8, 4)):
        super(MLP, self).__init__()
        self.elu = torch.nn.ELU()
        self.dropout = torch.nn.Dropout(0.2)
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i])
                                                  for i in range(1, len(layer_sizes))])

    def forward(self, inputs, training=False):
        x = inputs
        for layer in self.linear_layers[:-1]:
            x = self.elu(layer(x))
            if training:
                x = self.dropout(x)
        return self.linear_layers[-1](x)


def main():
    np.random.seed(12)
    # # Data
    #
    # Crabs dataset from: http://www.stats.ox.ac.uk/pub/PRNN/
    #
    # The crabs data frame has 200 rows and 8 columns, describing 5 morphological measurements on 50 crabs each of
    # two colour forms and both sexes, of the species Leptograpsus variegatus collected at Fremantle, W. Australia.
    #
    # - Multi-class: Male, Female, Blue, Orange.
    # - Multi-label: Only Male-Female and Blue-Orange are mutually exclusive.
    #
    df = pd.read_csv("datasets/crabs.dat", sep=" ", skipinitialspace=True)
    df = df.sample(frac=1)  # shuffle dataset

    features = np.array(df[['FL', 'RW', 'CL', 'CW', 'BD']]).astype(np.float32)
    labels_sex = np.array(df['sex'])
    labels_color = np.array(df['sp'])

    # create data loaders for training and testing the model
    train_loader = DataLoader(features[:160], labels_sex[:160], labels_color[:160], 64, True)
    test_loader = DataLoader(features[:160], labels_sex[:160], labels_color[:160], 64)

    # # LTN
    #
    # ### Predicate
    #
    # | index | class |
    # | --- | --- |
    # | 0 | Male |
    # | 1 | Female |
    # | 2 | Blue |
    # | 3 | Orange |
    #
    # Let's note that, since the classes are not mutually exclusive, the last layer of the model will be a `sigmoid`
    # and not a `softmax`.

    logits_model = MLP()
    p = ltn.Predicate(ltn.utils.LogitsToPredicateModel(logits_model, single_label=False))

    # Constants to index the classes
    class_male = ltn.constant(0)
    class_female = ltn.constant(1)
    class_blue = ltn.constant(2)
    class_orange = ltn.constant(3)


    # ### Axioms
    #
    # ```
    # forall x_blue: C(x_blue,blue)
    # forall x_orange: C(x_orange,orange)
    # forall x_male: C(x_male,male)
    # forall x_female: C(x_female,female)
    # forall x: ~(C(x,male) & C(x,female))
    # forall x: ~(C(x,blue) & C(x,orange))
    # ```

    Not = ltn.WrapperConnective(ltn.fuzzy_ops.NotStandard())
    And = ltn.WrapperConnective(ltn.fuzzy_ops.AndProd())
    Implies = ltn.WrapperConnective(ltn.fuzzy_ops.ImpliesGoguenStrong())
    Forall = ltn.WrapperQuantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantification_type="forall")

    formula_aggregator = ltn.fuzzy_ops.AggregPMeanError(p=2)

    # this function defines the variables and the axioms that need to be used to train the predicate P
    # it returns the satisfaction level of the given knowledge base (axioms)
    def axioms(features, labels_sex, labels_color, training=False):
        x = ltn.variable("x", features)
        x_blue = ltn.variable("x_blue", features[labels_color == "B"])
        x_orange = ltn.variable("x_orange", features[labels_color == "O"])
        x_male = ltn.variable("x_blue", features[labels_sex == "M"])
        x_female = ltn.variable("x_blue", features[labels_sex == "F"])
        axioms = [
            Forall(x_blue, p([x_blue, class_blue], training)),
            Forall(x_orange, p([x_orange, class_orange], training)),
            Forall(x_male, p([x_male, class_male], training)),
            Forall(x_female, p([x_female, class_female], training)),
            Forall(x, Not(And(p([x, class_blue], training), p([x, class_orange], training)))),
            Forall(x, Not(And(p([x, class_male], training), p([x, class_female], training))))
        ]
        axioms = torch.stack(ltn.Grounding.convert_groundings_to_tensors(axioms))
        sat_level = formula_aggregator(axioms, dim=0)
        return sat_level

    # define metrics

    # it computes the overall satisfaction level on the knowledge base using the given data loader (train or test)
    def compute_sat_level_axioms(loader):
        mean_sat = 0
        for features, labels_sex, labels_color in loader:
            mean_sat += axioms(features, labels_sex, labels_color)
        mean_sat /= len(loader)
        return mean_sat

    # it computes the satisfaction level of a formula phi using the given data loader (train or test)
    def compute_sat_level_phi(loader, phi):
        mean_sat = 0
        for features, _, _ in loader:
            mean_sat += phi(features)
        mean_sat /= len(loader)
        return mean_sat

    # it computes the overall accuracy on the given data loader (train or test)
    # since it is a multilabel task, it uses the hamming loss to compute the accuracy
    def compute_accuracy(loader, threshold=0.5, from_logits=False):
        mean_accuracy = 0.0
        for features, labels_sex, labels_color in loader:
            predictions = logits_model(torch.tensor(features)).detach().numpy()
            labels_male = (labels_sex == "M")
            labels_female = (labels_sex == "F")
            labels_blue = (labels_color == "B")
            labels_orange = (labels_color == "O")
            onehot = np.stack([labels_male, labels_female, labels_blue, labels_orange], axis=-1).astype(np.int32)
            if not from_logits:
                predictions = torch.nn.Sigmoid()(predictions)
            predictions = predictions > threshold
            predictions = predictions.astype(np.int32)
            nonzero = np.count_nonzero(onehot - predictions, axis=-1).astype(np.float32)
            multilabel_hamming_loss = nonzero / predictions.shape[-1]
            mean_accuracy += np.mean(1 - multilabel_hamming_loss)

        return mean_accuracy / len(loader)

    # # Training
    #
    # While training, we measure:
    # 1. The level of satisfiability of the Knowledge Base of the training data.
    # 2. The level of satisfiability of the Knowledge Base of the test data.
    # 3. The training accuracy.
    # 4. The test accuracy.
    # 5. The level of satisfiability of a formula phi_1 we expect to have a high truth value.
    #       forall x (p(x,blue)->~p(x,orange))
    # 6. The level of satisfiability of a formula phi_2 we expect to have a low truth value.
    #       forall x (p(x,blue)->p(x,orange))
    # 7. The level of satisfiability of a formula phi_3 we expect to have a neither high neither low truth value.
    #       forall x (p(x,blue)->p(x,male))

    # it returns the satisfaction level of the formula phi1 (querying). In the case of a formula, the satisfaction level
    # represents the truth value of the formula
    def phi1(features):
        x = ltn.variable("x", features)
        return Forall(x, Implies(p([x, class_blue]), Not(p([x, class_orange]))), p=5).tensor

    # it returns the satisfaction level of the formula phi2 (querying)
    def phi2(features):
        x = ltn.variable("x", features)
        return Forall(x, Implies(p([x, class_blue]), p([x, class_orange])), p=5).tensor

    # it returns the satisfaction level of the formula phi3 (querying)
    def phi3(features):
        x = ltn.variable("x", features)
        return Forall(x, Implies(p([x, class_blue]), p([x, class_male])), p=5).tensor

    optimizer = torch.optim.Adam(p.parameters(), lr=0.001)

    # training of the LTN model
    for epoch in range(500):
        train_loss = 0.0
        for batch_idx, (data, labels_sex, labels_color) in enumerate(train_loader):
            optimizer.zero_grad()
            sat_agg = axioms(data, labels_sex, labels_color, training=False)
            loss = 1. - sat_agg
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)

        # we print metrics every 20 epochs of training
        if epoch % 20 == 0:
            logger.info(" epoch %d | loss %.4f | Train Sat %.3f | Test Sat %.3f | Train Acc %.3f | Test Acc %.3f | "
                        "Test Sat Phi 1 %.3f | Test Sat Phi 2 %.3f | Test Sat Phi 3 %.3f ",
                        epoch, train_loss, compute_sat_level_axioms(train_loader),
                        compute_sat_level_axioms(test_loader),
                        compute_accuracy(train_loader, from_logits=True), compute_accuracy(test_loader, from_logits=True),
                        compute_sat_level_phi(test_loader, phi1), compute_sat_level_phi(test_loader, phi2),
                        compute_sat_level_phi(test_loader, phi3))


if __name__ == "__main__":
    main()
