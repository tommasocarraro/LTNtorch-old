import ltn
import numpy as np
import torch

class ModelC(torch.nn.Module):
    def __init__(self):
        super(ModelC, self).__init__()
        self.dense1 = torch.nn.Linear(4, 5).double()
        self.dense2 = torch.nn.Linear(5, 3).double()
        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax()

    def forward(self, inputs):
        x = inputs[:, :-3]
        l = inputs[:, -3:]
        x = self.dense1(x)
        x = self.dense2(x)
        return torch.sum(x * l, dim=1)

def main():
    np.random.seed(120)
    Not = ltn.WrapperConnective(ltn.fuzzy_ops.NotStandard())
    And = ltn.WrapperConnective(ltn.fuzzy_ops.AndProd())
    Or = ltn.WrapperConnective(ltn.fuzzy_ops.OrProbSum())
    Implies = ltn.WrapperConnective(ltn.fuzzy_ops.ImpliesReichenbach())
    x = ltn.variable('x', np.random.normal(0., 1., (10, 2)))  # 10 values in R²
    y = ltn.variable('y', np.random.normal(0., 2., (5, 2)))  # 5 values in R²

    c1 = ltn.constant([0.5, 0.0])
    c2 = ltn.constant([4.0, 2.0])

    Eq = ltn.Predicate(lambda_func=lambda args: torch.exp(-torch.norm(args[0] - args[1], dim=1)))

    print("Eq([c1, c2])", Eq([c1, c2]))
    print("Not(Eq([c1, c2]))", Not(Eq([c1, c2])))
    print("Implies(Eq([c1, c2]), Eq([c2, c1]))", Implies(Eq([c1, c2]), Eq([c2, c1])))

    print("shape of And(Eq([x, c1]), Eq([x, c2]))", And(Eq([x, c1]), Eq([x, c2])).shape)

    print("shape of Or(Eq([x, c1]), Eq([x, y]))", Or(Eq([x, c1]), Eq([x, y])).shape)

    Forall = ltn.WrapperQuantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantification_type="forall")
    Exists = ltn.WrapperQuantifier(ltn.fuzzy_ops.AggregPMean(p=5), quantification_type="exists")

    print("shape of Eq([x, y])", Eq([x, y]).shape)

    print("shape of Forall(x, Eq([x, y]))", Forall(x, Eq([x, y])).shape)

    print("blocco quantificatori")

    print(Forall([x, y], Eq([x, y])))

    print(Exists([x, y], Eq([x, y])))

    print(Forall(x, Exists(y, Eq([x, y]))))

    print("blocco p differenti")

    print(Forall(x, Eq([x, c1]), p=2))

    # %%
    print(Forall(x, Eq([x, c1]), p=10))

    # %%

    print(Exists(x, Eq([x, c1]), p=2))

    # %%

    print(Exists(x, Eq([x, c1]), p=10))

    print("guarded")

    points = np.random.rand(50, 2)  # 5 values in [0,1]^2
    x = ltn.variable('x', points)
    y = ltn.variable('y', points)
    d = ltn.variable('d', [[.1], [.2], [.3], [.4], [.5], [.6], [.7], [.8], [.9]])

    eucl_dist = lambda x, y: torch.unsqueeze(torch.norm(x - y, dim=1), dim=1)  # function measuring euclidian distance
    print(Exists(d,
           Forall([x, y],
                  Eq([x, y]),
                  mask_vars=[x, y, d],
                  mask_fn=lambda args: eucl_dist(args[0], args[1]) < args[2]
                  )))

    samples = np.random.rand(100, 2, 2)  # 100 R^{2x2} values
    labels = np.random.randint(3, size=100)  # 100 labels (class 0/1/2) that correspond to each sample
    onehot_labels = torch.nn.functional.one_hot(torch.tensor(labels), num_classes=3)

    x = ltn.variable("x", samples)
    l = ltn.variable("l", onehot_labels)

    # TODO sistemare sta cosa che voglio per forza una lista di liste sulle variabili perche' fa diventare matti

    model = ModelC()

    C = ltn.Predicate(model)

    # %%

    print(C([x, l]).shape)  # Computes the 100x100 combinations
    x, l = ltn.diag([x, l])  # sets the diag behavior for x and l
    print(C([x, l]).shape)  # Computes the 100 zipped combinations
    x, l = ltn.undiag([x, l])  # resets the normal behavior
    print(C([x, l]).shape)  # Computes the 100x100 combinations


if __name__ == "__main__":
    main()