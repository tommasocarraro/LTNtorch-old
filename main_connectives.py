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
    dom = ltn.Domain([2], 'R^2')
    x = ltn.Variable('x', dom, np.random.normal(0., 1., (10, 2)))  # 10 values in R²
    y = ltn.Variable('y', dom, np.random.normal(0., 2., (5, 2)))  # 5 values in R²

    c1 = ltn.Constant('c1', dom, [0.5, 0.0])
    c2 = ltn.Constant('c2', dom, [4.0, 2.0])

    Eq = ltn.Predicate('sim', [dom, dom], lambda_func=lambda args: torch.exp(-torch.norm(args[0] - args[1], dim=1)))  # predicate measuring similarity

    print("Eq([c1, c2])", Eq([c1.get_grounding(), c2.get_grounding()]))
    print("Not(Eq([c1, c2]))", Not(Eq([c1.get_grounding(), c2.get_grounding()])))
    print("Implies(Eq([c1, c2]), Eq([c2, c1]))", Implies(Eq([c1.get_grounding(), c2.get_grounding()]), Eq([c2.get_grounding(), c1.get_grounding()])))

    print("shape of And(Eq([x, c1]), Eq([x, c2]))", And(Eq([x.get_grounding(), c1.get_grounding()]), Eq([x.get_grounding(), c2.get_grounding()])).shape)

    print("shape of Or(Eq([x, c1]), Eq([x, y]))", Or(Eq([x.get_grounding(), c1.get_grounding()]), Eq([x.get_grounding(), y.get_grounding()])).shape)

    Forall = ltn.WrapperQuantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="forall")
    Exists = ltn.WrapperQuantifier(ltn.fuzzy_ops.AggregPMean(p=5), quantifier="exists")

    print("shape of Eq([x, y])", Eq([x.get_grounding(), y.get_grounding()]).shape)

    print("shape of Forall(x, Eq([x, y]))", Forall(x.get_grounding(), Eq([x.get_grounding(), y.get_grounding()])).shape)

    print("blocco quantificatori")

    print(Forall([x.get_grounding(), y.get_grounding()], Eq([x.get_grounding(), y.get_grounding()])))

    print(Exists([x.get_grounding(), y.get_grounding()], Eq([x.get_grounding(), y.get_grounding()])))

    print(Forall(x.get_grounding(), Exists(y.get_grounding(), Eq([x.get_grounding(), y.get_grounding()]))))

    print("blocco p differenti")

    print(Forall(x.get_grounding(), Eq([x.get_grounding(), c1.get_grounding()]), p=2))

    # %%
    print(Forall(x.get_grounding(), Eq([x.get_grounding(), c1.get_grounding()]), p=10))

    # %%

    print(Exists(x.get_grounding(), Eq([x.get_grounding(), c1.get_grounding()]), p=2))

    # %%

    print(Exists(x.get_grounding(), Eq([x.get_grounding(), c1.get_grounding()]), p=10))

    print("guarded")

    dom_points = ltn.Domain([2], "points")
    dom_var = ltn.Domain([1], 'dom_var')

    points = np.random.rand(50, 2)  # 5 values in [0,1]^2
    x = ltn.Variable('x', dom_points, points)
    y = ltn.Variable('y', dom_points, points)
    d = ltn.Variable('d', dom_var, [[.1], [.2], [.3], [.4], [.5], [.6], [.7], [.8], [.9]])

    eucl_dist = lambda x, y: torch.unsqueeze(torch.norm(x - y, dim=1), dim=1)  # function measuring euclidian distance
    print(Exists(d.get_grounding(),
           Forall([x.get_grounding(), y.get_grounding()],
                  Eq([x.get_grounding(), y.get_grounding()]),
                  mask_vars=[x.get_grounding(), y.get_grounding(), d.get_grounding()],
                  mask_fn=lambda args: eucl_dist(args[0], args[1]) < args[2]
                  )))

    samples = np.random.rand(100, 2, 2)  # 100 R^{2x2} values
    labels = np.random.randint(3, size=100)  # 100 labels (class 0/1/2) that correspond to each sample
    onehot_labels = torch.nn.functional.one_hot(torch.tensor(labels), num_classes=3)

    x_dom = ltn.Domain([2, 2], 'x_dom')
    l_dom = ltn.Domain([3], 'l_dom')

    x = ltn.Variable("x", x_dom, samples)
    l = ltn.Variable("l", l_dom, onehot_labels)

    # TODO sistemare sta cosa che voglio per forza una lista di liste sulle variabili perche' fa diventare matti
    # TODO capire il warning che viene lanciato

    model = ModelC()

    C = ltn.Predicate('c', [x_dom, l_dom], model)

    # %%

    print(C([x.get_grounding(), l.get_grounding()]).shape)  # Computes the 100x100 combinations
    x, l = ltn.diag([x.get_grounding(), l.get_grounding()])  # sets the diag behavior for x and l
    print(C([x, l]).shape)  # Computes the 100 zipped combinations
    x, l = ltn.undiag([x, l])  # resets the normal behavior
    print(C([x, l]).shape)  # Computes the 100x100 combinations


if __name__ == "__main__":
    main()