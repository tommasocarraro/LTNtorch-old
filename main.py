import ltn
import torch


class ModelP(torch.nn.Module):
    def __init__(self):
        super(ModelP, self).__init__()
        self.dense1 = torch.nn.Linear(2, 5)
        self.elu = torch.nn.ELU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dense2 = torch.nn.Linear(5, 1)  # returns one value in [0,1]

    def forward(self, x):
        x = self.elu(self.dense1(x))
        return self.sigmoid(self.dense2(x))

def main():
    d_img = ltn.Domain([2, 2], "d_img")
    l_img = ltn.Domain([3], "l_img")

    c = ltn.Constant('c', d_img, [[4., 5.], [2., 9.]])
    print(c)

    imgs = ltn.Variable('imgs', d_img, [[[3., 4.], [1., 3.]], [[3., 4.], [4., 3.]], [[2., 3.], [4., 9.]], [[3., 4.], [5., 9.]]])
    labels = ltn.Variable('labels', l_img, [[0., 0., 1.], [0., 1., 0.], [1., 0., 0.], [0., 1., 0.]])

    P = ltn.Predicate('p', [d_img, l_img], layers_size=(7, 2, 1))
    print(P)

    points = ltn.Domain([2], "points")
    var_point = ltn.Variable('var_point', points, [[2.1, 3.], [2., 9.]])
    c = ltn.Constant('point', points, [2.1, 3.])
    mu = torch.tensor([2., 3.])

    P1 = ltn.Predicate("p1", [points], lambda_func=lambda x: torch.exp(-torch.norm(x - mu, dim=1)))

    f1 = ltn.Function("f1", [d_img, l_img], d_img, layers_size=(7, 4, 2, 4, 2, 4))

    print(f1([imgs.get_grounding(), labels.get_grounding()]))

    print(P1([var_point.get_grounding()]))

    print(P1([c.get_grounding()]))

    print(P([imgs.get_grounding(), labels.get_grounding()]))

    prop = ltn.PropositionalVariable('prop', [0.4], False)

    print(prop)

    vars = ltn.diag([imgs, labels])

    print(vars)

    vars = ltn.undiag(vars)

    print(vars)


if __name__ == "__main__":
    main()