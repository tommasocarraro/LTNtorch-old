import ltn
import torch
from ltn.core import cross_grounding_values

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
    c = [[3., 2.], [4.3, 5.4]]
    con = ltn.constant(c)
    #print(con)
    v = ltn.variable('x', c)
    #print(v)

    g = ltn.variable('g', [[2., 4.], [5., 3.], [7., 5.]])
    h = ltn.variable('h', [[1., 5., 3.], [3., 4., 2.]])
    l = ltn.variable('l', [[[3., 0.], [1., 7.]], [[3., 5.], [1., 0.]]])

    x = ltn.variable('x', [[2., 4.], [5., 6.]])
    y = ltn.variable('y', [[2., 3.], [4., 7.], [5., 8.]])
    z = ltn.variable('z', [[1., 3., 4.], [2., 7., 9.], [5., 8., 7.], [4., 3., 2.]])

    #print(cross_args([x, y, z], flatten_dim0=True))
    # print(cross_grounding_values([con], flatten_dim0=True))
    #print(con)
    const = ltn.constant([3., 4.])
    modelP = ModelP()
    # P = ltn.Predicate(modelP)
    print(g)
    P = ltn.Predicate.MLP([2, 16, 14, 12, 1])
    print(P(g))


if __name__ == "__main__":
    main()