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
    #print(v)

    d = ltn.Domain((2, 2), "d")

    x = ltn.Variable('x', d, [[[3., 4.], [2., 6.]], [[9.5, 4.9], [3.4, 5.6]]])

    print(x)

    new_const = ltn.Constant('c', d, [[3., 4.], [2., 5.]], trainable=True)

    print(new_const)

    #print(cross_args([x, y, z], flatten_dim0=True))
    # print(cross_grounding_values([con], flatten_dim0=True))
    #print(con)
    modelP = ModelP()
    # P = ltn.Predicate(modelP)
    #print(g)
    #P = ltn.Predicate.MLP([2, 16, 14, 12, 1])
    #print(P(g))


if __name__ == "__main__":
    main()