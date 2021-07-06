import ltn
import torch


class ModelP4(torch.nn.Module):
    def __init__(self):
        super(ModelP4, self).__init__()
        self.elu = torch.nn.ELU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dense1 = torch.nn.Linear(4, 5)
        self.dense2 = torch.nn.Linear(5, 1)  # returns one value in [0,1]

    def forward(self, inputs):
        x = torch.cat(inputs, dim=1)
        x = self.elu(self.dense1(x))
        return self.sigmoid(self.dense2(x))

def main():
    mu = ltn.constant([2., 3.])
    P1 = ltn.Predicate(lambda_func=lambda x: torch.exp(-torch.norm(x[0] - mu, dim=1)))

    c1 = ltn.constant([2.1, 3])
    c2 = ltn.constant([4.5, 0.8])
    v = ltn.variable('v', [[2.1, 3.], [4.5, 0.8]])
    v2 = ltn.variable('v2', [[[3.9, 9.3], [2.9, 9.4]]])
    print(P1(c1))
    print(P1(c2))
    print(P1(v))

    P4 = ltn.Predicate(ModelP4())
    c1 = ltn.constant([2.1, 3])
    c2 = ltn.constant([4.5, 0.8])
    print(P4([c1, c2]))

    P2 = ltn.Predicate(layers_size=(4, 5, 1))
    print(P2([c1, c2]))
    print(P2(v2))



if __name__ == "__main__":
    main()