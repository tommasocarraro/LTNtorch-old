import ltn
import torch


class ModelP(torch.nn.Module):
    def __init__(self):
        super(ModelP, self).__init__()
        self.dense1 = torch.nn.Linear(7, 5)
        self.elu = torch.nn.ELU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dense2 = torch.nn.Linear(5, 1)  # returns one value in [0,1]

    def forward(self, x):
        x = self.elu(self.dense1(x))
        return self.sigmoid(self.dense2(x))


class ModelF(torch.nn.Module):
    def __init__(self):
        super(ModelF, self).__init__()
        self.dense1 = torch.nn.Linear(7, 5)
        self.elu = torch.nn.ELU()
        self.dense2 = torch.nn.Linear(5, 4)

    def forward(self, x):
        x = self.elu(self.dense1(x))
        return self.dense2(x)

def main():
    imgs = ltn.variable('imgs', [[[3., 4.], [1., 3.]], [[3., 4.], [4., 3.]], [[2., 3.], [4., 9.]], [[3., 4.], [5., 9.]]])
    labels = ltn.variable('labels', [[0., 0., 1.], [0., 1., 0.], [1., 0., 0.], [0., 1., 0.]])

    P = ltn.Predicate(layers_size=(7, 2, 1))
    print(P([imgs, labels]))
    P = ltn.Predicate(ModelP())
    print(P([imgs, labels]))
    mu = ltn.constant([2., 3.])
    P = ltn.Predicate(lambda_func=lambda x: torch.exp(-torch.norm(x - mu, dim=1)))
    c1 = ltn.constant([2.1, 3])
    print(P(c1))

    f1 = ltn.Function(lambda_func=lambda args: args[0] - args[1])

    c1 = ltn.constant([2.1, 3])
    c2 = ltn.constant([4.5, 0.8])

    print(f1([c1, c2], output_dim=1).shape)

    f2 = ltn.Function(ModelF())

    f3 = ltn.Function(layers_size=(7, 5, 4))

    print(f2([imgs, labels], output_dim=(2, 2)).shape)

    print(f3([imgs, labels], output_dim=(2, 2)).shape)



if __name__ == "__main__":
    main()