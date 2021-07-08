import torch
import ltn

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
        logits_model: a tf Model that outputs logits
        single_label: True for exclusive classes (logits are translated into probabilities using softmax),
                False for non-exclusive classes (logits are translated into probabilities using sigmoid)
        """
        super(LogitsToPredicateModel, self).__init__()
        self.logits_model = logits_model
        self.to_probs = torch.nn.Softmax(dim=1) if single_label else torch.nn.Sigmoid(dim=1)

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
        indices = torch.stack(inputs[1:], dim=1).long()
        return torch.gather(probs, 1, indices)