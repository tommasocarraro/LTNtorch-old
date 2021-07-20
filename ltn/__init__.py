from ltn.core import variable, Predicate, constant, Function, WrapperConnective, diag, undiag, WrapperQuantifier, propositional_variable
import ltn.fuzzy_ops
import ltn.utils
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")