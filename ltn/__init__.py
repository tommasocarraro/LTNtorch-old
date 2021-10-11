from ltn.core import Variable, Predicate, Constant, Function, Connective, diag, undiag, Quantifier, \
    LTNObject, process_ltn_objects
import ltn.fuzzy_ops
import ltn.utils
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")