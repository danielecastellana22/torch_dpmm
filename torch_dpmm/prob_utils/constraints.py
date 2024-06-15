import torch as th
from torch.linalg import eigvalsh

__all__ = ['BaseConstraint', 'AnyValue', 'GreaterThan', 'Positive', 'PositiveDefinite']


class BaseConstraint:

    def __call__(self, v):
        raise NotImplementedError

    def message(self, param_name, distr_name):
        raise NotImplementedError


class AnyValue(BaseConstraint):

    def __call__(self, v):
        return True


class GreaterThan(BaseConstraint):

    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def __call__(self, v):
        return th.all(v > self.lower_bound)

    def message(self, param_name, distr_name):
        return f'Params {param_name} of {distr_name} must be strictly greater than {self.lower_bound}.'


class Positive(GreaterThan):

    def __init__(self):
        super().__init__(0)


class PositiveDefinite(BaseConstraint):

    def __call__(self, v):
        return th.all(eigvalsh(v) > 0)

    def message(self, param_name, distr_name):
        return f'Params {param_name} of {distr_name} must be positive definite.'
