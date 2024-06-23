import torch as th
from torch_dpmm.prob_tools.utils import batch_outer_product, batched_trace_square_mat
from typing import Optional, Union


class BaseMatOps:

    @classmethod
    def det(cls, M):
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def log_det(cls, M):
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def inv_M(cls, M):
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def inv_M_v(cls, M, v):
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def vT_inv_M_v(cls, M, v):
        inv_M_v = cls.inv_M_v(M, v)
        return th.sum(v * inv_M_v, -1)

    @classmethod
    def inv_M_v_vT(cls, M, v):
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def trace_M(cls, M):
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def v_vT(cls, v):
        raise NotImplementedError('Should be implemented in subclasses!')

class FullMatOps(BaseMatOps):

    @classmethod
    def det(cls, M):
        return th.det(M)

    @classmethod
    def log_det(cls, M):
        return th.logdet(M)

    @classmethod
    def inv_M(cls, M):
        return th.linalg.inv(M)

    @classmethod
    def inv_M_v(cls, M, v):
        return th.linalg.solve(M, v)

    @classmethod
    def inv_M_v_vT(cls, M, v):
        v_vT = batch_outer_product(v, v)
        return th.linalg.solve(M, v_vT)

    @classmethod
    def trace_M(cls, M):
        return batched_trace_square_mat(M)

    @classmethod
    def v_vT(cls, v):
        return batch_outer_product(v ,v)


class DiagonalMatOps(BaseMatOps):

    @classmethod
    def det(cls, M):
        return th.prod(M, -1)

    @classmethod
    def log_det(cls, M):
        return th.sum(th.log(M), -1)

    @classmethod
    def inv_M(cls, M):
        return 1/M

    @classmethod
    def inv_M_v(cls, M, v):
        return v / M

    @classmethod
    def inv_M_v_vT(cls, M, v):
        return v**2 / M

    @classmethod
    def trace_M(cls, M):
        return th.sum(M, -1)

    @classmethod
    def v_vT(cls, v):
        return v**2


# TODO: implement choleksy parametrisation, low-rank parametrisation
class CholeksyMatOps(BaseMatOps):

    @classmethod
    def batch_cholesky_update(cls, L: th.Tensor, x: th.Tensor,  beta: Optional[th.Tensor | float] = 1.0) -> th.Tensor:
        '''
        This function computes the Cholesky factor of (LL^T + beta*xx^T). It supports batch updates.
        The code is mostly taken from https://brentyi.github.io/fannypack/utils/potpourri/#fannypack.utils.cholupdate
        Args:
            L: tensors of size (*, D, D) which contains batched Cholesky facotrs.
            beta: tensors of size * which contains the weights for the updates.
            x: tensors of size (*, D)

        Returns:
            (th.Tensor): The Cholesky factor of (LL^T + beta*xx^T).
        '''
        # Expected shapes: (*, dim, dim) and (*, dim)
        batch_dims = L.shape[:-2]
        matrix_dim = x.shape[-1]
        assert x.shape[:-1] == batch_dims
        assert matrix_dim == L.shape[-1] == L.shape[-2]

        # Flatten batch dimensions, and clone for tensors we need to mutate
        L = L.reshape((-1, matrix_dim, matrix_dim))
        x = x.reshape((-1, matrix_dim)).clone()
        L_out_cols = []

        sign_out: Union[float, th.Tensor]

        if isinstance(beta, float):
            beta = th.tensor(beta, device=x.device)

        x = x * th.sqrt(th.abs(beta))
        sign_out = th.sign(beta)

        # Cholesky update; mostly copied from Wikipedia:
        # https://en.wikipedia.org/wiki/Cholesky_decomposition
        for k in range(matrix_dim):
            r = th.sqrt(L[:, k, k] ** 2 + sign_out * x[:, k] ** 2)
            c = (r / L[:, k, k])[:, None]
            s = (x[:, k] / L[:, k, k])[:, None]

            # We build output column-by-column to avoid in-place modification errors
            L_out_col = th.zeros_like(L[:, :, k])
            L_out_col[:, k] = r
            L_out_col[:, k + 1:] = (L[:, k + 1:, k] + sign_out * s * x[:, k + 1:]) / c
            L_out_cols.append(L_out_col)

            # We clone x at each iteration, also to avoid in-place modification errors
            x_next = x.clone()
            x_next[:, k + 1:] = c * x[:, k + 1:] - s * L_out_col[:, k + 1:]
            x = x_next

        # Stack columns together
        L_out = th.stack(L_out_cols, dim=2)

        # Unflatten batch dimensions and return
        return L_out.reshape(batch_dims + (matrix_dim, matrix_dim))