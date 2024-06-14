import torch as th


class ExponentialFamilyDistribution:

    # eta: natural parameters of the distribution
    # theta: common parameters of the distribution
    # x: a sample of the distribution

    @classmethod
    def __h_x__(cls, x: list[th.Tensor]) -> th.Tensor:
        """ The base measure h(x).

        Args:
            x (list[th.Tensor]) :  the samples of the distribution. It is a list of tensor of size BS x ?.

        Returns:
            th.Tensor: the value of h(x).
        """
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def __A_eta__(cls, eta: list[th.Tensor]) -> th.Tensor:
        """ The log-partition function A(eta).

        Args:
            eta (list[th.Tensor]): the natural parameters. It is a list of tensors of size K x ?.

        Returns:
            th.Tensor: the value of A(eta).
        """
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def __T_x__(cls, x: list[th.Tensor], idx: int = None) -> list[th.Tensor] | th.Tensor:
        """ The sufficient statistics T(x). If idx is not None, we return only the statistic at in position idx.

        Args:
            x (list[th.Tensor]) :  the samples of the distribution. It is a list of tensor of size BS x ?.
            idx (int): the idx of the parameter we want to retrieve

        Returns:
            list[th.Tensor] : the list of the sufficient statistics T(x). It is a list of tensors if idx is None.
        """
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def expected_T_x(cls, eta: list[th.Tensor], idx: int = None) -> list[th.Tensor] | th.Tensor:
        """ The expected value of the sufficient statitistics T(x), i.e. E_{P(x|eta)}[T(x)].
        The expecation is equal to the derivative of A(eta) w.r.t. the eta.
        If idx is not None, we return only the expextation of the statistic at in position idx.

        Args:
            eta (list[th.Tensor]): the natural paramters. It is a list of tensors of size K x ?.
            idx (int): the idx of the parameter we want to retrieve

        Returns:
             list[th.Tensor]: the expectation of each sufficient statistics. It is a list of tensors if idx is None.
        """
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def natural_to_common(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        """ Transform the natural parameters to the common ones.

        Args:
            eta (list[th.Tensor]): the natural parameters. It is a list of tensors of size K x ?.

        Returns:
             list[th.Tensor]: the common parameters. It is a list of tensors of size K x ? if idx is None
        """
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def common_to_natural(cls, theta: list[th.Tensor]) -> list[th.Tensor]:
        """ Transform the common parameters to the natural ones.

        Args:
            theta (list[th.Tensor]): the common parameters. It is a list of tensors of size K x ?.

        Returns:
             list[th.Tensor]: the natural parameters. It is a list of tensors of size K x ? if idx is None
        """
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def kl_div(cls, p_eta: list[th.Tensor], q_eta: list[th.Tensor]):
        """ Compute the KL divergence beween p and q.

        Args:
            p_eta (list[th.Tensor]): the common parameters of p.
            q_eta (list[th.Tensor]): the common parameters of q.

        Returns:
            th.Tensor: the value of the kl divergence.
        """
        div = 0
        exp_T = cls.expected_T_x(p_eta)
        for i in range(len(p_eta)):
            term = (p_eta[i] - q_eta[i]) * exp_T[i]
            if p_eta[i].ndim > 1:
                term = th.sum(term, list(range(1, p_eta[i].ndim)))
            div += term
        div += -cls.__A_eta__(p_eta) + cls.__A_eta__(q_eta)
        return div
