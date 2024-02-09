import numpy as np
import scipy
from scipy.special import gammaln, digamma
from .bags import ParamBag, SuffStatBag

LOGTWOPI = np.log(2) + np.log(np.pi)
LOGTWO = np.log(2)

def calcELBO(**kwargs):
    """ Calculate ELBO objective for provided model state.
    """
    Llinear = calcELBO_LinearTerms(**kwargs)
    Lnon = calcELBO_NonlinearTerms(**kwargs)
    if 'todict' in kwargs and kwargs['todict']:
        assert isinstance(Llinear, dict)
        Llinear.update(Lnon)
        return Llinear
    return Lnon + Llinear


def calcELBO_LinearTerms(SS=None,
                         N=None,
                         eta1=None, eta0=None, ElogU=None, Elog1mU=None,
                         gamma1=1.0, gamma0=None,
                         afterGlobalStep=0, todict=0, **kwargs):
    """ Calculate ELBO objective terms that are linear in suff stats.
    """
    if SS is not None:
        N = SS.N
    K = N.size
    Lglobal = K * c_Beta(gamma1, gamma0) - c_Beta(eta1, eta0)
    if afterGlobalStep:
        if todict:
            return dict(Lalloc=Lglobal)
        return Lglobal
    # Slack term only needed when not immediately after a global step.
    N0 = convertToN0(N)
    if ElogU is None or Elog1mU is None:
        ElogU, Elog1mU = calcBetaExpectations(eta1, eta0)
    Lslack = np.inner(N + gamma1 - eta1, ElogU) + \
        np.inner(N0 + gamma0 - eta0, Elog1mU)
    if todict:
        return dict(Lalloc=Lglobal)
    return Lglobal + Lslack


def calcELBO_NonlinearTerms(SS=None, LP=None,
                            resp=None, Hresp=None,
                            returnMemoizedDict=0, todict=0, **kwargs):
    """ Calculate ELBO objective terms non-linear in suff stats.
    """
    if Hresp is None:
        if SS is not None and SS.hasELBOTerm('Hresp'):
            Hresp = SS.getELBOTerm('Hresp')
        else:
            Hresp = calcHrespFromLP(LP=LP, resp=resp)
    if returnMemoizedDict:
        return dict(Hresp=Hresp)
    Lentropy = np.sum(Hresp)
    if SS is not None and SS.hasAmpFactor():
        Lentropy *= SS.ampF
    if todict:
        return dict(Lentropy=Lentropy)
    return Lentropy

def calcRlogR_numpy_vectorized(R):
    """ Compute sum over columns of R * log(R). O(NK) memory. Vectorized.

    Args
    ----
    R : 2D array, N x K
        Each row must have entries that are strictly positive (> 0).
        No bounds checking is enforced!

    Returns
    -------
    H : 1D array, size K
        H[k] = np.sum(R[:,k] * log R[:,k])
    """
    EPS = 10 * np.finfo(float).eps
    H = np.sum(R * np.log(R+EPS), axis=0)
    return H


def calcHrespFromLP(LP=None, resp=None):
    if LP is not None and 'spR' in LP:
        nnzPerRow = LP['nnzPerRow']
        if nnzPerRow > 1:
            # Handles multiply by -1 already
            Hresp = calcSparseRlogR(**LP)
            assert np.all(np.isfinite(Hresp))
        else:
            Hresp = 0.0
    else:
        if LP is not None and 'resp' in LP:
            resp = LP['resp']
        Hresp = -1 * calcRlogR_numpy_vectorized(resp) #NumericUtil.calcRlogR(resp)
    return Hresp


def calcELBOGain_NonlinearTerms(beforeSS=None, afterSS=None):
    """ Compute gain in ELBO score by transition from before to after values.
    """
    L_before = beforeSS.getELBOTerm('Hresp').sum()
    L_after = afterSS.getELBOTerm('Hresp').sum()
    return L_after - L_before


def convertToN0(N):
    """ Convert count vector to vector of "greater than" counts.

    Parameters
    -------
    N : 1D array, size K
        each entry k represents the count of items assigned to comp k.

    Returns
    -------
    N0 : 1D array, size K
        each entry k gives the total count of items at index above k
        N0[k] = np.sum(N[k:])

    Example
    -------
    >>> convertToN0([1., 3., 7., 2])
    array([12.,  9.,  2.,  0.])
    """
    N = np.asarray(N)
    N0 = np.zeros_like(N)
    N0[:-1] = N[::-1].cumsum()[::-1][1:]
    return N0


def c_Beta(eta1, eta0):
    ''' Evaluate cumulant function of Beta distribution

    Parameters
    -------
    eta1 : 1D array, size K
        represents ON pseudo-count parameter of the Beta
    eta0 : 1D array, size K
        represents OFF pseudo-count parameter of the Beta

    Returns
    -------
    c : float
        = \sum_k c_B(eta1[k], eta0[k])
    '''
    return np.sum(gammaln(eta1 + eta0) - gammaln(eta1) - gammaln(eta0))


def c_Beta_ReturnVec(eta1, eta0):
    ''' Evaluate cumulant of Beta distribution for vector of parameters

    Parameters
    -------
    eta1 : 1D array, size K
        represents ON pseudo-count parameter of the Beta
    eta0 : 1D array, size K
        represents OFF pseudo-count parameter of the Beta

    Returns
    -------
    cvec : 1D array, size K
    '''
    return gammaln(eta1 + eta0) - gammaln(eta1) - gammaln(eta0)


def calcBetaExpectations(eta1, eta0):
    ''' Evaluate expected value of log u under Beta(u | eta1, eta0)

    Returns
    -------
    ElogU : 1D array, size K
    Elog1mU : 1D array, size K
    '''
    digammaBoth = digamma(eta0 + eta1)
    ElogU = digamma(eta1) - digammaBoth
    Elog1mU = digamma(eta0) - digammaBoth
    return ElogU, Elog1mU


def calcCachedELBOGap_SinglePair(SS, kA, kB,
                                 delCompID=None, dtargetMinCount=None):
    """ Compute (lower bound on) gap in cacheable ELBO

    Returns
    ------
    gap : scalar
        L'_entropy - L_entropy >= gap
    """
    assert SS.hasELBOTerms()
    # Hvec : 1D array, size K
    Hvec = -1 * SS.getELBOTerm('ElogqZ')
    if delCompID is None:
        # Use bound - r log r >= 0
        gap = -1 * (Hvec[kA] + Hvec[kB])
    else:
        # Use bound - (1-r) log (1-r) >= r for small values of r
        assert delCompID == kA or delCompID == kB
        gap1 = -1 * Hvec[delCompID] - SS.N[delCompID]
        gap2 = -1 * (Hvec[kA] + Hvec[kB])
        gap = np.maximum(gap1, gap2)
    return gap


def calcCachedELBOTerms_SinglePair(SS, kA, kB, delCompID=None):
    """ Calculate all cached ELBO terms under proposed merge.
    """
    assert SS.hasELBOTerms()
    # Hvec : 1D array, size K
    Hvec = -1 * SS.getELBOTerm('ElogqZ')
    newHvec = np.delete(Hvec, kB)
    if delCompID is None:
        newHvec[kA] = 0
    else:
        assert delCompID == kA or delCompID == kB
        if delCompID == kA:
            newHvec[kA] = Hvec[kB]
        newHvec[kA] -= SS.N[delCompID]
        newHvec[kA] = np.maximum(0, newHvec[kA])
    return dict(ElogqZ=-1 * newHvec)


def inplaceExpAndNormalizeRows_numpy(R):
    ''' Compute exp(R), normalize rows to sum to one, and set min val.

    Post Condition
    --------
    Each row of R sums to one.
    Minimum value of R is equal to minVal.
    '''
    R -= np.max(R, axis=1)[:, np.newaxis]
    np.exp(R, out=R)
    R /= R.sum(axis=1)[:, np.newaxis]


def calcLocalParams(Data, LP, Elogbeta=None, nnzPerRowLP=None, **kwargs):
    ''' Compute local parameters for each data item.

    Parameters
    -------
    Data : bnpy.data.DataObj subclass

    LP : dict
        Local parameters as key-value string/array pairs
        * E_log_soft_ev : 2D array, N x K
            E_log_soft_ev[n,k] = log p(data obs n | comp k)

    Returns
    -------
    LP : dict
        Local parameters, with updated fields
        * resp : 2D array, size N x K array
            Posterior responsibility each comp has for each item
            resp[n, k] = p(z[n] = k | x[n])
    '''
    lpr = LP['E_log_soft_ev']
    lpr += Elogbeta
    K = LP['E_log_soft_ev'].shape[1]
    if nnzPerRowLP and (nnzPerRowLP > 0 and nnzPerRowLP < K):
        # SPARSE Assignments
        LP['spR'] = sparsifyLogResp(lpr, nnzPerRow=nnzPerRowLP)
        assert np.all(np.isfinite(LP['spR'].data))
        LP['nnzPerRow'] = nnzPerRowLP
    else:
        # DENSE Assignments
        # Calculate exp in numerically stable manner (first subtract the max)
        #  perform this in-place so no new allocations occur
        #NumericUtil.inplaceExpAndNormalizeRows(lpr)
        inplaceExpAndNormalizeRows_numpy(lpr)
        LP['resp'] = lpr
    return LP


def calcSummaryStats(Data, LP,
                     doPrecompEntropy=False,
                     doPrecompMergeEntropy=False, mPairIDs=None,
                     mergePairSelection=None,
                     trackDocUsage=False,
                     **kwargs):
    ''' Calculate sufficient statistics for global updates.

    Parameters
    -------
    Data : bnpy data object
    LP : local param dict with fields
        resp : Data.nObs x K array,
            where resp[n,k] = posterior resp of comp k
    doPrecompEntropy : boolean flag
        indicates whether to precompute ELBO terms in advance
        used for memoized learning algorithms (moVB)
    doPrecompMergeEntropy : boolean flag
        indicates whether to precompute ELBO terms in advance
        for certain merge candidates.

    Returns
    -------
    SS : SuffStatBag with K components
        Summarizes for this mixture model, with fields
        * N : 1D array, size K
            N[k] = expected number of items assigned to comp k

        Also has optional ELBO field when precompELBO is True
        * ElogqZ : 1D array, size K
            Vector of entropy contributions from each comp.
            ElogqZ[k] = \sum_{n=1}^N resp[n,k] log resp[n,k]

        Also has optional Merge field when precompMergeELBO is True
        * ElogqZ : 2D array, size K x K
            Each term is scalar entropy of merge candidate
    '''
    if mPairIDs is not None and len(mPairIDs) > 0:
        M = len(mPairIDs)
    else:
        M = 0
    if 'resp' in LP:
        Nvec = np.sum(LP['resp'], axis=0)
        K = Nvec.size
    else:
        # Sparse assignment case
        Nvec = as1D(toCArray(LP['spR'].sum(axis=0)))
        K = LP['spR'].shape[1]

    if hasattr(Data, 'dim'):
        SS = SuffStatBag(K=K, D=Data.dim, M=M)
    else:
        SS = SuffStatBag(K=K, D=Data.vocab_size, M=M)
    SS.setField('N', Nvec, dims=('K'))
    if doPrecompEntropy:
        Mdict = calcELBO_NonlinearTerms(LP=LP, returnMemoizedDict=1)
        if type(Mdict['Hresp']) == float:
            # SPARSE HARD ASSIGNMENTS
            SS.setELBOTerm('Hresp', Mdict['Hresp'], dims=None)
        else:
            SS.setELBOTerm('Hresp', Mdict['Hresp'], dims=('K',))

    if doPrecompMergeEntropy:
        m_Hresp = None
        if 'resp' in LP:
            m_Hresp = -1 * NumericUtil.calcRlogR_specificpairs(
                LP['resp'], mPairIDs)
        elif 'spR' in LP:
            if LP['nnzPerRow'] > 1:
                m_Hresp = calcSparseMergeRlogR(
                    spR_csr=LP['spR'],
                    nnzPerRow=LP['nnzPerRow'],
                    mPairIDs=mPairIDs)
        else:
            raise ValueError("Need resp or spR in LP")
        if m_Hresp is not None:
            assert m_Hresp.size == len(mPairIDs)
            SS.setMergeTerm('Hresp', m_Hresp, dims=('M'))
    if trackDocUsage:
        Usage = np.sum(LP['resp'] > 0.01, axis=0)
        SS.setSelectionTerm('DocUsageCount', Usage, dims='K')

    return SS


class Mock_DP:

    def __init__(self, K, eta1, eta0, gamma1, gamma0):
        self.K = K
        self.gamma1 = gamma1
        self.gamma0 = gamma0
        self.eta1 = eta1
        self.eta0 = eta0
        self.ElogU, self.Elog1mU = calcBetaExpectations(self.eta1, self.eta0)

        # Calculate expected mixture weights E[ log \beta_k ]
        # Using copy() allows += without modifying ElogU
        self.Elogbeta = self.ElogU.copy()
        self.Elogbeta[1:] += self.Elog1mU[:-1].cumsum()

    def calc_local_params(self, Data, LP, **kwargs):
        ''' Compute local parameters for provided data.

        Args
        ----
        Data : :class:`bnpy.data.DataObj`
        LP : dict
            Local parameters as key-value string/array pairs
            * E_log_soft_ev : 2D array, N x K
                E_log_soft_ev[n,k] = log p(data obs n | comp k)

        Returns
        -------
        LP : dict
            Local parameters, with updated fields
            * resp : 2D array, size N x K array
                Posterior responsibility each comp has for each item
                resp[n, k] = p(z[n] = k | x[n])
        '''
        return calcLocalParams(
            Data, LP, Elogbeta=self.Elogbeta, **kwargs)

    def get_global_suff_stats(self, Data, LP,
                              **kwargs):
        ''' Calculate sufficient statistics for global updates.

        Parameters
        -------
        Data : bnpy data object
        LP : local param dict with fields
            resp : Data.nObs x K array,
                where resp[n,k] = posterior resp of comp k
        doPrecompEntropy : boolean flag
            indicates whether to precompute ELBO terms in advance
            used for memoized learning algorithms (moVB)
        doPrecompMergeEntropy : boolean flag
            indicates whether to precompute ELBO terms in advance
            for certain merge candidates.

        Returns
        -------
        SS : SuffStatBag with K components
            Summarizes for this mixture model, with fields
            * N : 1D array, size K
                N[k] = expected number of items assigned to comp k

            Also has optional ELBO field when precompELBO is True
            * ElogqZ : 1D array, size K
                Vector of entropy contributions from each comp.
                ElogqZ[k] = \sum_{n=1}^N resp[n,k] log resp[n,k]

            Also has optional Merge field when precompMergeELBO is True
            * ElogqZ : 2D array, size K x K
                Each term is scalar entropy of merge candidate
        '''
        return calcSummaryStats(Data, LP, **kwargs)

    def calc_evidence(self, Data, SS, LP=None, todict=0, **kwargs):
        """ Calculate ELBO objective function value for provided state.

        Returns
        -------
        L : float
        """
        return calcELBO(SS=SS, LP=LP,
                        eta1=self.eta1, eta0=self.eta0,
                        ElogU=self.ElogU, Elog1mU=self.Elog1mU,
                        gamma1=self.gamma1, gamma0=self.gamma0,
                        todict=todict,
                        **kwargs)

    def update_global_params(self, SS, rho=None, **kwargs):
        """ Update eta1, eta0 to optimize the ELBO objective.

        Post Condition for VB
        -------
        eta1 and eta0 set to valid posterior for SS.K components.
        """
        N = SS.getCountVec()
        self.K = SS.K
        eta1 = self.gamma1 + N
        eta0 = self.gamma0 + convertToN0(N)
        self.eta1 = eta1
        self.eta0 = eta0
        #self.set_helper_params()