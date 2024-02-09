import numpy as np
import scipy
from scipy.special import gammaln, digamma
from .bags import SuffStatBag

LOGTWOPI = np.log(2) + np.log(np.pi)
LOGTWO = np.log(2)

# we take code this code from bnpy to test the correcteness of our implementation.
# https://github.com/bnpy/bnpy

try:
    import scipy.linalg.blas
    try:
        fblas = scipy.linalg.blas.fblas
    except AttributeError:
        # Scipy changed location of BLAS libraries in late 2012.
        # See http://github.com/scipy/scipy/pull/358
        fblas = scipy.linalg.blas._fblas
except:
    raise ImportError(
        "BLAS libraries for efficient matrix multiplication not found")


def dotATB(A, B):
    ''' Compute matrix product A.T * B
        using efficient BLAS routines (low-level machine code)
    '''
    if A.shape[1] > B.shape[1]:
        return fblas.dgemm(1.0, A, B, trans_a=True)
    else:
        return np.dot(A.T, B)


def dotABT(A, B):
    ''' Compute matrix product A* B.T
        using efficient BLAS routines (low-level machine code)
    '''
    if B.shape[0] > A.shape[0]:
        return fblas.dgemm(1.0, A, B, trans_b=True)
    else:
        return np.dot(A, B.T)


def dotATA(A):
    ''' Compute matrix product A.T * A
        using efficient BLAS routines (low-level machine code)
    '''
    return fblas.dgemm(1.0, A.T, A.T, trans_b=True)


def c_Diff(nu1, logdetB1, m1, kappa1,
           nu2, logdetB2, m2, kappa2):
    ''' Evaluate difference of cumulant functions c(params1) - c(params2)

    May be more numerically stable than directly using c_Func
    to find the difference.

    Returns
    -------
    diff : scalar real value of the difference in cumulant functions
    '''
    if logdetB1.ndim >= 2:
        logdetB1 = np.log(np.linalg.det(logdetB1))
    if logdetB2.ndim >= 2:
        logdetB2 = np.log(np.linalg.det(logdetB2))
    D = m1.size
    dvec = np.arange(1, D + 1, dtype=float)
    return - 0.5 * D * LOGTWO * (nu1 - nu2) \
           - np.sum(gammaln(0.5 * (nu1 + 1 - dvec))) \
        + np.sum(gammaln(0.5 * (nu2 + 1 - dvec))) \
        + 0.5 * D * (np.log(kappa1) - np.log(kappa2)) \
        + 0.5 * (nu1 * logdetB1 - nu2 * logdetB2)

class Mock_Gauss:

    def __init__(self, K, D, post_params, prior_params):
        self.D = D
        self.K = K
        self.Post = post_params
        self.Prior = prior_params
        self.Cache = {}
        self.Cache['cholB'] = self._cholB('all').astype(float)
        self.Cache['E_logdetL'] = self._E_logdetL('all')
        self.inferType = 'VB'

    def GetCached(self, key, k=None):
        #if n in self.cache:
        #    return self.cache[n][i]
        #else:
        #    return self.__getattribute__('_'+n)(i)

        ckey = key + '-' + str(k)
        try:
            return self.Cache[ckey]
        except KeyError:
            Val = getattr(self, '_' + key)(k)
            self.Cache[ckey] = Val
            return Val

    def calc_evidence(self, Data, SS, LP, todict=0, **kwargs):
        """ Evaluate objective function at provided state.

        Returns
        -----
        L : float
        """
        if self.inferType == 'EM':
            # Handled entirely by evidence field of LP dict
            # which  is used in the allocation model.
            return 0
        else:
            if todict:
                return dict(Ldata=self.calcELBO_Memoized(SS, **kwargs))
            return self.calcELBO_Memoized(SS, **kwargs)

    def calc_local_params(self, Data, LP=None, **kwargs):
        """ Calculate local 'likelihood' params for each data item.

        Returns
        -------
        LP : dict
            local parameters as key/value pairs, with fields
            * 'E_log_soft_ev' : 2D array, N x K
                Entry at row n, col k gives (expected value of)
                likelihood that observation n is produced by component k
        """
        if LP is None:
            LP = dict()
        LP['obsModelName'] = str(self.__class__.__name__)
        if self.inferType == 'EM':
            LP['E_log_soft_ev'] = self.calcLogSoftEvMatrix_FromEstParams(
                Data, **kwargs)
        else:
            L = self.calcLogSoftEvMatrix_FromPost(
                Data, **kwargs)
            if isinstance(L, dict):
                LP.update(L)
            else:
                LP['E_log_soft_ev'] = L
        return LP

    def get_global_suff_stats(self, Data, SS, LP, **kwargs):
        """ Compute sufficient statistics for provided local parameters.

        Returns
        ----
        SS : bnpy.suffstats.SuffStatBag
            Updated in place from provided value of SS.
        """
        SS = self.calcSummaryStats(Data, SS, LP, **kwargs)
        return SS

    def calcSummaryStats(self, Data, SS, LP, **kwargs):

        X = Data.X
        if 'resp' in LP:
            resp = LP['resp']
            K = resp.shape[1]
            # 1/2: Compute mean statistic
            S_x = dotATB(resp, X)
            # 2/2: Compute expected outer-product statistic
            S_xxT = np.zeros((K, Data.dim, Data.dim))
            sqrtResp_k = np.sqrt(resp[:, 0])
            sqrtRX_k = sqrtResp_k[:, np.newaxis] * Data.X
            S_xxT[0] = dotATA(sqrtRX_k)
            for k in range(1, K):
                np.sqrt(resp[:, k], out=sqrtResp_k)
                np.multiply(sqrtResp_k[:, np.newaxis], Data.X, out=sqrtRX_k)
                S_xxT[k] = dotATA(sqrtRX_k)
        else:
            spR = LP['spR']
            K = spR.shape[1]
            # 1/2: Compute mean statistic
            S_x = spR.T * X
            # 2/2: Compute expected outer-product statistic
            S_xxT = calcSpRXXT(X=X, spR_csr=spR)

        if SS is None:
            SS = SuffStatBag(K=K, D=Data.dim)
        # Expected mean for each state k
        SS.setField('x', S_x, dims=('K', 'D'))
        # Expected outer-product for each state k
        SS.setField('xxT', S_xxT, dims=('K', 'D', 'D'))
        # Expected count for each k
        #  Usually computed by allocmodel. But just in case...
        if not hasattr(SS, 'N'):
            if 'resp' in LP:
                SS.setField('N', LP['resp'].sum(axis=0), dims='K')
            else:
                SS.setField('N', as1D(toCArray(LP['spR'].sum(axis=0))), dims='K')
        return SS

    def calcELBO_Memoized(self, SS, returnVec=0, afterMStep=False, **kwargs):
        """ Calculate obsModel's objective using suff stats SS and Post.

        Args
        -------
        SS : bnpy SuffStatBag
        afterMStep : boolean flag
            if 1, elbo calculated assuming M-step just completed

        Returns
        -------
        obsELBO : scalar float
            Equal to E[ log p(x) + log p(phi) - log q(phi)]
        """
        elbo = np.zeros(SS.K)
        Post = self.Post
        Prior = self.Prior
        for k in range(SS.K):
            elbo[k] = c_Diff(Prior.nu,
                             self.GetCached('logdetB'),
                             Prior.m, Prior.kappa,
                             Post.nu[k],
                             self.GetCached('logdetB', k),
                             Post.m[k], Post.kappa[k],
                             )
            if not afterMStep:
                aDiff = SS.N[k] + Prior.nu - Post.nu[k]
                bDiff = SS.xxT[k] + Prior.B \
                                  + Prior.kappa * np.outer(Prior.m, Prior.m) \
                    - Post.B[k] \
                    - Post.kappa[k] * np.outer(Post.m[k], Post.m[k])
                cDiff = SS.x[k] + Prior.kappa * Prior.m \
                    - Post.kappa[k] * Post.m[k]
                dDiff = SS.N[k] + Prior.kappa - Post.kappa[k]
                elbo[k] += 0.5 * aDiff * self.GetCached('E_logdetL', k) \
                    - 0.5 * self._trace__E_L(bDiff, k) \
                    + np.inner(cDiff, self.GetCached('E_Lmu', k)) \
                    - 0.5 * dDiff * self.GetCached('E_muLmu', k)
        if returnVec:
            return elbo - (0.5 * SS.D * LOGTWOPI) * SS.N
        return elbo.sum() - 0.5 * np.sum(SS.N) * SS.D * LOGTWOPI

    def calcPostParams(self, SS):
        ''' Calc updated params (nu, B, m, kappa) for all comps given suff stats

            These params define the common-form of the exponential family
            Normal-Wishart posterior distribution over mu, diag(Lambda)

            Returns
            --------
            nu : 1D array, size K
            B : 3D array, size K x D x D, each B[k] is symmetric and pos. def.
            m : 2D array, size K x D
            kappa : 1D array, size K
        '''
        Prior = self.Prior
        nu = Prior.nu + SS.N
        kappa = Prior.kappa + SS.N
        m = (Prior.kappa * Prior.m + SS.x) / kappa[:, np.newaxis]
        Bmm = Prior.B + Prior.kappa * np.outer(Prior.m, Prior.m)
        B = SS.xxT + Bmm[np.newaxis, :]
        for k in range(B.shape[0]):
            B[k] -= kappa[k] * np.outer(m[k], m[k])
        return nu, B, m, kappa

    def updatePost(self, SS):
        ''' Update attribute Post for all comps given suff stats.

        Update uses the variational objective.

        Post Condition
        ---------
        Attributes K and Post updated in-place.
        '''
        #self.ClearCache()
        if not hasattr(self, 'Post') or self.Post.K != SS.K:
            self.Post = ParamBag(K=SS.K, D=SS.D)

        nu, B, m, kappa = self.calcPostParams(SS)
        self.Post.setField('nu', nu, dims=('K'))
        self.Post.setField('kappa', kappa, dims=('K'))
        self.Post.setField('m', m, dims=('K', 'D'))
        self.Post.setField('B', B, dims=('K', 'D', 'D'))
        self.K = SS.K

    def update_global_params(self, SS, rho=None, **kwargs):
        """ Update parameters to maximize objective given suff stats.

        Post Condition
        -------
        Either EstParams or Post attributes updated in place.
        """
        if self.inferType == 'EM':
            return self.updateEstParams_MaxLik(SS)
        elif rho is not None and rho < 1.0:
            return self.updatePost_stochastic(SS, rho)
        else:
            return self.updatePost(SS)

    def calcLogSoftEvMatrix_FromPost(self, Data, **kwargs):
        ''' Calculate expected log soft ev matrix under Post.

        Returns
        ------
        L : 2D array, size N x K
        '''
        K = self.K
        L = np.zeros((Data.nObs, K))
        for k in range(K):
            L[:, k] = - 0.5 * self.D * LOGTWOPI \
                      + 0.5 * self.GetCached('E_logdetL', k) \
                      - 0.5 * self._mahalDist_Post(Data.X, k)
        return L

    def getDatasetScale(self, SS):
        ''' Get number of observed scalars in dataset from suff stats.

        Used for normalizing the ELBO so it has reasonable range.

        Returns
        ---------
        s : scalar positive integer
        '''
        return SS.N.sum() * SS.D

    def _cholB(self, k=None):
        if k == 'all':
            retArr = np.zeros((self.K, self.D, self.D))
            for kk in range(self.K):
                retArr[kk] = self.GetCached('cholB', kk)
            return retArr
        elif k is None:
            B = self.Prior.B
        else:
            # k is one of [0, 1, ... K-1]
            B = self.Post.B[k]
        return scipy.linalg.cholesky(B, lower=True)

    def _logdetB(self, k=None):
        cholB = self.GetCached('cholB', k)
        return 2 * np.sum(np.log(np.diag(cholB)))

    def _E_logdetL(self, k=None):
        dvec = np.arange(1, self.D + 1, dtype=float)
        if k == 'all':
            dvec = dvec[:, np.newaxis]
            retVec = self.D * LOGTWO * np.ones(self.K)
            for kk in range(self.K):
                retVec[kk] -= self.GetCached('logdetB', kk)
            nuT = self.Post.nu[np.newaxis, :]
            retVec += np.sum(digamma(0.5 * (nuT + 1 - dvec)), axis=0)
            return retVec
        elif k is None:
            nu = self.Prior.nu
        else:
            nu = self.Post.nu[k]
        return self.D * LOGTWO \
            - self.GetCached('logdetB', k) \
            + np.sum(digamma(0.5 * (nu + 1 - dvec)))

    def _trace__E_L(self, Smat, k=None):
        if k is None:
            nu = self.Prior.nu
            B = self.Prior.B
        else:
            nu = self.Post.nu[k]
            B = self.Post.B[k]
        return nu * np.trace(np.linalg.solve(B, Smat))

    def _E_Lmu(self, k=None):
        if k is None:
            nu = self.Prior.nu
            B = self.Prior.B
            m = self.Prior.m
        else:
            nu = self.Post.nu[k]
            B = self.Post.B[k]
            m = self.Post.m[k]
        return nu * np.linalg.solve(B, m)

    def _E_muLmu(self, k=None):
        if k is None:
            nu = self.Prior.nu
            kappa = self.Prior.kappa
            m = self.Prior.m
            B = self.Prior.B
        else:
            nu = self.Post.nu[k]
            kappa = self.Post.kappa[k]
            m = self.Post.m[k]
            B = self.Post.B[k]
        Q = np.linalg.solve(self.GetCached('cholB', k), m.T)
        return self.D / kappa + nu * np.inner(Q, Q)

    def _mahalDist_Post(self, X, k):
        ''' Calc expected mahalonobis distance from comp k to each data atom

            Returns
            --------
            distvec : 1D array, size nObs
                   distvec[n] gives E[ (x-\mu) \Lam (x-\mu) ] for comp k
        '''
        Q = np.linalg.solve(self.GetCached('cholB', k),
                            (X - self.Post.m[k]).T)
        Q *= Q
        return self.Post.nu[k] * np.sum(Q, axis=0) \
               + self.D / self.Post.kappa[k]


