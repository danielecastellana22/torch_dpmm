import numpy as np

class ParamBag(object):

    def __init__(self, K=0, doCollapseK1=False, **kwargs):
        ''' Create a ParamBag object with specified number of components.

        Args
        --------
        K : integer number of components this bag will contain
        D : integer dimension of parameters this bag will contain
        '''
        self.K = K
        self.D = 0
        for key, val in kwargs.items():
            setattr(self, key, val)
        self._FieldDims = dict()
        self.doCollapseK1 = doCollapseK1

    def setField(self, key, rawArray, dims=None):
        ''' Set a named field to particular array value.

        Raises
        ------
        ValueError
            if provided rawArray cannot be parsed into
            shape expected by the provided dimensions tuple
        '''
        # Parse dims tuple
        if dims is None and key in self._FieldDims:
            dims = self._FieldDims[key]
        else:
            self._FieldDims[key] = dims
        # Parse value as numpy array
        setattr(self, key, rawArray) #setattr(self, key, self.parseArr(rawArray, dims=dims, key=key))

    def __add__(self, PB):
        ''' Add. Returns new ParamBag, with fields equal to self + PB
        '''
        # TODO: Decide on what happens if PB has more fields than self
        if self.K != PB.K or self.D != PB.D:
            raise ValueError('Dimension mismatch')
        PBsum = ParamBag(K=self.K, D=self.D, doCollapseK1=self.doCollapseK1)
        for key in self._FieldDims:
            arrA = getattr(self, key)
            arrB = getattr(PB, key)
            PBsum.setField(key, arrA + arrB, dims=self._FieldDims[key])
        return PBsum

    def __iadd__(self, PB):
        ''' In-place add. Updates self, with fields equal to self + PB.
        '''
        if self.K != PB.K or self.D != PB.D:
            raise ValueError('Dimension mismatch')
        if len(list(self._FieldDims.keys())) < len(list(PB._FieldDims.keys())):
            for key in PB._FieldDims:
                arrB = getattr(PB, key)
                try:
                    arrA = getattr(self, key)
                    self.setField(key, arrA + arrB)
                except AttributeError:
                    self.setField(key, arrB.copy(), dims=PB._FieldDims[key])
        else:
            for key in self._FieldDims:
                arrA = getattr(self, key)
                arrB = getattr(PB, key)
                self.setField(key, arrA + arrB)

        return self

    def subtractSpecificComps(self, PB, compIDs):
        ''' Subtract (in-place) from self the entire bag PB
                self.Fields[compIDs] -= PB
        '''
        assert len(compIDs) == PB.K
        for key in self._FieldDims:
            arr = getattr(self, key)
            if arr.ndim > 0:
                arr[compIDs] -= getattr(PB, key)
            else:
                self.setField(key, arr - getattr(PB, key), dims=None)

    def __sub__(self, PB):
        ''' Subtract.

        Returns new ParamBag object with fields equal to self - PB.
        '''
        if self.K != PB.K or self.D != PB.D:
            raise ValueError('Dimension mismatch')
        PBdiff = ParamBag(K=self.K, D=self.D, doCollapseK1=self.doCollapseK1)
        for key in self._FieldDims:
            arrA = getattr(self, key)
            arrB = getattr(PB, key)
            PBdiff.setField(key, arrA - arrB, dims=self._FieldDims[key])
        return PBdiff

    def __isub__(self, PB):
        ''' In-place subtract. Updates self, with fields equal to self - PB.
        '''
        if self.K != PB.K or self.D != PB.D:
            raise ValueError('Dimension mismatch')
        for key in self._FieldDims:
            arrA = getattr(self, key)
            arrB = getattr(PB, key)
            self.setField(key, arrA - arrB)
        return self

class SuffStatBag(object):


    def __init__(self, K=0, uids=None, **kwargs):
        '''

        Post Condition
        ---------------
        Creates an empty SuffStatBag object,
        with valid values of uids and K.
        '''
        self._Fields = ParamBag(K=K, **kwargs)


    def setField(self, key, value, dims=None):
        ''' Set named field to provided array-like value.

        Thin wrapper around ParamBag's setField method.
        '''
        self._Fields.setField(key, value, dims=dims)

    def hasELBOTerms(self):
        return hasattr(self, '_ELBOTerms')

    def hasELBOTerm(self, key):
        if not hasattr(self, '_ELBOTerms'):
            return False
        return hasattr(self._ELBOTerms, key)

    def hasAmpFactor(self):
        return hasattr(self, 'ampF')

    def getCountVec(self):
        ''' Return vector of counts for each active topic/component
        '''
        if 'N' in self._Fields._FieldDims:
            if self._Fields._FieldDims['N'] == ('K','K'):
                return self.N.sum(axis=0) # relational models
            else:
                return self.N
        elif 'SumWordCounts' in self._Fields._FieldDims:
            return self.SumWordCounts
        raise ValueError('Counts not available')

    def __add__(self, PB):
        if self.K != PB.K or self.D != PB.D:
            raise ValueError('Dimension mismatch')
        if not np.allclose(self.uids, PB.uids):
            raise ValueError('Cannot combine stats for differing uids.')
        SSsum = SuffStatBag(K=self.K, D=self.D, uids=self.uids)
        SSsum._Fields = self._Fields + PB._Fields
        if hasattr(self, '_ELBOTerms') and hasattr(PB, '_ELBOTerms'):
            SSsum._ELBOTerms = self._ELBOTerms + PB._ELBOTerms
        elif hasattr(PB, '_ELBOTerms'):
            SSsum._ELBOTerms = PB._ELBOTerms.copy()
        if hasattr(self, '_MergeTerms') and hasattr(PB, '_MergeTerms'):
            SSsum._MergeTerms = self._MergeTerms + PB._MergeTerms
        elif hasattr(PB, '_MergeTerms'):
            SSsum._MergeTerms = PB._MergeTerms.copy()
        if hasattr(self, '_SelectTerms') and hasattr(PB, '_SelectTerms'):
            SSsum._SelectTerms = self._SelectTerms + PB._SelectTerms
        if not hasattr(self, 'mUIDPairs') and hasattr(PB, 'mUIDPairs'):
            self.setMergeUIDPairs(PB.mUIDPairs)
        return SSsum

    def __iadd__(self, PB):
        if self.K != PB.K or self.D != PB.D:
            raise ValueError('Dimension mismatch')
        if not np.allclose(self.uids, PB.uids):
            raise ValueError('Cannot combine stats for differing uids.')
        self._Fields += PB._Fields
        if hasattr(self, '_ELBOTerms') and hasattr(PB, '_ELBOTerms'):
            self._ELBOTerms += PB._ELBOTerms
        elif hasattr(PB, '_ELBOTerms'):
            self._ELBOTerms = PB._ELBOTerms.copy()
        if hasattr(self, '_MergeTerms') and hasattr(PB, '_MergeTerms'):
            self._MergeTerms += PB._MergeTerms
        elif hasattr(PB, '_MergeTerms'):
            self._MergeTerms = PB._MergeTerms.copy()
        if hasattr(self, '_SelectTerms') and hasattr(PB, '_SelectTerms'):
            self._SelectTerms += PB._SelectTerms
        if not hasattr(self, 'mUIDPairs') and hasattr(PB, 'mUIDPairs'):
            self.setMergeUIDPairs(PB.mUIDPairs)
        return self

    def __sub__(self, PB):
        if self.K != PB.K or self.D != PB.D:
            raise ValueError('Dimension mismatch')
        if not np.allclose(self.uids, PB.uids):
            raise ValueError('Cannot combine stats for differing uids.')
        SSsum = SuffStatBag(K=self.K, D=self.D, uids=self.uids)
        SSsum._Fields = self._Fields - PB._Fields
        if hasattr(self, '_ELBOTerms') and hasattr(PB, '_ELBOTerms'):
            SSsum._ELBOTerms = self._ELBOTerms - PB._ELBOTerms
        if hasattr(self, '_MergeTerms') and hasattr(PB, '_MergeTerms'):
            SSsum._MergeTerms = self._MergeTerms - PB._MergeTerms
        return SSsum

    def __isub__(self, PB):
        if self.K != PB.K or self.D != PB.D:
            raise ValueError('Dimension mismatch')
        if not np.allclose(self.uids, PB.uids):
            raise ValueError('Cannot combine stats for differing uids.')
        self._Fields -= PB._Fields
        if hasattr(self, '_ELBOTerms') and hasattr(PB, '_ELBOTerms'):
            self._ELBOTerms -= PB._ELBOTerms
        if hasattr(self, '_MergeTerms') and hasattr(PB, '_MergeTerms'):
            self._MergeTerms -= PB._MergeTerms
        return self

    def __getattr__(self, key):
        _Fields = object.__getattribute__(self, "_Fields")
        _dict = object.__getattribute__(self, "__dict__")
        if key == "_Fields":
            return _Fields
        elif hasattr(_Fields, key):
            return getattr(_Fields, key)
        elif key == '__deepcopy__':  # workaround to allow copying
            return None
        elif key in _dict:
            return _dict[key]
        # Field named 'key' doesnt exist.
        errmsg = "'SuffStatBag' object has no attribute '%s'" % (key)
        raise AttributeError(errmsg)