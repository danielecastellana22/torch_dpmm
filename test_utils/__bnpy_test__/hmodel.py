

class Mock_HModel:

    def __init__(self, allocModel, obsModel):
        ''' Constructor assembles HModel given fully valid subcomponents
        '''
        self.allocModel = allocModel
        self.obsModel = obsModel

    def calc_evidence(self, Data=None, SS=None, LP=None,
                      scaleFactor=None, todict=False,
                      doLogElapsedTime=False, **kwargs):
        ''' Compute evidence lower bound (ELBO) objective function.
        '''
        if doLogElapsedTime:
            ElapsedTimeLogger.startEvent('global', 'ev')

        if Data is not None and LP is None and SS is None:
            LP = self.calc_local_params(Data, **kwargs)
            SS = self.get_global_suff_stats(Data, LP)
        evA = self.allocModel.calc_evidence(
            Data, SS, LP, todict=todict, **kwargs)
        evObs = self.obsModel.calc_evidence(
            Data, SS, LP, todict=todict, **kwargs)
        if scaleFactor is None:
            if hasattr(SS, 'scaleFactor'):
                scaleFactor = SS.scaleFactor
            else:
                scaleFactor = self.obsModel.getDatasetScale(SS)

        if doLogElapsedTime:
            ElapsedTimeLogger.stopEvent('global', 'ev')

        if todict:
            evA.update(evObs)
            for key in evA:
                evA[key] /= scaleFactor
            # Identify unique keys, ignoring subdivided terms
            # eg Lalloc_top_term1 and Lalloc_top_term2 are not counted,
            # since we expect they are already aggregated in term Lalloc
            ukeys = list(set([key.split('_')[0] for key in list(evA.keys())]))
            evA['Ltotal'] = sum([evA[key] for key in ukeys])
            return evA
        else:
            return (evA + evObs) / scaleFactor

    def calc_local_params(self, Data, LP=None,
            doLogElapsedTime=False, **kwargs):
        ''' Calculate local parameters specific to each data item.

            This is the E-step of the EM algorithm.
        '''
        if LP is None:
            LP = dict()
        if doLogElapsedTime:
            ElapsedTimeLogger.startEvent('local', 'obsupdate')
        # Calculate  "soft evidence" each component has for each item
        # Fills in LP['E_log_soft_ev'], N x K array
        LP = self.obsModel.calc_local_params(Data, LP, **kwargs)
        if doLogElapsedTime:
            ElapsedTimeLogger.stopEvent('local', 'obsupdate')
            ElapsedTimeLogger.startEvent('local', 'allocupdate')
        # Combine with allocModel probs of each cluster
        # Fills in LP['resp'], N x K array whose rows sum to one
        LP = self.allocModel.calc_local_params(Data, LP, **kwargs)
        if doLogElapsedTime:
            ElapsedTimeLogger.stopEvent('local', 'allocupdate')
        return LP

    def get_global_suff_stats(self, Data, LP,
            doLogElapsedTime=False,
            **kwargs):
        ''' Calculate sufficient statistics for each component.

        These stats summarize the data and local parameters
        assigned to each component.

        This is necessary prep for the Global Step update.
        '''
        if doLogElapsedTime:
            ElapsedTimeLogger.startEvent('local', 'allocsummary')
        SS = self.allocModel.get_global_suff_stats(Data, LP, **kwargs)
        if doLogElapsedTime:
            ElapsedTimeLogger.stopEvent('local', 'allocsummary')
            ElapsedTimeLogger.startEvent('local', 'obssummary')
        SS = self.obsModel.get_global_suff_stats(Data, SS, LP, **kwargs)
        if doLogElapsedTime:
            ElapsedTimeLogger.stopEvent('local', 'obssummary')
        return SS

    def update_global_params(self, SS, rho=None,
            doLogElapsedTime=False,
            **kwargs):
        ''' Update (in-place) global parameters given provided suff stats.
            This is the M-step of EM.
        '''
        if doLogElapsedTime:
            ElapsedTimeLogger.startEvent('global', 'alloc')
        self.allocModel.update_global_params(SS, rho, **kwargs)
        if doLogElapsedTime:
            ElapsedTimeLogger.stopEvent('global', 'alloc')
            ElapsedTimeLogger.startEvent('global', 'obs')
        self.obsModel.update_global_params(SS, rho, **kwargs)
        if doLogElapsedTime:
            ElapsedTimeLogger.stopEvent('global', 'obs')