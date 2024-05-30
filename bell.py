# Functions for working with Bell polynomials.

import numpy as np
import logfac
from logfac import logfacTable
import scipy.special as spsp

import warnings
warnings.filterwarnings("error")

class Bell():
    '''
    Class for constructing and computing Bell polynomials,
    evaluated on a given sequence xseq.
    '''

    def __init__(self, logxseq ):
        '''
        logxseq : callable, encoding log x_n for n=1,2,...
		Sequence x_n assumed to satisfy x_k=0 for k <= 0.

        Ultimately, this object is meant to encode
        the Bell polynomials B_{m,k}, evaluated on the given sequence xseq,
        for all 1 <= k <= m.
        '''

        self.set_logxseq( logxseq )
        self.logB = None # Will be np.array with logB[m,k] =log B_{m,k}.
        self.logx_memo = None # Will be used later for the logx seq.
        self.n = None # value of n we are currently equipped to compute for

    def memoize( self, n ):
        self.n = n
        self.intseq = np.arange( self.n+1 ) # Need ints 0,1,2,...,self.n
        self.lft = logfac.logfacTable( n )
        self.memoize_logxseq( n )
        self.memoize_logB( n )
        # don't memoize B or xseq unless explicitly need it later

    def get_logxseq( self ):
        return self.logxseq
    def set_logxseq( self, logxseq ):
        if logxseq is None:
            self.logxseq = None
        elif not callable(logxseq):
            raise TypeError('logxseq must be callable.')
        else:
            self.logxseq = logxseq
    def memoize_logxseq( self, n ):
        logxvals = [self.eval_logxseq(k) for k in range(n+1)]
        self.logx_memo = np.array( logxvals )

    def eval_logxseq( self, n ):
        if n < 0:
            raise ValueError('n should be a nonnegative integer')
        if self.logxseq is not None:
            return self.logxseq(n)
        else:
            raise RuntimeError('No logxseq set?')

    def memoize_logB( self, n ):
        '''
        Do the precomputation necessary to compute the matrix
        logB[m,k] = log B_{m,k}(x) for all $0 \le k \le m \le n.
        So we technically only care about the lower-diagonal half of
        this matrix, but whatever.
        If row- vs col-access comes to matter, it might make sense to make
        it symmetric with logB[i,j] = \log B_{maxij, minij} (x).
        '''
        self.logB = -np.inf * np.ones( (n+1,n+1) ) # index from (0,0) to (n,n).

        # First, handle the base cases.
        # See recurrence relations defined in wiki page
        # for the m=0 or k=0 but m neq k base cases.
	# B_{n,1} = x_n is on Comtet pg 135ish.
        self.logB[0,0] = 0.0 # log 1 = 0
        self.logB[1:,0] = -np.inf # B_{m,0} = 0 for m>0.
        self.logB[1:,1] = self.logx_memo[1:]
        self.logB[0,1:] = -np.inf

        # Now do the recursive steps.
        # B_{m,k}(x1,x2,...,x_{m-k+1})
        # = \sum_{i=1}^{m-k+1} \binom{m-1}{i-1} xi
        #                       B_{m-i,k-1}(x1,x2,...,x{m-i-k})
        # So, to figure out the m,k entry of logB,
        # we need to figure out B_{ell, k-1} for ell=k-1,k,k+1,...,m-1.
        # That is, we need to have computed the previous column.

        '''
        For each column k, we'll build out a big matrix such that reducing
        along the columns gets us a vector of $B_{m,k}$
        for each m=k,k+1,k+2,...,n
        The one problem is that the sum is over a different index set
        for different values of m >= k.
        So for fixed k and for m >= k, we want to zero out the entries
        of our reduction matrix for all i > m-k+1.
        Because we are reducing with logaddexp, this corresponds to
        making these entries -Infinity.
        '''
        # Every column uses the same x_i sequence, so that one's easy.
        logx = self.logx_memo
        logxmx = np.repeat( logx.reshape((n+1,1)), n+1, axis=1 )
        # We want a matrix with (m,i) entry log binom(m-1,i-1)
        # for 1 <= i <= m-k+1, k<= m
        # Ideally, all other entries should be -np.inf,
        # and that is achieved by logbinom memoization in lft.
        logbinom = -np.inf * np.ones((n+1,n+1))
        # Transpose because we need M_{i,m} = log binom{m-1}{i-1},
        # but we populate the logfac table the other way.
        logbinom[1:,1:] = self.lft.logbinom[:n,:n].T

        # Allocate this once for the whole for-loop.
        logB_mi_kminus1 = -np.inf * np.ones((n+1,n+1))
        for k in range(2,n+1):
            '''
            Construct a matrix M with n+1 columns such that reducing logaddexp
            down the m-th column corresponds to computing (in log space)
            B_{m,k}(x1,x2,...,x_{m-k+1})
            = \sum_{i=1}^{m-k+1} binom{m-1}{i-1} xi
                                  B_{m-i,k-1}(x1,x2,...,x{m-i-k})
            for m=k,k+1,...,n.

            For m < k, should be trivially 0, so that means -np.infs.
            Similarly, to account for the index values
            i > m-k+1, we need those entries to be -np.inf so that they don't
            contribute to the logaddexp reduction, either.

            So we want
            M_{i,m} = log binom{m-1}{i-1} + log xi + log B_{m-i,k-1}
                for m >= k and 1 <= i <= m-k+1
            M_{i,m} = -np.inf otherwise.

            We need three matrices:
            logbinom, the x-sequence and the (k-1)-column of logB
            '''

            '''
            logx and logbinom don't depend on k, so we grab them just once,
            before this for-loop (see above).
            '''
            
            '''
            Lastly, we need M_{i,m} = (stuff) + log B_{m-i,k-1}
            for 1 <= i <= m-k+1 and k <= m.
            Again, ideally it's -np.inf for all other entries,
            but easier to just do that all at once in the big matrix M.
            '''
            logB_mi_kminus1 = -np.inf * np.ones((n+1,n+1))
            logBrow = self.logB[:,k-1]
            # Each row is going to get a copy of this, but shifted by one.
            for m in range(k,n+1):
                # Need entries m-(m-k+1)=k-1 to m-1 of logB[:,k]
                logBsubseq = logBrow[(m-1):(k-2):-1]
                logB_mi_kminus1[1:m-k+2,m] = logBsubseq
     
            #axis=0 reduces down columns.
            self.logB[:,k] = np.logaddexp.reduce( logbinom+logxmx+logB_mi_kminus1, axis=0 )

            #for m in range(k,n+1):
            #    iseq = np.arange(1,(m-k+2))
            #    lB = np.logaddexp.reduce( self.lft.logbinom[ m-1, iseq-1 ] + self.logx_memo[iseq] + logB_m_kminus1[m-iseq] )
            #    self.logB[m,k] = lB

    def memoize_B( self, n ):
        '''
        Do the precomputation necessary to compute the matrix
        B[m,k] = B_{m,k}(x) for all $0 \le k \le m \le n.
        So we technically only care about the lower-diagonal half of
        this matrix, but whatever.
        If row- vs col-access comes to matter, it might make sense to make
        it symmetric with B[i,j] = B_{maxij, minij} (x).
        '''
        if n != self.n:
            raise RuntimeError('Memoizing B, n should be self.n')
        self.B = np.zeros( (n+1,n+1) ) # Want to index (0,0) to (n,n).
        self.memoize_xseq( n )

        # First, handle the base cases.
        # See recurrence relations defined in wiki page
        # for the m=0 or k=0 but m neq k base cases.
	# B_{n,1} = x_n is on Comtet pg 135ish.
        self.B[0,0] = 1.0 # log 1 = 0
        for m in range(1,n+1):
            self.B[m,0] = 0.0 # B_{m,0} = 0 for m>0.
            self.B[m,1] = self.x_memo[m]
        for k in range(1,n+1):
            self.B[0,k] = 0.0 # B_{0,k} = 0 for k>0
        # Now do the recursive steps.
        # B_{m,k}(x1,x2,...,x_{m-k+1})
        # = \sum_{i=1}^{m-k+1} \binom{m-1}{i-1} xi
        #                       B_{m-i,k-1}(x1,x2,...,x{m-i-k})
        # So, to figure out the m,k entry of logB,
        # we need to figure out B_{ell, k=1} for ell=k-1,k,k+1,...,m-1.
        # That is, we need to have computed the previous column.
        for k in range(1,n+1):
            for m in range(k,n+1):
                self.B[m,k] = np.exp( self.logB[m,k] )

    def eval( self, m, k ):
        '''
        Compute the Bell polynomials B_{m,k} evaluated on xseq,

        Returns
        Bell coefficient B_{m,k}(x1,x2,...,x_{n-k+1})
        where x is given by self.xseq.
        '''
        m = int(m)
        k = int(k)
        if not (0 <= k and k <=m ):
            errstr='Must have 0 <= k <= m. Got k=%d, m=%d.' % (k,m)
            raise ValueError(errstr)

        # Make sure we actually know how to return values.
        # It is also possible that we have memoized x, but only up to
        # some n < m.
        # If we have memoized up to n, then we should have n+1 elements.
        if (self.logB is None):
            if (self.n is None) or (m > self.n):
                self.memoize(m)
            else:
                self.memoize(self.n)

        return np.exp( self.logB[m,k] )

    #def logeval( self, m, k ):
    #    '''
    #    Compute the log Bell polynomials \log B_{m,k} evaluated on xseq,

    #    Returns
    #    Logarithm of Bell coefficient B_{m,k}(x1,x2,...,x_{n-k+1})
    #    where x is given by self.xseq.
    #    '''
    #    (m,k) = (int(m), int(k))

    #    return float( self.logeval_helper( m, k ) )

    #def logeval_helper( self, m, k ):
    #    if m >= self.logB.shape[0]:
    #        self.memoize_logeval(m)
    #        if m < 1 or k < 1:
    #            raise ValueError('m and k should be positive?')
    #        if not (0 <= k and k <=m ):
    #            errstr='Must have 0 <= k <= m. Got k=%d, m=%d.' % (k,m)
    #            raise ValueError(errstr)
    #        self.memoize_logeval(m)

    #    return self.logB[m,k]

    def logaddexp_helper( self, m, k ):
        '''
        Wrapper method for doing the actual computation of
        log Bmk = log \sum_{i=1}^{m-k+1} binom(m-1,i-1)x_i B_{m-i,k-1}
        using the logaddexp accumulate pattern.
        '''
        #iseq = np.arange(1,m-k+2)
        #iseq = self.intseq[1:m-k+2]
        # binom(m-1,i-1)x_i B_{m-i,k-1} = exp( log binom + log x + log B ).
        # Do everything in log space for stability.
        #logbinoms = logfac.logbinom( m-1, iseq-1)
        #logbinoms = self.lft.logbinom[ m-1, iseq-1 ]
        #logxsubseq = self.logx_memo[iseq]
        #logBseq = self.logB[m-iseq,k-1]
        # log (A+B) = log( e^{log A} + e^{ log B}), so reduce works.
        #return np.logaddexp.reduce( logbinoms+logxsubseq+logBseq )
        return np.logaddexp.reduce( self.construct_logsum(m,k) )

    def construct_logsum( self, m, k ):
        (logbinoms, logxsubseq, logBseq) = self.retrieve_logsumseqs(m,k)
        return logbinoms+logxsubseq+logBseq

    def retrieve_logsumseqs( self, m, k ):
        iseq = self.intseq[1:m-k+2]
        logbinoms = self.lft.logbinom[ m-1, iseq-1 ]
        logxsubseq = self.logx_memo[iseq]
        logBseq = self.logB[m-iseq,k-1]
        return (logbinoms, logxsubseq, logBseq)

    def compute_useq( self, n ):
        '''
        Use our x-sequence to compute the corresponding u-sequence,
        u_0,u_1,u_2,...,u_n.

        u_k = Pr[ E_k ] is the probability of a renewal at time k.
        u_0 = 1, since t=0 is always a renewal by convention.

        We are going to use the fact that we can express u_k in terms of mu_j
        terms via Faa di Bruno's formula.
        n! u_n = \sum_{k=1}^n k!
                    B_{n,k}(\mu_1, 2\mu_2, \dots, (n-k+1)! \mu_{n-k+1}).
        '''
        useq = np.ones( n+1 )
        self.memoize( n )
        for m in range(1,n+1): # start at m=1 bc u[0]=1 trivially.
            # We should also be able to speed this up using clever identities? 
            kseq = np.arange(1,m+1) # sequence k=1,2,...,m
            # We want to compute k!/m!,
            # but using binomial directly can cause overflow.
            #logratio = np.array([self.logfac.eval_ratio(k,m) for k in kseq])
            #logratio = logfac.logratio(kseq, m)
            logratio = self.lft.logratio[kseq, m]

            #logBseq = np.array([self.logeval(m,k) for k in kseq])
            #logB_mk_vec = np.vectorize( lambda k : self.logeval( m, k ) )
            #logBseq = logB_mk_vec( kseq )
            logBseq = self.logB[m,kseq]
            # Do everything in log space, then exponentiate, THEN sum B_{mk}
            useq[m] = np.sum( np.exp( logBseq + logratio ) )
        return useq

    def compute_loguseq( self, n ):
        '''
        Use our x-sequence to compute the logarithm of the
        corresponding u-sequence, u_0,u_1,u_2,...,u_n.

        u_k = Pr[ E_k ] is the probability of a renewal at time k.
        u_0 = 1, since t=0 is always a renewal by convention.

        We are going to use the fact that we can express u_k in terms of mu_j
        terms via Faa di Bruno's formula.
        n! u_n = \sum_{k=1}^n k!
                    B_{n,k}(\mu_1, 2\mu_2, \dots, (n-k+1)! \mu_{n-k+1}).
        '''
        loguseq = np.zeros( n+1 ) # analogous to initializing probs to be 1
        self.memoize( n )
        for m in range(1,n+1): # start at m=1 bc u[0]=1 trivially.
            # We should also be able to speed this up using clever identities? 
            kseq = np.arange(1,m+1) # sequence k=1,2,...,m
            # We want to compute k!/m!,
            # but using binomial directly can cause overflow.
            #logratio = np.array([self.logfac.eval_ratio(k,m) for k in kseq])
            #logratio = logfac.logratio(kseq, m)
            logratio = self.lft.logratio[kseq, m]

            #logBseq = np.array([self.logeval(m,k) for k in kseq])
            #logB_mk_vec = np.vectorize( lambda k : self.logeval( m, k ) )
            #logBseq = logB_mk_vec( kseq )
            logBseq = self.logB[m,kseq]
            # Do everything in log space, then exponentiate, THEN sum B_{mk}
            #useq[m] = np.sum( np.exp( logBseq + logratio ) )
            loguseq[m] = np.log( np.sum( np.exp( logBseq + logratio ) ) )
        return loguseq

    def compute_loguseq_new( self, n ):
        '''
        Use our x-sequence to compute the logarithm of the
        corresponding u-sequence, u_0,u_1,u_2,...,u_n.

        u_k = Pr[ E_k ] is the probability of a renewal at time k.
        u_0 = 1, since t=0 is always a renewal by convention.

        We are going to use the fact that we can express u_k in terms of mu_j
        terms via Faa di Bruno's formula.
        n! u_n = \sum_{k=1}^n k!
                    B_{n,k}(\mu_1, 2\mu_2, \dots, (n-k+1)! \mu_{n-k+1}).
        '''
        self.memoize(n)

    def compute_Kprob( self ):
        '''
        Return the (n+1)-dimensional vector whose entries are
        Pr[ K_n = k \mid E_n, bmu ] as encoded in the log-mu and/or log-u
        vector.

        We will compute the logs of those probabilities, though, first.
        '''
        kseq = np.arange( self.n+1 )
        logfacs = spsp.loggamma( kseq+1 )
        logBell = self.logB[ self.n, kseq ]
        '''
        Note that this needs to be normalized to make it a prob later.
        kprobs[0] is zero, because K_n \ge 1.
        so log kprobs[0] is -infty.
        '''
        logkprobs = logfacs + logBell - spsp.loggamma( self.n )
        logkprobs[0] = -np.inf
        # Exponentiate and renormalize
        kprobs = np.exp( logkprobs  )
        kprobs[0] = 0.0
        return kprobs/np.sum(kprobs)

class Bell_preset( Bell ):
    '''
    Class for computing Bell polynomials when we know a closed-form
    expression for the coefficients (and therefore don't need to compute
    the terms ourselves).
    '''

    def __init__( self, fn, n ):

        n = int(n) # Just to be sure n is an integer.
        if n < 1:
            raise ValueError('n must be a positive integer.')
        self.set_n( n )

        # fn should be a callable that takes two args m,k and returns
        # the coefficient B_{m,k}
        if not callable(fn):
            raise TypeError('fn needs to be callable.')
        self.B = fn

    def eval( self, m, k ):
        m = int(m)
        k = int(k)
        if not (1 <= k and k <=m ):
            errstr='Must have 1 <= k <= m. Got k=%d, m=%d.' % (k,m)
            raise ValueError(errstr)

        return self.B(m,k)

