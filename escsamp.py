# Functions for sampling from ESC models.
import numpy as np
import scipy.special as spsp

import bell
#import logfac
from logfac import logfacTable

class ESCsampler():
    '''
    Object for generating samples from an ESC process with cluster size
    distribution mu and total number of elements n.

    That process proceeds by drawing X_1,X_2,... iid according to
    a distribution on the positive integers
    (specified here by mufn; see below).
    Conditional on the event that
    there exists a k such that \sum_{i=1}^k X_i = n,
    we return a partition whose sizes are (up to ordering)
    (X_1,X_2,...,X_k).
    '''

    def __init__(self, n, logmufn, logufn=None ):
        '''
        Constructor.

        n is a positive integer.

        logmufn is a callable that returns the log of a distribution
		on the positive integers.
            for n=1,2,3,..., mufn(n) is the n-th element in the sequence
            that is, the (marginal) probability that a cluster has size n.
            mufn should specify a probability, in that
            sum_{n=1}^infty mufn(n) = 1.
            This should return log mufn(n).

        logufn is a callable that returns log( u(k) )  for all positive k,
            log( u(0) ) = log( 0 ) = -np.inf
            for n=1,2,3,..., ufn(n) returns the n-th element in the sequence
            u_n = Pr[ E_n ], where E_n is the event that there exists 
            an integer k such that X_1+X_2+...+X_k = n, with X_i drawn iid mu.
            u_0 = 1 (because t=0 is always a renewal).

        '''

        if logufn is not None: 
            self.set_logufn( logufn ) 
        else:
            self.null_logufn( ) 

        self.n = int(n)
        self.reset_memoization() # Reset all the memoization.
        self.set_logmufn( logmufn )

    def reset_memoization( self ):
        self.logmufn_memo = dict()
        self.logufn_memo = dict()
        self.Xprob_memo = dict()

    def reset_n( self, n ):
        '''
        Change the current choice of n for the generator.
        Update the probability generation functions accordingly.
        '''
        n = int(n)
        if n < 1:
            raise ValueError('n should be a positive integer')
        self.n = n

        # As soon as n gets updated, we have to update the truncated dist
        # Signal this by setting it to None.
        self.null_truncated()
        self.null_museq()
        self.null_useq()

    def get_n( self ):
        '''
        Retrieve the current choice of n for the generator.
        '''
        return self.n

    def null_logmufn( self ):
        self.logmufn = None
        self.logmufn_memo = dict() # still need this, even if logmufn is null
    def null_logufn( self ):
        self.logufn = None
        self.logufn_memo = dict() # still need this, even if logufn is null

    def set_logmufn( self, logmufn ):
        if not callable(logmufn):
            raise TypeError('logmufn must be callable.')
        self.logmufn = logmufn 

    def set_logufn( self, logufn ):
        if not callable(logufn):
            raise TypeError('logufn must be callable.')
        self.logufn = logufn 

    def eval_logmufn( self, k ):
        '''
        eval log mu_k.
        '''
        k = int(k)
        if k<0:
            raise ValueError('mu sequence is only defined for k>=0. Cannot compute log mu_k.')

        if k not in self.logmufn_memo:
            if self.logmufn is not None:
                self.logmufn_memo[k] = self.logmufn(k)
            else: # We don't have logmu, compute it directly.
                if k==0: # mu_0=0 by definition, so logmu_0 = -inf.
                    self.logmufn_memo[k] = -np.inf
                else:
                    # We need to be careful of div-by-zero in the log.
                    # We could do this slightly more gracefully by setting up
                    # a context or using a numpy where statement, but...
                    if self.eval_mufn(k) < 1e-323: #1e-323 determined by exp.
                        self.logmufn_memo[k] = -np.inf
                    else:
                        self.logmufn_memo[k] = np.log( self.eval_mufn(k) )
        return self.logmufn_memo[k]

    def eval_logufn( self, k ):
        '''
        eval log u_k.
        '''
        k = int(k)
        if k<0:
            raise ValueError('u-sequence is only defined for k>=0. Cannot compute log u_k.')

        if k not in self.logufn_memo:
            if self.logufn is None:
                self.construct_logufn( )
            self.logufn_memo[k] = self.logufn(k)
        return self.logufn_memo[k]

    def construct_logufn( self ):
        '''
        Build and populate the u-sequence using the mu-sequence.
        '''

        logxseq = lambda k : spsp.loggamma(k+1) + self.eval_logmufn(k)
        b = bell.Bell( logxseq )

        n = self.get_n()
        loguseq_vec = b.compute_loguseq( n )
        # useq_vec has n+1 entries, useq[0]=1.
        def loguseq_fn( k ):
            k = int(k)
            if k < 0:
                raise ValueError('k must be non-negative.')
            if k > n:
                raise ValueError('k out of range for useq.')
            return loguseq_vec[k]
        self.set_logufn( loguseq_fn )
        self.bell = b # For use later in calls to kprob.

    def precompute_logmuseq( self ):
        '''
        Use self.logmufn or self.mufn
		 to compute logmu(k) for k=0,1,2,...,self.n,
        which we can then use in our computations elsewhere
	(namely, in compute_Xprob).
        '''
        self.logmuseq = -np.inf * np.ones( self.get_n()+1 )
        #self.logmuseq[0] = -np.inf, because mu_0 = 0.
        for k in range( 1, self.get_n()+1 ):
            self.logmuseq[k] = self.eval_logmufn(k)

    def precompute_loguseq( self ):
        '''
        Use self.logufn to compute u(k) for k=0,1,2,...,self.n,
        which we can then use in our computations elsewhere
	(namely, in compute_Xprob).
        '''
        self.loguseq = np.zeros( self.get_n()+1 )
        self.loguseq[0] = np.log(1.0) # u_0 = 1.
        for k in range( 1, self.get_n()+1 ):
            self.loguseq[k] = self.eval_logufn(k)

    def precompute_loguseq_new( self ):
        '''
        Use self.logufn to compute u(k) for k=0,1,2,...,self.n,
        which we can then use in our computations elsewhere
	(namely, in compute_Xprob).
        '''
        kseq = np.arange( self.get_n()+1 )
        self.loguseq = self.eval_logufn( kseq )

    def get_logmuseq( self, m ):
        '''
        Retrieve the array (logmu_0,logmu_1,...,logmu_m), return as numpy array.
        '''
        if self.logmuseq is None:
            self.precompute_logmuseq()
        return self.logmuseq[0:m+1] 
    def get_loguseq( self, m ):
        '''
        Retrieve the array (log u_0,log u_1,...,log u_m), return as numpy array.
        '''
        if self.loguseq is None:
            self.precompute_loguseq()
        return self.loguseq[0:m+1] 

    def precompute( self ):
        '''
        Just bite the bullet and do all the computation to get Xprob_memo[m]
        for all m=1,2,...,self.n.
        '''

        # First compute mu(k) for k=0,1,2,...,n and store in a big vector.
        #print('Precomputing museq...')
        #self.precompute_museq()

        # Compute logmu(k) for k=0,1,2,...,n and store in a big vector.
        print('Precomputing logmuseq...')
        self.precompute_logmuseq()

        # Compute u(k) for k=0,1,2,...,n.
        #print('Precomputing useq...')
        #self.precompute_useq()

        # Compute log u(k) for k=0,1,2,...,n.
        print('Precomputing loguseq...')
        self.precompute_loguseq()

        print('Precomputing Xprob...')
        for m in range( 1, self.get_n()+1 ):
            self.compute_Xprob(m)

        print('Finished with precomputing.')

    def compute_Xprob2( self, m ):
        '''
        Compute a distribution over 1,2,...,m with
        p(s) = mu_s u_{m-s} / u_m.

        For our sanity, include the s=0 entry,
        so the returned object is length m+1.
        '''
        m = int(m)
        if m < 1:
            raise ValueError('m should be a positive integer')

        if m not in self.Xprob_memo:
            mu = self.get_museq(m)
            # We also need u(k) for k=0,1,2,...,m.
            u = self.get_useq(m)
            prod = mu*(u[::-1]) #... but we need u in reverse.
            # return renormalized so it's a probability distribution.
            self.Xprob_memo[m] = prod/np.sum(prod)
        return self.Xprob_memo[m]

    def compute_Xprob( self, m ):
        '''
        Compute a distribution over 1,2,...,m with
        p(s) = mu_s u_{m-s} / u_m.

        For our sanity, include the s=0 entry,
        so the returned object is length m+1.

        When m is big, normalization by u_m is off a bit.
        To avoid this, so everything in log space, THEN renorm.
        '''
        m = int(m)
        if m < 1:
            raise ValueError('m should be a positive integer')

        if m not in self.Xprob_memo:
            if m==1:
                self.Xprob_memo[m] = np.array( [0.0, 1.0] )
            else:
                logmu = self.get_logmuseq(m)
                # We also need logu(k) for k=0,1,2,...,m.
                logu = self.get_loguseq(m)
                # ultimately we want probabilities of the form
                # prod = mu*(u[::-1])
                # NOTE: we need logu in reverse, hence ::-1.
                logprob = logmu + logu[::-1] - logu[m]
                probs = np.exp( logprob )
                probs[0] = 0.0 # Just to be sure.
                self.Xprob_memo[m] = probs/np.sum(probs) # Even more sure?
        return self.Xprob_memo[m]

    def compute_Xprob_try1( self, m ):
        '''
        Compute a distribution over 1,2,...,m with
        p(s) = mu_s u_{m-s} / u_m.

        For our sanity, include the s=0 entry,
        so the returned object is length m+1.

        When m is big, normalization by u_m is off a bit.
        To avoid this, so everything in log space, THEN renorm.
        '''
        m = int(m)
        if m < 1:
            raise ValueError('m should be a positive integer')

        if m not in self.Xprob_memo:
            if m==1:
                self.Xprob_memo[m] = np.array( [0.0, 1.0] )
            else:
                logmu = self.get_logmuseq(m)
                # We also need logu(k) for k=0,1,2,...,m.
                logu = self.get_loguseq(m)
                # ultimately we want probabilities of the form
                # prod = mu*(u[::-1])
                # NOTE: we need logu in reverse, hence ::-1.
                logprod = logmu + logu[::-1]
                # self.Xprob_memo[m] = prod/np.sum(prod)
                # avoiding over/underflow in that sum requires logaddexp.
                # Recall that logaddexp(A,B) returns log(expA + expB)
                lognormalize = np.logaddexp.reduce( logprod )
                # return renormalized so it's a probability distribution.
                probs = np.exp( logprod - lognormalize )
                probs[0] = 0.0 # Just to be sure.
                self.Xprob_memo[m] = probs
        return self.Xprob_memo[m]

    def generate( self ):
        '''
        Generate a sample from the ESC on n elements using our more clever
        (hopefully) approach.

        Returns a sequence summing to self.n with random length.
        '''
        m = self.get_n()
        k = 1 # Counts how many parts we have generated so far.
        parts = list() # Initially empty, but we'll put X_1,X_2,... here.
        while m > 0:
            prseq = self.compute_Xprob( m )
            # Reminder: prseq[0] is zero because no zero waiting times.
            Xk = np.random.choice( m+1, p=prseq )
            parts.append( Xk )
            m = m-Xk
        return parts 

    def draw_waiting_time( self ):
        '''
        Draw a waiting time from the mu distribution.

        Since waiting times longer than self.n are essentially ignorable,
        we instead just glom the whole $X > self.n$ event into the event
        X = self.n+1.
        '''

        # prseq includes an entry for 0, with prseq[0]=0,
        # and an entry for n+1, with the "rest" of the prob.
        # So the total set of "possible" outcomes in n+2...
        prseq = self.get_truncated()
        assert( len(prseq)==self.get_n()+2 )
        # And so we need to draw from [0,1,2,...,n,n+1] = arange(n+2)
        Xk = np.random.choice( self.get_n()+2, p=prseq )
        return int(Xk)

    def null_truncated( self ):
        self.truncated = None

    def set_truncated( self, trunc ):
        self.truncated = np.array(trunc)
         
    def get_truncated( self ):
        if self.truncated is None:
            self.construct_truncated()

        return self.truncated
        
    def construct_truncated( self ):
        '''
        Construct a probability distribution on [0,1,2,...,self.n,self.n+1]
        with
        Pr[0]=0
        Pr[k] = mu_k for k=1,2,...,self.n
        and
        Pr[self.n+1] = sum_{k=self.n+1}^infty mu_k.
        '''

        trunc = np.zeros( self.n+2 )
        trunc[0] = 0.0 # waiting time 0 has no probability.
        for k in range(1,self.n+1):
            trunc[k] = self.eval_mufn(k)
        # Last entry gets the rest of the probability mass.
        trunc[self.n+1] = np.max( [1.0-np.sum( trunc ), 0.0] )

        self.set_truncated( trunc ) #Store our new truncated distribution.

    def generate_naive( self ):
        '''
        Generate a sample from the ESC on n elements using the naive
        approach of repeatedly generating sequences and hoping they sum to n.

        Returns a sequence summing to self.n with random length.
        '''
        parts = list()
        cusum = 0 # Avoid recomputing the sum of the parts every step.

        while sum( parts ) < self.n:
            X = self.draw_waiting_time()
            parts.append(X)
            cusum += X
            if cusum == self.n:
                # We successfully generated a partition of n.
                return parts
            if cusum > self.n:
                # We over-shot n, and have to start over and try again.
                parts = list()
                cusum = 0

    def generate_naive_exact( self, gen ):
        '''
        Generate a sample from the ESC on n elements using the naive
        approach of repeatedly generating sequences and hoping they sum to n.

        Returns a sequence summing to self.n with random length.

        gen is a callable that returns a single RV.
        '''
        parts = list()
        cusum = 0 # Avoid recomputing the sum of the parts every step.

        while sum( parts ) < self.n:
            X = gen()
            parts.append(X)
            cusum += X
            if cusum == self.n:
                # We successfully generated a partition of n.
                return parts
            if cusum > self.n:
                # We over-shot n, and have to start over and try again.
                parts = list()
                cusum = 0

    def Kprob( self ):
        if self.bell is not None:
            return self.bell.compute_Kprob()
        else:
            raise RuntimeError('Bell never got set up.')
