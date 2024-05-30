import numpy as np
import scipy.special as spsp

# Tools for fast computation of log factorials.

class logfacFactory():
    '''
    Object for computing log factorials quickly.

    Specifically, log n! and log (k!/n!) for k <= n.
    '''

    def __init__(self):
        self.logfac_memo = dict()
        self.logratio_memo = dict()
        self.logbinom_memo = dict()

    def eval( self, n ):
        # Evaluate log n!.
        n = int(n)
        if n < 0:
            raise ValueError('n should be a non-negative integer.')

        if n > 4500: # Guard against recursion depth?
            self.eval( n//2 )

        if n==0 or n==1:
            return 0.0 # log 0! = log 1! = log 1 = 0

        if n not in self.logfac_memo:
            # log n! = log n + log (n-1)!
            self.logfac_memo[n] = np.log(n) + self.eval( n-1 )

        return self.logfac_memo[n]

    def eval_ratio( self, k, n ):
        # Evaluate \log k!/n! = log k! - log n!
        (k,n) = (int(k), int(n))
        if k < 0:
            raise ValueError('k should be a non-neg integer.')
        if n < 0:
            raise ValueError('n should be a non-neg integer.')

        if (k,n) not in self.logbinom_memo:
            self.logratio_memo[ (k,n) ] = self.eval(k) - self.eval(n)

        return self.logratio_memo[ (k,n) ]

    def eval_binom( self, n, k ):
        # Evaluate log \binom{n}{k} = log n! - log k! - log(n-k)!
        (n,k) = (int(n), int(k))
        if (n,k) not in self.logbinom_memo:
            if k < 0:
                raise ValueError('k should be a non-neg integer.')
            if n < 0:
                raise ValueError('n should be a non-neg integer.')

            #logbinom = self.eval(n) - self.eval(k) -self.eval(n-k)
            logbinom = -self.eval(n-k) - self.eval_ratio(k,n)
            self.logbinom_memo[ (n,k) ] = logbinom
        return self.logbinom_memo[ (n,k) ]

def logfac( n ):
    return spsp.loggamma( n+1 )

def logratio( k, n ):
    return spsp.loggamma( k+1) - spsp.loggamma( n+1 )

def logbinom( n, k ):
    '''
    Compute log of n choose k, defined by
    binom{ r }{ m } = frac{ (r)_{m} }{ m ! },
    where $(r)_m$ denotes the falling factorial,
    $(r)_m = r(r-1)(r-2)\cdots(r-m+1)$.

    Unfortunately, we can't compute log of Pochhammer in scipy, so we have
    to do things in the clumsier way. 

    Ideally, we would just use
    spsp.loggamma( n+1 ) - spsp.loggamma(k+1) - spsp.loggamma(n-k+1)
    but this can fail if any of n+1, k+1 or n-k+1 is negative.
    '''
    k = int(k) # Really would prefer to error check, but it's annoying.

    # Let's handle some of the dumb cases first.
    if k < 0:
        return -np.inf # n choose k is 0 when k < 0.
    if k==0:
        return 0.0 # n choose 0 is 1, log 1 = 0.
    if n==0:
        return -np.inf # 0 choose k is 0 for k!=0.
    if n > 0: # The "straight-forward" case.
        '''
        This can run into trouble if any of n+1, k+1 or n-k+1 are negative ints
        But n>0 handles n+1, k>0 handles k+1, and
        n-k+1 is handled by this if-statement.
        '''
        if k > n:
            return -np.inf # n choose k for k>n>0 is 0.
        else:
            lb = spsp.loggamma(n+1) - spsp.loggamma(k+1) - spsp.loggamma(n-k+1)
            return lb
    '''
    Now, here's the annoying case. If n < 0, then we have to use the identity
    binom{t}{m} = (-1)^m binom{m-t-1}{m}
	= (-1)^m (m-(t+1))! / m! (-(t+1))!
    But then if m is odd, the log is ill-defined, except if we recurse again...

    Let's punt on this.
    '''
    if n < 0:
        raise ValueError('We are cowards and are not handling n choose k for n<0.')
    raise RuntimeError('Did I miss a case in logbinom?')

def OLDlogbinom( n, k ):
    if k < 0:
        return -np.inf # n choose k is 0 when k < 0.
    if k==0:
        return 0.0 # n choose 0 is 1, log 1 = 0.
    if n==0:
        return -np.inf # 0 choose k is 0 for k!=0.
    if n<0:
        #return ((-1)**k) * logbinom( n+k-1, k)
        #return ((-1)**k)*(spsp.loggamma( n+k )-spsp.loggamma(k+1)-spsp.loggamma(n+k-1 -k +1))
        m=0
        while n+k-1 < 0: # Loop terminates eventually because n>0, k>=0.
            n = (n+k-1)
            m += 1
        # Write this out instead of relying on recursion, though I'm worried
        # that this is still going to result in some gamma( negative int )...
        lb = spsp.loggamma( n+k )-spsp.loggamma(k+1)-spsp.loggamma(n)
        return ((-1)**(m*k))*lb
    else:
        return spsp.loggamma( n+1 ) - spsp.loggamma(k+1) - spsp.loggamma(n-k+1)

class logfacTable:
    '''
    Class for fast lookup of log-factorial computations.
    '''

    def __init__( self, n ):
        '''
        Precompute all the values we need for 0 <= k <= n.
        '''

        self.n = n
        self.memoize_logfac()
        self.memoize_logratio()
        self.memoize_logbinom()

    def memoize_logfac( self ):
        self.logfac = np.ones( self.n+1 )
        for k in range(self.n+1):
            self.logfac[k] = logfac( k )

    def memoize_logratio( self ):
        self.logratio = -np.inf * np.ones( (self.n+1, self.n+1) )
        for k in range(self.n+1):
            for m in range(k,self.n+1):
                logmf = self.logfac[m]
                logkf = self.logfac[k]
                self.logratio[k,m] = logkf - logmf

    def memoize_logbinom( self ):
        self.logbinom = -np.inf * np.ones( (self.n+1, self.n+1) )
        for k in range(self.n+1):
            for m in range(k,self.n+1):
                logmf = self.logfac[m]
                logkf = self.logfac[k]
                logmkf = self.logfac[m-k]
                self.logbinom[m,k] = logmf - logkf - logmkf
