#==============================================================================
#
# Forecast - Models
#
#------------------------------------------------------------------------------
#
# Contains:
#
#   Model_UBN  - Univariate
#                Box-Cox transform
#                Normal distribution
#
#   Model_UBAN - Univariate
#                Autoregressive AR(1)
#                Box-Cox transform
#                Normal distribution
#
#   Model_UBMN - Univariate
#                Markov chain states
#                Box-Cox transform
#                Normal distribution
#
#   Model_UMM  - Univariate
#                Markov chain states
#                Multinomial symbols
#
#   random_normal_std - 
#
#------------------------------------------------------------------------------
#
# Notes:
#
#   - Every model should provide:
#       __init__()  - Instantiate
#       print()     - Print info and parameters
#       fit()       - Fit parameters, provide diagnostics
#       sample()    - Sample from fitted parameters
#
#
#==============================================================================


# Imports
import sys
import numpy as np
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from statsmodels.tsa.stattools import acf


# Local imports
from Forecast_Common import report_error, report_warning


# Local variables
DEBUG = 0  # Debug level (0=None, 1=Some, 2=More, ...)




#------------------------------------------------------------------------------
# Univariate, Box-Cox transform, Normal distribution (UBN) stochastic model
#
#   - Univariate
#   - Box-Cox transformation (alpha, lmbda)
#   - Normal distribution (mean, stdev)
#   - Parameters:
#       pars[0] = alpha
#       pars[1] = lmbda
#       pars[2] = mean
#       pars[3] = stdev
#   - No parameter uncertainty
#
#------------------------------------------------------------------------------
class Model_UBN:

    
    #--------------------------------------------------------------------------
    # Instantiate object
    #--------------------------------------------------------------------------
    def __init__(self):

        # Basic variables
        self.name = 'Univariate, Box-Cox Transform, Normal distribution'
        self.fit_flag = False

        # Parameter variables
        self.alpha = None
        self.lmbda = None
        self.mean = None
        self.std = None
        
    # end def




    #--------------------------------------------------------------------------
    # Print variables
    #--------------------------------------------------------------------------
    def print(self):

        # Print variables
        print('\nModel variables:')
        print('  name =', self.name)
        print('  fit_flag =', self.fit_flag)
        if self.fit_flag:
            # ----------
            print('  alpha =', self.alpha)
            print('  lmbda =', self.lmbda)
            print('  mean =', self.mean)
            print('  std =', self.std)
        # end if

    # end def




    #--------------------------------------------------------------------------
    # Fit model
    #--------------------------------------------------------------------------
    def fit(self, x, alpha=0.0, lambda_range=None, diagnostics=False):

        # Fit Box-Cox transform
        self.alpha = alpha
        xt, self.lmbda = boxcox(x + self.alpha)

        # Enforce lambda range
        if not lambda_range is None:
            if self.lmbda < lambda_range[0]:
                self.lmbda = lambda_range[0]
                xt = boxcox(x + self.alpha, self.lmbda)
            elif self.lmbda > lambda_range[1]:
                self.lmbda = lambda_range[1]
                xt = boxcox(x + self.alpha, self.lmbda)
            # end if
        # end if
    
        # Fit standardisation
        self.mean = xt.mean()
        self.std = xt.std()

        # Finish residuals calculation
        xt = (xt - self.mean) / self.std
       
        # TODO: If required produce diagnostics
        if diagnostics:
            report_warning('Diagnostics not functional')
        # end if

        # Set parameters-fitted flag
        self.fit_flag = True
       
    # end def




    #--------------------------------------------------------------------------
    # Sample from model
    #--------------------------------------------------------------------------
    def sample(self, num_reps, num_years, seed=None, z_rng=None):

        # Check if model has been fitted
        if not self.fit_flag:
            report_error('Model not fitted yet')
        
        # If required set random number seed
        if not seed is None:
            np.random.seed(seed)

        # Create stochastic data
        z = random_normal_std((num_years, num_reps), z_rng)
        xt = z * self.std + self.mean
        x = inv_boxcox(xt, self.lmbda) - self.alpha

        # Return stochastic data
        return x
        
    # end def


# end class




#------------------------------------------------------------------------------
# Univariate, Box-Cox transform, AR(1), Normal distribution (UBAN) stochastic model
#
#   - Univariate
#   - Box-Cox transformation (alpha, lmbda)
#   - Normal distribution (mean, stdev)
#   - Lag-1 auto-correlation (acf1)
#   - Parameters:
#       pars[0] = alpha
#       pars[1] = lmbda
#       pars[2] = mean
#       pars[3] = stdev
#       pars[4] = acf1
#   - No parameter uncertainty
#   - Need to supply initial x.
#
#------------------------------------------------------------------------------
class Model_UBAN:

    
    #--------------------------------------------------------------------------
    # Instantiate object
    #--------------------------------------------------------------------------
    def __init__(self):

        # Basic variables
        self.name = 'Univariate, Box-Cox transfrom, AR(1), Normal distribution'
        self.fit_flag = False

        # Parameter variables
        self.alpha = None
        self.lmbda = None
        self.mean = None
        self.std = None
        self.acf1 = None
        
    # end def




    #--------------------------------------------------------------------------
    # Print variables
    #--------------------------------------------------------------------------
    def print(self):

        # Print variables
        print('\nModel variables:')
        print('  name =', self.name)
        print('  fit_flag =', self.fit_flag)
        if self.fit_flag:
            print('  alpha =', self.alpha)
            print('  lmbda =', self.lmbda)
            print('  mean =', self.mean)
            print('  std =', self.std)
            print('  acf1 =', self.acf1)
        # end if

    # end def




    #--------------------------------------------------------------------------
    # Fit model
    #--------------------------------------------------------------------------
    def fit(self, x, alpha=0.0, lambda_range=None, diagnostics=False):

        # Fit Box-Cox transform
        self.alpha = alpha
        xt, self.lmbda = boxcox(x + self.alpha)

        # Enforce lambda range
        if not lambda_range is None:
            if self.lmbda < lambda_range[0]:
                self.lmbda = lambda_range[0]
                xt = boxcox(x + self.alpha, self.lmbda)
            elif self.lmbda > lambda_range[1]:
                self.lmbda = lambda_range[1]
                xt = boxcox(x + self.alpha, self.lmbda)
            # end if
        # end if
    
        # Fit standardisation
        self.mean = xt.mean()
        self.std = xt.std()
        self.acf1 = acf(xt, nlags=1)[1]

        # Finish residuals calculation
        xt = (xt - self.mean) / self.std
        xt1 = np.zeros_like(xt)
        xt1[0] = xt[0]
        xt1[1:] = xt[1:] - self.acf1 * xt[:-1]

        # TODO: If required produce diagnostics
        if diagnostics:
            report_warning('Diagnostics not functional')
        # end if

        # Set fitted parameters flag
        self.fit_flag = True

    # end def




    #--------------------------------------------------------------------------
    # Sample from model
    #--------------------------------------------------------------------------
    def sample(self, num_reps, num_years, x0=None, seed=None, z_rng=None):

        # Check if model has been fitted
        if not self.fit_flag:
            report_error('Model not fitted yet')
        
        # If required set random number seed
        if not seed is None:
            np.random.seed(seed)

        # Transform and standardise initial value
        if x0 is None:
            xt0 = random_normal_std((1,1), z_rng)[0,0]
        else:
            xt0 = boxcox(x0 + self.alpha, self.lmbda)
            xt0 = (xt0 - self.mean) / self.std
        # end if

        # Create stochastic data
        z = random_normal_std((num_years, num_reps), z_rng)
        xt = np.zeros((num_years, num_reps), dtype=float)
        xt[0,:] = z[0,:] + self.acf1 * xt0
        for i in range(1, num_years):
            xt[i,:] = z[i,:] + self.acf1 * xt[i-1,:]
        xt = self.std * xt + self.mean
        x = inv_boxcox(xt, self.lmbda) - self.alpha

        # Return stochastic data
        return x
        
    # end def

    
# end class




#------------------------------------------------------------------------------
# Univariate, Box-Cox transform, Markov chain, Normal distribution (UBMN) stochastic model
#
#   - Univariate
#   - Box-Cox transformation (alpha, lambda)
#   -
#   - Normal distribution (mean, stdev)
#   - Parameters:
#       ns      = Number of states
#       pars[:] = Initial probabilities (ns)
#       pars[:] = Transition Matrix (ns*(ns-1))
#       pars[:] = alpha (ns)
#       pars[:] = lambda (ns)
#       pars[:] = mean (ns)
#       pars[:] = stdev (ns)
#   - No parameter uncertainty
#
# Notes
#
#   - Should you first transform then do states, or first do states then transform?
#   - 
#
#------------------------------------------------------------------------------
class Model_UBMN:

    
    #--------------------------------------------------------------------------
    # Instantiate object
    #--------------------------------------------------------------------------
    def __init__(self):

        # Basic variables
        self.name = 'Univariate, Box-Cox transform, Markov chain, Normal distribution'
        self.fit_flag = False

        # Parameter variables
        self.ns = None
        self.sprobs = None
        self.strans = None
        # ----------
        self.alpha = None
        self.lmbda = None
        self.mean = None
        self.std = None
        
    # end def




    #--------------------------------------------------------------------------
    # Print variables
    #--------------------------------------------------------------------------
    def print(self):

        # Print variables
        print('\nModel variables:')
        print('  name =', self.name)
        print('  fit_flag =', self.fit_flag)
        if self.fit_flag:
            print('  ns =', self.ns)
            print('  sprobs:', np.array2string(self.sprobs, prefix='  sprobs:'))
            print('  strans:', np.array2string(self.strans, prefix='  strans:'))
            # ----------
            print('  alpha =', self.alpha)
            print('  lmbda =', self.lmbda)
            print('  mean:', np.array2string(self.mean, prefix='  mean:'))
            print('  std:', np.array2string(self.std, prefix='  std:'))
        # end if

    # end def




    #--------------------------------------------------------------------------
    # Fit model
    #--------------------------------------------------------------------------
    def fit(self, x, ns, s, alpha=0.0, lambda_range=None, diagnostics=False,
            adjust_zero_states=False):
        
        # Define variables
        self.ns = ns
        self.sprobs = np.zeros((ns), dtype=float)
        self.strans = np.zeros((ns,ns), dtype=float)
        self.mean = np.zeros((ns), dtype=float)
        self.std = np.zeros((ns), dtype=float)

        # Create state and transition probabilites
        for i in range(s.shape[0]):
            self.sprobs[s[i]] += 1.0
            if i>0: self.strans[s[i-1],s[i]] += 1.0
        # end for
        if adjust_zero_states:
            self.strans[self.strans < 0.5] = 0.5
        self.sprobs /= self.sprobs.sum()
        for i in range(ns):
            self.strans[i,:] /= self.strans[i,:].sum()
        # end for

        # Safer to transform the entire array first, then have separate
        # means and stds for each state

        # Fit Box-Cox transform
        self.alpha = alpha
        xt, self.lmbda = boxcox(x + self.alpha)

        # Enforce lambda range
        if not lambda_range is None:
            if self.lmbda < lambda_range[0]:
                self.lmbda = lambda_range[0]
                xt = boxcox(x + self.alpha, self.lmbda)
            elif self.lmbda > lambda_range[1]:
                self.lmbda = lambda_range[1]
                xt = boxcox(x + self.alpha, self.lmbda)
            # end if
        # end if

        # Loop through states
        for i in range(ns):

            # Assemble x values for state i
            xts = xt[s==i]
        
            # Fit standardisation
            self.mean[i] = xts.mean()
            self.std[i] = xts.std()
    
            # Finish residuals calculation
            xts = (xts - self.mean[i]) / self.std[i]

        # end for

        # TODO: If required produce diagnostics
        if diagnostics:
            report_warning('Diagnostics not functional')
        # end if

        # Set fitted parameters flag
        self.fit_flag = True

    # end def




    #--------------------------------------------------------------------------
    # Sample from model
    #--------------------------------------------------------------------------
    def sample(self, num_reps, num_years, s0=None, s1=None, seed=None, z_rng=None):

        # Check if model has been fitted
        if not self.fit_flag:
            report_error('Model not fitted yet')
        
        # If required set random number seed
        if not seed is None:
            np.random.seed(seed)
        
        # Create cumulative state probability arrays
        csp = np.cumsum(self.sprobs)
        cst = np.cumsum(self.strans, axis=1)
        if DEBUG > 0:
            print('\ncsp:')
            print(csp)
            print('\ncst:')
            print(cst)
        # end if

        # Create random states
        u0 = np.random.uniform(size=(num_reps))
        u = np.random.uniform(size=(num_years, num_reps))
        s = np.zeros((num_years, num_reps), dtype=int)
        for i in range(num_reps):
            if s1 is None:
                if s0 is None:
                    s0_ = np.searchsorted(csp, u0[i])
                else:
                    s0_ = s0
                # end if
                s[0,i] = np.searchsorted(cst[s0_,:], u[0,i])
            else:
                s[0,i] = s1
            # end if
            for j in range(1, num_years):
                s[j,i] = np.searchsorted(cst[s[j-1,i],:], u[j,i])
            # end for
        # end for

        # Create random data
        z = random_normal_std((num_years, num_reps), z_rng)
        x = np.zeros((num_years, num_reps), dtype=float)
        for i in range(num_reps):
            for j in range(num_years):
                sji = s[j,i]
                xt = z[j,i] * self.std[sji] + self.mean[sji]
                x[j,i] = inv_boxcox(xt, self.lmbda) - self.alpha
            # end for
        # end for

        # Return stochastic data
        return x, s
        
    # end def

    
# end class




#------------------------------------------------------------------------------
# Univariate, Markov Chain States, Multinomial Symbols (UMM) stochastic model
#
#   - Univariate
#   - Markov chain states
#   - Multinomial symbols
#   - Parameters:
#       nstates = Number of states
#       nsymbols= Number of symbols
#       pars[:] = State initial probabilities (ns)
#       pars[:] = State transition matrix (ns*(ns-1))
#       pars[:] = Symbol emission probabilities (ns)
#   - No parameter uncertainty
#   - Includes methods to forcast samples and probabilities
#   - Optionaly adjusts transition matrix for zero occurences (uses 0.5 occurences)
#
# Notes
#
#   - 
#
#------------------------------------------------------------------------------
class Model_UMM:

    
    #--------------------------------------------------------------------------
    # Instantiate object
    #--------------------------------------------------------------------------
    def __init__(self):

        # Basic variables
        self.name = 'Univariate, Markov Chain States, Multinomial Symbols'
        self.fit_flag = False

        # Parameter variables
        self.nstates = None
        self.nsymbols = None
        self.ssprobs = None
        self.seprobs = None
        self.strans = None
        self.eprobs = None
        
    # end def




    #--------------------------------------------------------------------------
    # Print variables
    #--------------------------------------------------------------------------
    def print(self):

        # Print variables
        print('\nModel variables:')
        print('  name =', self.name)
        print('  fit_flag =', self.fit_flag)
        if self.fit_flag:
            print('  nstates =', self.nstates)
            print('  nsymbols =', self.nsymbols)
            print('  ssprobs:', np.array2string(self.ssprobs, prefix='  ssprobs:'))
            print('  seprobs:', np.array2string(self.seprobs, prefix='  seprobs:'))
            print('  strans:', np.array2string(self.strans, prefix='  strans:'))
            print('  eprobs:', np.array2string(self.eprobs, prefix='  eprobs:'))
        # end if

    # end def




    #--------------------------------------------------------------------------
    # Fit model
    #--------------------------------------------------------------------------
    def fit(self, num_states, states, num_symbols, symbols, diagnostics=False,
            adjust_zero_states=False):
        
        # Define variables
        self.nstates = num_states
        self.nsymbols = num_symbols
        self.ssprobs = np.zeros((num_states), dtype=float)
        self.seprobs = np.zeros((num_symbols), dtype=float)
        self.strans = np.zeros((num_states,num_states), dtype=float)
        self.eprobs = np.zeros((num_states,num_symbols), dtype=float)

        # Create start state probabilities
        for i in range(states.shape[0]):
            self.ssprobs[states[i]] += 1.0
        # end for
        self.ssprobs /= self.ssprobs.sum()

        # Create start symbol probabilities
        for i in range(symbols.shape[0]):
            self.seprobs[symbols[i]] += 1.0
        # end for
        self.seprobs /= self.seprobs.sum()

        # Create state transition probabilites
        for i in range(1, states.shape[0]):
            self.strans[states[i-1],states[i]] += 1.0
        # end for
        if adjust_zero_states:
            self.strans[self.strans < 0.5] = 0.5
        # end if
        for i in range(num_states):
            self.strans[i,:] /= self.strans[i,:].sum()
        # end for

        # Create symbol emission probabilites
        for i in range(states.shape[0]):
            self.eprobs[states[i],symbols[i]] += 1.0
        # end for
        for i in range(num_states):
            self.eprobs[i,:] /= self.eprobs[i,:].sum()
        # end for

        # TODO: If required produce diagnostics
        if diagnostics:
            report_warning('Diagnostics not functional')
        # end if

        # Set fitted parameters flag
        self.fit_flag = True

    # end def




    #--------------------------------------------------------------------------
    # Forecast from model by sampling
    #--------------------------------------------------------------------------
    def sample(self, num_reps, num_years, state0=None, state1=None, seed=None):

        # Check if model has been fitted
        if not self.fit_flag:
            report_error('Model not fitted yet')
        
        # If required set random number seed
        if not seed is None:
            np.random.seed(seed)
        
        # Create cumulative probability arrays
        csp = np.cumsum(self.ssprobs)
        cst = np.cumsum(self.strans, axis=1)
        cep = np.cumsum(self.eprobs, axis=1)
        if DEBUG > 0:
            print('\ncsp:')
            print(csp)
            print('\ncst:')
            print(cst)
            print('\ncep:')
            print(cep)
        # end if

        # Create states
        u0 = np.random.uniform(size=(num_reps))
        u = np.random.uniform(size=(num_years, num_reps))
        states = np.zeros((num_years, num_reps), dtype=int)
        for i in range(num_reps):
            if state1 is None:
                if state0 is None:
                    state0_ = np.searchsorted(csp, u0[i])
                else:
                    state0_ = state0
                # end if
                states[0,i] = np.searchsorted(cst[state0_,:], u[0,i])
            else:
                states[0,i] = state1
            # end if
            for j in range(1, num_years):
                states[j,i] = np.searchsorted(cst[states[j-1,i],:], u[j,i])
            # end for
        # end for

        # Create symbols
        u = np.random.uniform(size=(num_years, num_reps))
        symbols = np.zeros((num_years, num_reps), dtype=int)
        for i in range(num_reps):
            for j in range(num_years):
                symbols[j,i]  = np.searchsorted(cep[states[j,i],:], u[j,i])
            # end for
        # end for

        # Return stochastic data
        return states, symbols
        
    # end def


    #--------------------------------------------------------------------------
    # Forecast model probabilities
    #--------------------------------------------------------------------------
    def probs(self, num_years, state0=None, state1=None):

        # Check if model has been fitted
        if not self.fit_flag:
            report_error('Model not fitted yet')

        # Create state probabilities
        statep = np.zeros((num_years, self.nstates), dtype=float)
        if state1 is None:
            statep0 = np.zeros((self.nstates), dtype=float)
            if state0 is None:
                statep0 = self.ssprobs
            else:
                statep0[state0] = 1.0
            # end if
            statep[0,:] = np.matmul(statep0, self.strans)
        else:
            statep[0,state1] = 1.0
        # end for
        for j in range(1, num_years):
            statep[j,:] = np.matmul(statep[j-1,:], self.strans)
        # end for

        # Create symbol probabilities
        symbolp = np.zeros((num_years, self.nsymbols), dtype=float)
        for j in range(num_years):
            symbolp[j,:]  = np.matmul(statep[j,:], self.eprobs)
        # end for

        # Return stochastic data
        return statep, symbolp
        
    # end def

    
    #--------------------------------------------------------------------------
    # Forecast model Brier Skill Scores for some test data
    #--------------------------------------------------------------------------
    def BSS(self, num_years, test_states, test_symbols,
            ref_state_probs=None, ref_symbol_probs=None):

        # Check if model has been fitted
        if not self.fit_flag:
            report_error('Model not fitted yet')

        # --- Make Forecasts ---
        
        # Forecast probabilities for each forecast point
        test_years = test_states.shape[0]
        state_p = np.full((test_years, num_years, self.nstates), np.nan, dtype=float)
        symbol_p = np.full((test_years, num_years, self.nsymbols), np.nan, dtype=float)

        # Loop through forecast starting points
        for i in range(test_years):
        
            # Check if forecast is possible
            #   (1) No. previous data points required
            #   (2) No. forecast data points possible
            t1_prev = 1
            t1_fore = min(num_years, test_years-i)
            if i >= t1_prev and t1_fore > 0:
        
                # Calculate forecast probabilities and put into output arrays
                statep1, symbolp1 = self.probs(t1_fore, state0=test_states[i-1])
                for j in range(t1_fore):
                    state_p[i+j,j,:] = statep1[j,:]
                    symbol_p[i+j,j,:] = symbolp1[j,:]
                # end for
        
            # end for
        
        # end for

        # --- Brier Skill Score (states) ---

        # Define reference probabilities (states)
        if ref_state_probs is None:
            state_r = self.ssprobs  # Use what the model was fitted to
        else:
            state_r = ref_state_probs
        # end if
        
        # Calculate Brier Scores BS, BSref & BSS (states)
        # (A value of BSS >0.05 shows skill)
        state_BS = np.zeros((num_years), dtype=float)
        state_BSref = np.zeros((num_years), dtype=float)
        state_BSS = np.zeros((num_years), dtype=float)
        state_o = np.zeros((self.nstates), dtype=float)
        for j in range(num_years):
            for i in range(test_years):
                if not np.isnan(state_p[i,j,0]):
                    state_o[:] = 0.0; state_o[test_states[i]] = 1.0  # Observed probability
                    for k in range(self.nstates):
                        state_BS[j] += (state_p[i,j,k] - state_o[k]) ** 2.0
                        state_BSref[j] += (state_r[k] - state_o[k]) ** 2.0
                    # end for
                # end if
            # end for
            state_BSS[j] = 1.0 - state_BS[j] / state_BSref[j]
        # end for
        
        # --- Brier Skill Score (symbols) ---
        
        # Define reference probabilities (symbols)
        if ref_symbol_probs is None:
            symbol_r = self.seprobs  # Use what the model was fitted to
        else:
            symbol_r = ref_symbol_probs
        # end if
        
        # Calculate Brier Scores BS, BSref & BSS (symbols)
        # (A value of BSS >0.05 shows skill)
        symbol_BS = np.zeros((num_years), dtype=float)
        symbol_BSref = np.zeros((num_years), dtype=float)
        symbol_BSS = np.zeros((num_years), dtype=float)
        symbol_o = np.zeros((self.nsymbols), dtype=float)
        for j in range(num_years):
            for i in range(test_years):
                if not np.isnan(symbol_p[i,j,0]):
                    symbol_o[:] = 0.0; symbol_o[test_symbols[i]] = 1.0  # Observed probability
                    for k in range(self.nsymbols):
                        symbol_BS[j] += (symbol_p[i,j,k] - symbol_o[k]) ** 2.0
                        symbol_BSref[j] += (symbol_r[k] - symbol_o[k]) ** 2.0
                    # end for
                # end if
            # end for
            symbol_BSS[j] = 1.0 - symbol_BS[j] / symbol_BSref[j]
        # end for

        # --- Return Outputs ---

        # Return Brier Skill Scores
        return state_BSS, symbol_BSS

    # end def


# end class




#------------------------------------------------------------------------------
# Generate standard normal deviates of specific size (tuple)
#   Optionally limit to range ([min,max])
#------------------------------------------------------------------------------
def random_normal_std(size, z_rng=None):

    # Generate deviates
    z = np.random.normal(size=size)

    # If required apply limit to deviates
    if not z_rng is None:
        while np.min(z) < z_rng[0]:
            for i,j in zip(*np.where(z<z_rng[0])):
                z[i,j] = np.random.normal()
            # end for
        # end while
        while np.max(z) > z_rng[1]:
            for i,j in zip(*np.where(z>z_rng[1])):
                z[i,j] = np.random.normal()
            # end for
        # end while
    # end if
    
    # Return deviates
    return z

# end def

