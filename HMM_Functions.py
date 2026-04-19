#==============================================================================
# Extra routines
#
#------------------------------------------------------------------------------
# Contains:
#
#   GetStartProbs()
#   GetTransitionMatrix()
#   GetEmissionProbs()
#
#------------------------------------------------------------------------------
# Notes:
#
#==============================================================================


# Modules
import numpy as np




#------------------------------------------------------------------------------
# Get start probabilities
#------------------------------------------------------------------------------
def GetStartProbs(num_states, states):
    
    n = states.shape[0]
    p = np.zeros((num_states), dtype=float)
    for i in range(n):
        p[states[i]] += 1.0
    # end for
    p /= n
    return p

# end def




#------------------------------------------------------------------------------
# Get transition matrix
#------------------------------------------------------------------------------
def GetTransitionMatrix(num_states, states, adjust_zero_states=False):
    
    n = states.shape[0]
    tm = np.zeros((num_states, num_states), dtype=float)
    for i in range(1, n):
        tm[states[i-1],states[i]] += 1.0
    # end for
    if adjust_zero_states:
        tm[tm < 0.5] = 0.5
    for i in range(num_states):
        tm[i,:] /= tm[i,:].sum()
    # end for
    return tm

# end def




#------------------------------------------------------------------------------
# Get emission probabilities
#------------------------------------------------------------------------------
def GetEmissionProbs(num_states, states, num_obs, obs):
    
    n = states.shape[0]
    em = np.zeros((num_states, num_obs), dtype=float)
    for i in range(n):
        em[states[i],obs[i]] += 1.0
    # end for
    for i in range(num_states):
        em[i,:] /= em[i,:].sum()
    # end for
    return em

# end def

