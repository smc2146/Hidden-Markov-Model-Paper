#==============================================================================
#
# Forecast UMM model - ENSO
#
#==============================================================================


# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Local imports
import Forecast_Models as smodel
import Forecast_Plots as splot


# Local variables
DEBUG = 0  # Debug level (0=None, 1=Some, 2=More, ...)


# --- Setup Model -------------------------------------------------------------


# Print header
print('\nUMM model - ENSO')        


# Read data into dataframe
data_df = pd.read_csv(r'data\QLD_Apr_Mar_SOI_Jun_Mar_TPI_Jul_Jun.csv', comment='#')


# Get time data
t = data_df['Year'].to_numpy()
t_num = t.shape[0]


# Get state data (3-state based on ENSO)
# 0 = El Nino, 1 = Neutral, 2 = La Nina
states = np.zeros((t_num), dtype=int)
num_states = 3
q1 = -3.85
q2 = 4.5
for i in range(t_num):
    if data_df['SOI_Jun_Mar'][i] < q1:
        states[i] = 0
    elif data_df['SOI_Jun_Mar'][i] <= q2:
        states[i] = 1
    else:
        states[i] = 2
    # end if
# end for
print('\nStates:')
print('  num_states =', num_states)
print('  state q1 =', q1)
print('  state q2 =', q2)
print('  states:\n', states)


# Get observed symbol data
symbols = np.zeros((t_num), dtype=int)
num_symbols = 3
q1 = np.quantile(data_df['QLD_Apr_Mar'],0.33)
q2 = np.quantile(data_df['QLD_Apr_Mar'], 0.67)
for i in range(t_num):
    if data_df['QLD_Apr_Mar'][i] < q1:
        symbols[i] = 0
    elif data_df['QLD_Apr_Mar'][i] < q2:
        symbols[i] = 1
    else:
        symbols[i] = 2
    # end if
# end for
print('\nObserved symbols:')
print('  num_symbols =', num_symbols)
print('  symbol q1 =', q1)
print('  symbol q2 =', q2)
print('  symbols:\n', symbols)


#----------------------------------------------------------


# Print header
print('\nForecast UMM model - ENSO')        


# Create output data/plots flag
make_outputs = True


# Fit model
m1 = smodel.Model_UMM()
m1.fit(num_states, states, num_symbols, symbols)
m1.print()


# Set forecast variables
r1_num = 10000
t1_num = 10
t0 = 0.0
t1 = np.arange(1.0, 1.0 + float(t1_num), 1.0)
# state0 = 0
# state0 = 1
state0 = states[-1]  # Last observed state


# Forecast from model by samples
states1, symbols1 = m1.sample(r1_num, t1_num, state0=state0, seed=1)
if make_outputs:

    # Print forecast
    print(states1)
    print(symbols1)

    # Calculate probability of each state and symbol
    states1p = np.zeros((t1_num,num_states), dtype=float)
    symbols1p = np.zeros((t1_num,num_symbols), dtype=float)
    for i in range(t1_num):
        for j in range(num_states):
            states1p[i,j] = np.count_nonzero(states1[i,:] == j) / r1_num
        # end for
        for j in range(num_symbols):
            symbols1p[i,j] = np.count_nonzero(symbols1[i,:] == j) / r1_num
        # end for
    # end for
    
    # Make state probabilities plot
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    fig.suptitle('Forecast state probabilites from replicates', fontweight='bold')
    ax.set_xlabel('t')
    ax.set_ylabel('sp')
    ax.grid()
    ax.plot(t1, states1p, label=['st{}'.format(k) for k in range(num_states)])
    fig.legend(loc='upper center', frameon=False,  ncol=10, bbox_to_anchor=(0.5, 0.95))
    ax.set_xlim([0.0, 10.0])
    ax.set_ylim([0.0, 1.0])
    plt.show()
    plt.close()
    
    # Make symbol probabilities plot
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    fig.suptitle('Forecast symbol probabilites from replicates', fontweight='bold')
    ax.set_xlabel('t')
    ax.set_ylabel('op')
    ax.grid()
    ax.plot(t1, symbols1p, label=['sy{}'.format(k) for k in range(num_symbols)])
    fig.legend(loc='upper center', frameon=False,  ncol=10, bbox_to_anchor=(0.5, 0.95))
    ax.set_xlim([0.0, 10.0])
    ax.set_ylim([0.0, 1.0])
    plt.show()
    plt.close()

# end for


# Forecast from model by probabilities
statep1, symbolp1 = m1.probs(t1_num, state0=state0)
if make_outputs:

    # Print forecast
    print(statep1)
    print(symbolp1)

    # Make state probabilities plot
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    fig.suptitle('Forecast state probabilites', fontweight='bold')
    ax.set_xlabel('t')
    ax.set_ylabel('state-p')
    ax.grid()
    ax.plot(t1, statep1, label=['st{}'.format(k) for k in range(num_states)])
    fig.legend(loc='upper center', frameon=False,  ncol=10, bbox_to_anchor=(0.5, 0.95))
    ax.set_xlim([0.0, 10.0])
    ax.set_ylim([0.0, 1.0])
    plt.show()
    plt.close()
        
    # Make symbol probabilities plot
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    fig.suptitle('Forecast symbol probabilites', fontweight='bold')
    ax.set_xlabel('t')
    ax.set_ylabel('symbol-p')
    ax.grid()
    ax.plot(t1, symbolp1, label=['sy{}'.format(k) for k in range(num_symbols)])
    fig.legend(loc='upper center', frameon=False,  ncol=10, bbox_to_anchor=(0.5, 0.95))
    ax.set_xlim([0.0, 10.0])
    ax.set_ylim([0.0, 1.0])
    plt.show()
    plt.close()

# end if


#----------------------------------------------------------


# Print header
print('\nForecast statistics - ENSO')


# Create output data/plots flag
make_outputs = True


# Fit model
m1 = smodel.Model_UMM()
m1.fit(num_states, states, num_symbols, symbols)
m1.print()


# Set forecast variables
t1_num = 10
t0 = 0.0
t1 = np.arange(1.0, 1.0 + float(t1_num), 1.0)


# Calculate Brier Skill Scores
state_BSS, symbol_BSS = m1.BSS(t1_num, states, symbols)
if make_outputs:

    # Print Brier info
    print('\nstate_BSS:\n', state_BSS)
    print('\nsymbol_BSS:\n', symbol_BSS)
    
    # Make BSS plot
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    fig.suptitle('Brier Skill Scores (state forecast)', fontweight='bold')
    ax.set_xlabel('t')
    ax.set_ylabel('BSS')
    ax.grid()
    ax.axhline(y=0.05, c='tab:red', ls=':')
    ax.plot(t1, state_BSS, label='BSS')
    fig.legend(loc='upper center', frameon=False,  ncol=10, bbox_to_anchor=(0.5, 0.95))
    ax.set_xlim([0.0, 10.0])
    ax.set_ylim([0.0, 0.2])
    plt.show()
    plt.close()

    # Make BSS plot
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    fig.suptitle('Brier Skill Scores (symbol forecast)', fontweight='bold')
    ax.set_xlabel('t')
    ax.set_ylabel('BSS')
    ax.grid()
    ax.axhline(y=0.05, c='tab:red', ls=':')
    ax.plot(t1, symbol_BSS, label='BSS')
    fig.legend(loc='upper center', frameon=False,  ncol=10, bbox_to_anchor=(0.5, 0.95))
    ax.set_xlim([0.0, 10.0])
    ax.set_ylim([0.0, 0.2])
    plt.show()
    plt.close()

# end for


#----------------------------------------------------------

# Write footer
print('\nCompleted')

