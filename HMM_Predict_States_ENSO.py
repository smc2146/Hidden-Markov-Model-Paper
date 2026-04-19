#==============================================================================
#
# Hidden Markov Model - Generate from known states - ENSO
#
#------------------------------------------------------------------------------
#
# Notes:
#   - This assumes:
#       (a) Known model.
#       (b) Known states.
#
#==============================================================================


# Modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.utils import check_random_state
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc


# Local modules
from HMM_Functions import GetStartProbs, GetTransitionMatrix, GetEmissionProbs


# Variables
DEBUG = 1  # Debug level (0=None, 1=Some, 2=More, ...)


# -------------------------------------
# Read in data from file
# -------------------------------------

# Catchment name
c_name = 'QLD'


# Output plot files
plot_gen1_states = r'plots\{CATCHMENT}.gen1_states.png'
plot_gen1_symbols = r'plots\{CATCHMENT}.gen1_symbols.png'
# ----------
plot_rel_diag = r'plots\{CATCHMENT}.rel_diag.png'
plot_roc_curve = r'plots\{CATCHMENT}.roc_curve.png'
plot_rel_diag_all = r'plots\{CATCHMENT}.rel_diag_all.png'
plot_roc_curve_all = r'plots\{CATCHMENT}.roc_curve_all.png'
# ----------
plot_fcst_states_p = r'plots\{CATCHMENT}.fcst_states_p.png' 
plot_fcst_symbols_p = r'plots\{CATCHMENT}.fcst_symbols_p.png'
plot_fcst_states_r = r'plots\{CATCHMENT}.fcst_states_r.png'
plot_fcst_symbols_r = r'plots\{CATCHMENT}.fcst_symbols_r.png'
# ----------
# If required make output folder
if not os.path.exists('plots'):
    os.makedirs('plots')


# Flags to turn off displaying data and plots
show_data_flag = False
show_plots_flag = True


# Setup variables
c_num = 1
c_data_names = ['Yr0', 'Yr1', 'Yr2', 'Yr3', 'Yr4', 'Yr5', 'Yr6', 'Yr7', 'Yr8', 'Yr9', 'Yr10']
c_data_num = 11
c_data = np.zeros((c_num, c_data_num))
c_data_file = 'QLD_prediction.csv'


# Read data into dataframe
data_df = pd.read_csv(r'data\QLD_Apr_Mar_SOI_Jun_Mar_TPI_Jul_Jun.csv', comment='#')
print('\nInput data file:')
print(data_df)


# Get time data
t_data = data_df['Year'].to_numpy()
t_data_num = t_data.shape[0]
print('\nNo. time data =', t_data_num)
print('Time data:')
print(t_data)


# Calculate observations from climate data
# 0 = Below Normal (x<500), 1 = Normal (500<x<644.8), 2 = Above normal (x>644.8), 3 = BB, 4 = NN, 5 = AA
o_data_num = 3
o_data = np.zeros((t_data_num), dtype=int)
for i in range(t_data_num):
    if data_df['QLD_Apr_Mar'][i] < data_df['QLD_Apr_Mar'].quantile(0.33):
        o_data[i] = 0
    elif data_df['QLD_Apr_Mar'][i] <= data_df['QLD_Apr_Mar'].quantile(0.67):
        o_data[i] = 1
    else:
        o_data[i] = 2
    # end if
# end for
print('\nNo. obs states =', o_data_num)
print( 'Tercile1', data_df['QLD_Apr_Mar'].quantile(0.33))
print( 'Tercile1', data_df['QLD_Apr_Mar'].quantile(0.67))
print('Observations:')
print(o_data)


# Calculate states from climate index
# 0 = El Nino, 1 = Neutral, 2 = La Nina
s_data_num = 3
s_data = np.zeros((t_data_num), dtype=int)
for i in range(t_data_num):
    if data_df['SOI_Jun_Mar'][i] <= -3.85:
        s_data[i] = 0
    elif data_df['SOI_Jun_Mar'][i] <= 4.5:
        s_data[i] = 1
    else:
        s_data[i] = 2
    # end if
# end for
print('\nNo. hidden states =', s_data_num)
print('States:')
print(s_data)


# -------------------------------------
# Calculate properties
# -------------------------------------


# Calculate state probabilities
s_probs = GetStartProbs(s_data_num, s_data)
print('\nStart probabilities:')
print(s_probs)


# Calculate Transition matrix
t_matrix = GetTransitionMatrix(s_data_num, s_data)
print('\nTransmission matrix:')
print(t_matrix)


# Calculate emission matrix
e_probs = GetEmissionProbs(s_data_num, s_data, o_data_num, o_data)
print('\nEmission probabilities:')
print(e_probs)


# -------------------------------------
# Sample data from known model
# -------------------------------------


# Seed random number generator is used to initialise the random number generator i.e. a random number to start with (a seed value) to be able to generate random numbers
rs = check_random_state(1)


# Set up model
model = hmm.CategoricalHMM(n_components=3)  # 3 hidden states
model.n_features = 3 # 3 observed symbols
model.startprob_ = s_probs
model.transmat_ = t_matrix
model.emissionprob_ = e_probs


# Make a sample
gen_symbols, gen_states = model.sample(n_samples=1000, random_state=rs)


# Print sample's states
print('\nStates:')
# print(gen_states)
fig, ax = plt.subplots()
ax.set_title('Generated states (first 100)')
ax.plot(gen_states[:100])
plt.show()


# Print sample's symbols
print('\nSymbols:')
# print(gen_symbols.flatten())
fig, ax = plt.subplots()
ax.set_title('Generated symbols (first 100)')
ax.plot(gen_symbols[:100])
plt.show()


print('\nBack-calculate probable states from observations:')
print('Original states:')
print(s_data)
X = o_data[:,np.newaxis]
Y = model.predict(X)
print('Back-calculated states:')
print(Y)

S = model.predict_proba(X)


# -------------------------------------
# Model Evaluation
# -------------------------------------

s_data_EN = np.zeros((t_data_num), dtype=int)
s_data_N = np.zeros((t_data_num), dtype=int)
s_data_LN = np.zeros((t_data_num), dtype=int)

for i in range(t_data_num):
    if s_data[i] == 0:
        s_data_EN[i] = 1
    elif s_data[i] == 1:
        s_data_N[i] = 1    
    elif s_data[i] == 2:
        s_data_LN[i] = 1
    else:
        s_data_EN[i] = 0 
        
        s_data_N[i] = 0
        
        s_data_LN[i] = 0
        
    # end if
# end for
print("\nEN prediction ")
print(s_data_EN)

print("\nN prediction ")
print(s_data_N)

print("\nLN prediction ")
print(s_data_LN)


predictions_EN = S[:,0]  # Example predicted probabilities
ground_truth_EN = s_data_EN  # Example ground truth labels (0 or 1)


predictions_N = S[:,1]  # Example predicted probabilities
ground_truth_N = s_data_N  # Example ground truth labels (0 or 1)


predictions_LN = S[:,2]  # Example predicted probabilities
ground_truth_LN = s_data_LN  # Example ground truth labels (0 or 1)


# Create reliability diagram
prob_true_EN, prob_pred_EN = calibration_curve(ground_truth_EN, predictions_EN, n_bins=10, strategy='uniform')
prob_true_N, prob_pred_N = calibration_curve(ground_truth_N, predictions_N, n_bins=10, strategy='uniform')
prob_true_LN, prob_pred_LN = calibration_curve(ground_truth_LN, predictions_LN, n_bins=10, strategy='uniform')


# Plot reliability diagram
plt.figure()
plt.plot(prob_pred_EN, prob_true_EN, marker='o', color='red', label='EN curve')
plt.plot(prob_pred_N, prob_true_N, marker='o', color='yellow', label='N curve')
plt.plot(prob_pred_LN, prob_true_LN, marker='o', color='blue', label='LN curve')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.plot([0, 1], [0.165, 0.665], color='grey', linestyle='dashdot', label='No skill')
plt.axhline(y=0.33,  color='grey', linestyle='dotted', label='No Resolution')
plt.axvline(x=0.33,  color='grey', linestyle='dotted', label=None)
x1=[0.33,1,1,0.33]
y1=[0.33,0.665,1,1]
x2=[0,0.33,0.33,0]
y2=[0,0,0.33,0.165]
x3=[0.65,]
y3=[0.9]
plt.fill_between(x1,y1, color='lightgray')
plt.fill_between(x2,y2, color='lightgray')
plt.text(0.65,0.9, 'Skill')
plt.xlabel('Mean predicted probability', fontweight='bold')
plt.ylabel('Fraction of positives', fontweight='bold')
plt.title('QLD Reliability Diagram', fontweight='bold')
plt.legend(loc='lower right')
plt.show()


# Calculate ROC curve
fpr_EN, tpr_EN, thresholds = roc_curve(ground_truth_EN, predictions_EN)
fpr_N, tpr_N, thresholds = roc_curve(ground_truth_N, predictions_N)
fpr_LN, tpr_LN, thresholds = roc_curve(ground_truth_LN, predictions_LN)


# Calculate AUC (Area Under Curve)
roc_auc_EN = auc(fpr_EN, tpr_EN)
roc_auc_N = auc(fpr_N, tpr_N)
roc_auc_LN = auc(fpr_LN, tpr_LN)


# Plot ROC curve
plt.figure()
plt.plot(fpr_EN, tpr_EN, color='red', lw=2, label=f'EN_ROC curve (AUC = {roc_auc_EN:.2f})')
plt.plot(fpr_N, tpr_N, color='yellow', lw=2, label=f'N_ROC curve (AUC = {roc_auc_N:.2f})')
plt.plot(fpr_LN, tpr_LN, color='blue', lw=2, label=f'LN_ROC curve (AUC = {roc_auc_LN:.2f})')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontweight='bold')
plt.ylabel('True Positive Rate', fontweight='bold')
plt.title('QLD Relative Operating Characteristic (ROC)', fontweight='bold')
plt.legend(loc='lower right')
plt.show()


#--------------------------------------------------------------------
# Predict next state
#--------------------------------------------------------------------


S1 = np.zeros([11,3], dtype=float)
S1[0,:] = np.array([1, 0, 0])
t_mat=t_matrix.transpose()
Y1=np.zeros([10], dtype=int)
def forecast(S1, t_mat, e_probs):
    alpha = np.zeros([e_probs.shape[1],t_mat.shape[1] ], dtype=float)
    max_values = np.zeros(t_mat.shape[1], dtype=float)
    for t in range(1, S1.shape[0]):
        for k in range(e_probs.shape[1]):
            for j in range(t_mat.shape[0]):
                alpha[k, j] = S1[t-1].dot(t_matrix[:, j])
        max_values = alpha.max(axis=0)
        print('\n Max values')
        print(max_values)
        S1[t,:] =  max_values
    Predicted_states = S1[1:].argmax(axis=1)
    print('\n Predicted probabilities')
    print(S1)
    print('\n Prediced states')
    print(Predicted_states)
    return S1
  

S1 = forecast(S1, t_mat, e_probs)


# Number of hidden states
s_num = 3 
s_str = ['s{}'.format(k) for k in range(s_num)]


# Number of observed symbols
o_num = 3
o_str = ['o{}'.format(k) for k in range(o_num)]


# Forecast time array (first element is intial state at time=0.0)
t1_num = 1 + 10
t1 = np.arange(0.0, float(t1_num), 1.0)


# Forecast starting state probabilities
sp0 = np.array([0.0, 0.0, 1.0], dtype=float) #2021/22 observed state


# Forecast starting state (first state is numbered 0)
s0 = s_data[-1]  #Last observed state


# Forecast starting symbol (first symbol is numbered 0)
o0 = o_data[-1]  # Last observed state


# Number of replicates
r1_num = 10000


# See random number generator
np.random.seed(0)


# --- Forecast State Probabilities ---
if show_data_flag:
    print('\nForecast state probabilities test')


# Forecast state probabilities
tmatrix_t = t_matrix.transpose()
sp1 = np.zeros((t1_num, s_num), dtype=float)
sp1[0,:] = sp0
for l in range(1, t1_num):
    sp1[l,:] = np.matmul(sp1[l-1,:], t_matrix)
# end for
    

# Make state probabilities plot
fig, ax = plt.subplots(1, 1, figsize=(10,5))
fig.suptitle('Forecast state probabilites - ' + 'QLD', fontweight='bold')
ax.set_xlabel('t')
ax.set_ylabel('sp')
ax.grid()
ax.plot(t1, sp1, label=s_str)
fig.legend(loc='upper center', frameon=False,  ncol=10, bbox_to_anchor=(0.5, 0.95))
ax.set_xlim([0.0, 10.0])
ax.set_ylim([0.0, 1.0])
if show_plots_flag:
    plt.show()
else:
    plt.savefig(plot_fcst_states_p.replace('{CATCHMENT}', c_name),
                dpi=300, format='png', bbox_inches='tight')
    plt.close()


# --- Forecast Symbol Probabilities ---

# Forecast symbol probabilities
op1 = np.zeros((t1_num, o_num), dtype=float)
op0 = np.array([0, 1, 0], dtype=float) #2021/22 observed sysmbol
op1[0,:] = op0
for l in range(1, t1_num):
    op1[l,:] = np.matmul(sp1[l,:], e_probs)
# end for


# Make symbol probabilities plot
fig, ax = plt.subplots(1, 1, figsize=(10,5))
fig.suptitle('Forecast symbol probabilites - ' + 'QLD', fontweight='bold')
ax.set_xlabel('t')
ax.set_ylabel('op')
ax.grid()
ax.plot(t1, op1, label=o_str)
fig.legend(loc='upper center', frameon=False,  ncol=10, bbox_to_anchor=(0.5, 0.95))
ax.set_xlim([0.0, 10.0])
ax.set_ylim([0.0, 1.0])
if show_plots_flag:
    plt.show()
else:
    plt.savefig(plot_fcst_symbols_p.replace('{CATCHMENT}', c_name),
                dpi=300, format='png', bbox_inches='tight')
    plt.close()


# --- Forecast State by Replicates ---
if show_data_flag:
    print('\nForecast state replicates test')

# Forecast state replicates
u = np.random.uniform(size=(t1_num, r1_num))
s1 = np.zeros((t1_num, r1_num), dtype=int)
tmatix_c = np.cumsum(t_matrix, axis=1)
for k in range(r1_num):
    s1[0,k] = s0
    for l in range(1, t1_num):
        s1[l,k] = np.searchsorted(tmatix_c[s1[l-1,k],:], u[l,k])
    # end for
# end for
if show_data_flag:
    print('tmatix_c:')
    print('  ', np.array2string(tmatix_c, prefix='  ', formatter={'float_kind':lambda x: "%.3f" % x}))
    print(' u:')
    print('  ', np.array2string(u[0:11,1], prefix='  ', formatter={'float_kind':lambda x: "%.3f" % x}))
    print('s1:')
    print('  ', np.array2string(s1[0:11,1], prefix='  '))

# Plot probability of each state
sp2 = np.zeros((t1_num,s_num), dtype=float)
sp2[0,:] = sp0
for l in range(1, t1_num):
    for k in range(s_num):
        sp2[l,k] = np.count_nonzero(s1[l,:] == k) / r1_num
    # end for
# end for


# Make state probabilities plot
fig, ax = plt.subplots(1, 1, figsize=(10,5))
fig.suptitle('Forecast state probabilites from replicates - ' + 'QLD', fontweight='bold')
ax.set_xlabel('t')
ax.set_ylabel('sp')
ax.grid()
ax.plot(t1, sp2, label=s_str)
fig.legend(loc='upper center', frameon=False,  ncol=10, bbox_to_anchor=(0.5, 0.95))
ax.set_xlim([0.0, 10.0])
ax.set_ylim([0.0, 1.0])
if show_plots_flag:
    plt.show()
else:
    plt.savefig(plot_fcst_states_r.replace('{CATCHMENT}', c_name),
                dpi=300, format='png', bbox_inches='tight')
    plt.close()


# --- Forecast Symbols by Replicates ---

# Forecast symbols
u = np.random.uniform(size=(t1_num, r1_num))
o1 = np.zeros((t1_num, r1_num), dtype=int)
eprobs_c = np.cumsum(e_probs, axis=1)
for k in range(r1_num):
    o1[0,k] = o0
    for l in range(1, t1_num):
        o1[l,k] = np.searchsorted(eprobs_c[s1[l,k],:], u[l,k])
    # end for
# end for


# Plot probability of each symbol
op2 = np.zeros((t1_num,o_num), dtype=float)
for l in range(t1_num):
    for k in range(o_num):
        op2[l,k] = np.count_nonzero(o1[l,:] == k) / r1_num
    # end for
# end for


# Make state probabilities plot
fig, ax = plt.subplots(1, 1, figsize=(10,5))
fig.suptitle('Forecast symbol probabilites from replicates - ' + 'QLD', fontweight='bold')
ax.set_xlabel('t')
ax.set_ylabel('op')
ax.grid()
ax.plot(t1, op2, label=o_str)
fig.legend(loc='upper center', frameon=False,  ncol=10, bbox_to_anchor=(0.5, 0.95))
ax.set_xlim([0.0, 10.0])
ax.set_ylim([0.0, 1.0])
if show_plots_flag:
    plt.show()
else:
    plt.savefig(plot_fcst_symbols_r.replace('{CATCHMENT}', c_name),
                dpi=300, format='png', bbox_inches='tight')
    plt.close()

