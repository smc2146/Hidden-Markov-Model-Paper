#==============================================================================
#
# Script is based on annual catchment rainfall from April to March,
# SOI (Jun-Mar) and TPI is from Nov of previous year to Oct just prior
# to the QLD rainfall summer season between from nov to March.
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
from sklearn.metrics import roc_auc_score


# Local modules
from HMM_Functions import GetStartProbs, GetTransitionMatrix, GetEmissionProbs
from DF_Aggregate import daily_to_yearly 


# Variables
DEBUG = 1  # Debug level (0=None, 1=Some, 2=More, ...)


# -------------------------------------
# Read in data from file
# -------------------------------------


# Queensland catchments
c_shape_field = 'BASIN_NAME'
c_shape_values = [
    'Archer', 'Baffle', 'Balonne-Condamine', 'Barron',
    'Black', 'Border Rivers', 'Boyne', 'Brisbane',
    'Bulloo', 'Burdekin', 'Burnett', 'Burrum',
    'Calliope', 'Coleman', 'Cooper Creek', 'Curtis Island',
    'Daintree', 'Diamantina', 'Don', 'Ducie',
    'Embley', 'Endeavour', 'Fitzroy', 'Flinders',
    'Fraser Island', 'Georgina', 'Gilbert', 'Haughton',
    'Herbert', 'Hinchinbrook Island', 'Holroyd', 'Jacky Jacky',
    'Jardine', 'Jeannie', 'Johnstone', 'Kolan',
    'Leichhardt', 'Lockhart', 'Logan-Albert', 'Maroochy',
    'Mary', 'Mitchell', 'Moonie', 'Moreton Bay Islands',
    'Morning', 'Mornington Island', 'Mossman', 'Mulgrave-Russell',
    'Murray', 'Nicholson', 'Noosa', 'Norman',
    'Normanby', "O'Connell", 'Olive-Pascoe', 'Paroo',
    'Pine', 'Pioneer', 'Plane', 'Proserpine',
    'Ross', 'Settlement', 'Shoalwater', 'South Coast',
    'Staaten', 'Stewart', 'Styx', 'Torres Strait Islands',
    'Tully', 'Warrego', 'Waterpark', 'Watson',
    'Wenlock']
c_names = [
    'archer', 'baffle', 'balonne_condamine', 'barron',
    'black', 'border_rivers', 'boyne', 'brisbane',
    'bulloo', 'burdekin', 'burnett', 'burrum',
    'calliope', 'coleman', 'cooper_creek', 'curtis_island',
    'daintree', 'diamantina', 'don', 'ducie',
    'embley', 'endeavour', 'fitzroy', 'flinders',
    'fraser_island', 'georgina', 'gilbert', 'haughton',
    'herbert', 'hinchinbrook_island', 'holroyd', 'jacky_jacky',
    'jardine', 'jeannie', 'johnstone', 'kolan',
    'leichhardt', 'lockhart', 'logan_albert', 'maroochy',
    'mary', 'mitchell', 'moonie', 'moreton_bay_islands',
    'morning', 'mornington_island', 'mossman', 'mulgrave_russell',
    'murray', 'nicholson', 'noosa', 'norman',
    'normanby', 'o_connell', 'olive_pascoe', 'paroo',
    'pine', 'pioneer', 'plane', 'proserpine',
    'ross', 'settlement', 'shoalwater', 'south_coast',
    'staaten', 'stewart', 'styx', 'torres_strait_islands',
    'tully', 'warrego', 'waterpark', 'watson',
    'wenlock']
c_files = r'catchment_averages\{CATCHMENT}.daily.csv'


# Setup variables
c_num = len(c_names)
c_data_names = ['ENP', 'ENN', 'LNP', 'LNN']
c_data_num = 4
c_data = np.zeros((c_num, c_data_num))
c_data_file = r'outputs\Catchment_4_State_Jun.csv'
dfs = []
dff = pd.DataFrame()
# ----------
# If required make output folder
if not os.path.exists('outputs'):
    os.makedirs('outputs')


# Read SOI and TPI data into dataframe
data_df = pd.read_csv(r'data\QLD_Apr_Mar_SOI_Jun_Mar_TPI_Jul_Jun.csv', comment='#')
print('\nInput data file:')
print(data_df)


# Loop through catchment files
for i in range(c_num):

    # Read in data from file (and fix up index)
    df = pd.read_csv(c_files.replace('{CATCHMENT}', c_names[i]))
    df['_Date_'] = pd.to_datetime(df.iloc[:, 0])
    df.drop(columns=df.columns[0], inplace=True)
    df.set_index('_Date_', inplace=True)
    df.index.name = 'Date'


    # Get time data
    dff = daily_to_yearly(df)
    t_data_num = dff.shape[0]
    print('Time data:')
    print('\n  No. time data =', t_data_num)
    print('\n  Catchment_name', c_names[i])


    # Calculate observations from climate data
    # 0 = Below Normal (x<500), 1 = Normal (500<x<644.8), 2 = Above normal (x>644.8), 3 = BB, 4 = NN, 5 = AA
    o_data_num =2
    o_data = np.zeros((t_data_num), dtype=int)
    for j in range(t_data_num):
        if dff['daily_rain'].iloc[j] <= dff['daily_rain'].quantile(0.5):
            o_data[j] = 0
        else:
            o_data[j] = 1
        # end if
    # end for
    print('\nNo. obs states =', o_data_num)
    print('Observations:')
    print(o_data)
    

    # Calculate states from climate index
    # 0 = El Nino, 1 = Neutral, 2 = La Nina
    s_data_num = 2
    s_data = np.zeros((t_data_num), dtype=int)
    for j in range(t_data_num):
        if data_df['SOI_Jun_Mar'][j] <= 1.1:
            s_data[j] = 0
        else:
            s_data[j] = 1
        # end if
    # end for
    print('\nNo. hidden states =', s_data_num)
    print('States:')
    print(s_data)
   
    
    # # Calculate TPI states
    # # 0=P(Positive TPI), 1=N (negative TPI)
    tpi_data_num = 2
    tpi_data = np.zeros((t_data_num), dtype=int)
    for j in range(t_data_num):
        if data_df['TPI_Jul_Jun_+01'][j] >= -0.02:
            tpi_data[j] = 0
        else:
            tpi_data[j] = 1
        # end if
    # end for
    print('\nNo. Tpi states =', tpi_data_num)
    print('TPI states:')
    print(tpi_data)
    
    
    # # Calculate ENSO-TPI states
    # # 0=ENP, 1=ENN, 2=NP, 3=NN, 4=LNP, 5=LNN
    s_tpi_data_num=4
    s_tpi_data=np.zeros((t_data_num), dtype=int)
    for j in range(t_data_num):
        if s_data[j] == 0 and tpi_data[j] == 0:
            s_tpi_data[j] = 0
        elif s_data[j] == 0 and tpi_data[j] == 1:
            s_tpi_data[j] = 1
        elif s_data[j] == 1 and tpi_data[j] == 0:
            s_tpi_data[j] = 2    
        else:
            s_tpi_data[j] = 3
        # end if
        print('\nNo. ENSO_Tpi states =', s_tpi_data_num)
        print('ENSO_Tpi states:')
        print(s_tpi_data)
    # end for
    
    # -------------------------------------
    # Calculate properties
    # -------------------------------------


    # Calculate state probabilities
    s_probs = GetStartProbs(s_tpi_data_num, s_tpi_data)
    print('\nStart probabilities:')
    print(s_probs)


    # Calculate Transition matrix
    t_matrix = GetTransitionMatrix(s_tpi_data_num, s_tpi_data, adjust_zero_states=True)
    print('\nTransmission matrix:')
    print(t_matrix)


    # Calculate emission matrix
    e_probs = GetEmissionProbs(s_tpi_data_num, s_tpi_data, o_data_num, o_data)
    print('\nEmission probabilities:')
    print(e_probs)


    # -------------------------------------
    # Sample data from known model
    # -------------------------------------


    # Seed random number generator is used to initialise the random number generator i.e. a random number to start with (a seed value) to be able to generate random numbers
    rs = check_random_state(1)


    # Set up model
    model = hmm.CategoricalHMM(n_components=4)  # 3 hidden states
    model.n_features = 2 # 3 observed symbols
    model.startprob_ = s_probs
    model.transmat_ = t_matrix
    model.emissionprob_ = e_probs


    # Make a sample
    gen_symbols, gen_states = model.sample(n_samples=1000, random_state=rs)
    
    
    # Print sample's states
    print('\nStates:')
    fig, ax = plt.subplots()
    ax.set_title('Generated states (first 100)')
    ax.plot(gen_states[:100])
    plt.show()
    
    
    # Print sample's symbols
    print('\nSymbols:')
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


    S=model.predict_proba(X)

    
    # -------------------------------------
    # Model Evaluation
    # -------------------------------------
    
    
    s_tpi_data_ENP = np.zeros((t_data_num), dtype=int)
    s_tpi_data_ENN = np.zeros((t_data_num), dtype=int)
    s_tpi_data_NP = np.zeros((t_data_num), dtype=int)
    s_tpi_data_NN = np.zeros((t_data_num), dtype=int)
    s_tpi_data_LNP = np.zeros((t_data_num), dtype=int)
    s_tpi_data_LNN = np.zeros((t_data_num), dtype=int)
    for j in range(t_data_num):
        if s_tpi_data[j] == 0:
            s_tpi_data_ENP[j] = 1
        elif s_tpi_data[j] == 1:
            s_tpi_data_ENN[j] = 1
        elif s_tpi_data[j] == 2:
            s_tpi_data_LNP[j] = 1
        elif s_tpi_data[j] == 3:
            s_tpi_data_LNN[j] = 1
        else:
            s_tpi_data_ENP[j] = 0 
            s_tpi_data_ENN[j] = 0
            s_tpi_data_LNP[j] = 0
            s_tpi_data_LNN[j] = 0
        # end if
    # end for
    print("\nENP prediction ")
    print(s_tpi_data_ENP)
    print("\nENN prediction ")
    print(s_tpi_data_ENN)
    print("\nLNP prediction ")
    print(s_tpi_data_LNP)
    print("\nLNN prediction ")
    print(s_tpi_data_LNN)

    
    predictions_ENP = S[:,0]  # Example predicted probabilities
    ground_truth_ENP = s_tpi_data_ENP  # Example ground truth labels (0 or 1)

    
    predictions_ENN = S[:,1]  # Example predicted probabilities
    ground_truth_ENN = s_tpi_data_ENN  # Example ground truth labels (0 or 1)

    
    predictions_LNP = S[:,2]  # Example predicted probabilities
    ground_truth_LNP = s_tpi_data_LNP  # Example ground truth labels (0 or 1)

    
    predictions_LNN = S[:,3]  # Example predicted probabilities
    ground_truth_LNN = s_tpi_data_LNN  # Example ground truth labels (0 or 1)


    #------------------------------------------------------------------------------
    #Generate ROC file
    #------------------------------------------------------------------------------
    
    # Calculate ROC AUC score
    roc_auc_ENP = roc_auc_score(ground_truth_ENP, predictions_ENP)
    roc_auc_ENN = roc_auc_score(ground_truth_ENN, predictions_ENN)
    roc_auc_LNP = roc_auc_score(ground_truth_LNP, predictions_LNP)
    roc_auc_LNN = roc_auc_score(ground_truth_LNN, predictions_LNN)
    
    # Calculate output statitics
    c_data[i,0] = roc_auc_ENP
    c_data[i,1] = roc_auc_ENN
    c_data[i,2] = roc_auc_LNP
    c_data[i,3] = roc_auc_LNN


# end for

    
# Write output file
with open(c_data_file, 'w') as f:
     f.write(c_shape_field)
     for k in range(c_data_num):
         f.write(',' + c_data_names[k])
     f.write('\n')
     for i in range(c_num):
         f.write(c_shape_values[i])
         for k in range(c_data_num):
             f.write(',' + str(c_data[i,k]))
         f.write('\n')
     # end for
# end with      
    
