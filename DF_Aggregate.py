#==============================================================================
#
# Data utility routines module for data processing
#
#------------------------------------------------------------------------------
#
# Functions:
#
#   daily_to_monthly_df()
#   daily_to_yearly_df()
#
#------------------------------------------------------------------------------
#
# Notes:
#
#==============================================================================


# Modules
import sys
import pandas as pd
from calendar import month_abbr




#------------------------------------------------------------------------------
# Aggregate daily dataframe into monthly dataframe
#------------------------------------------------------------------------------
def daily_to_monthly(daily_df, agg_option='SUM', date_option='END'):
    """
    Aggregate daily dataframe into monthly dataframe
    
    - Trims start and end incomplete months
    - Can calculate sum or average of month
    - Can specify output dates at start or end of month

    Parameters
    ----------
    daily_df : datafame
        Daily data with datetime index
    agg_option : str, optional
        Aggregate interval as either 'AVG', 'SUM'
    date_option : str, optional
        Output dates at 'START' or 'END' of interval

    Returns
    ----------
    dataframe
        Monthly data with datetime as index

    """

    # Aggregate to monthly
    rule = 'MS'
    if agg_option.upper() == 'SUM':
        monthly_df = daily_df.resample(rule).sum()
    elif agg_option.upper() == 'AVG':
        monthly_df = daily_df.resample(rule).mean()
    else:
        report_error('Invalid agg_option', agg_option)
    # end if
        
    # Apply output date option
    if date_option.upper() == 'START':
        pass
    elif date_option.upper() == 'END':
        monthly_df.index = monthly_df.index + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    else:
        report_error('Invalid date_option', date_option)
    # end if

    # Trim incomplete months
    inum = daily_df.resample(rule).count()
    itot = ((monthly_df.index[0] + pd.DateOffset(months=1)) - monthly_df.index[0]).days
    if inum.iloc[0,0] < itot:
        monthly_df.drop(monthly_df.index[0], inplace=True)
    # end if
    itot = ((monthly_df.index[-1] + pd.DateOffset(months=1)) - monthly_df.index[-1]).days
    if inum.iloc[-1,0] < itot:
        monthly_df.drop(monthly_df.index[-1], inplace=True)
    # end if

    # Return monthly dataframe
    return monthly_df

# end def




#------------------------------------------------------------------------------
# Aggregate daily dataframe into yearly dataframe
#------------------------------------------------------------------------------
def daily_to_yearly(daily_df, agg_option='SUM', date_option='END', start_month=7):
    """
    Aggregate daily dataframe into yearly dataframe
    
    - Trims start and end incomplete years
    - Can calculate sum or average of year
    - Can specify output dates at start or end of year
    - Can specify aggregation start month

    Parameters
    ----------
    daily_df : datafame
        Daily data with datetime index
    agg_option : str, optional
        Aggregate interval as either 'AVG', 'SUM'
    date_option : str, optional
        Output dates at 'START' or 'END' of interval

    Returns
    ----------
    dataframe
        Yearly data with datetime as index

    """

    # Aggregate to yearly
    if start_month >= 1 and start_month <= 12:
        rule = 'YS-' + month_abbr[start_month].upper()
    else:
        report_error('Invalid start_month', start_month)
    # end if
    if agg_option.upper() == 'SUM':
        yearly_df = daily_df.resample(rule).sum()
    elif agg_option.upper() == 'AVG':
        yearly_df = daily_df.resample(rule).mean()
    else:
        report_error('Invalid agg_option', agg_option)
    # end if
        
    # Apply output date option
    if date_option.upper() == 'START':
        pass
    elif date_option.upper() == 'END':
        yearly_df.index = yearly_df.index + pd.DateOffset(years=1) - pd.DateOffset(days=1)
    else:
        report_error('Invalid date_option', date_option)
    # end if

    # Trim incomplete years
    inum = daily_df.resample(rule).count()
    itot = ((yearly_df.index[0] + pd.DateOffset(years=1)) - yearly_df.index[0]).days
    if inum.iloc[0,0] < itot:
        yearly_df.drop(yearly_df.index[0], inplace=True)
    # end if
    itot = ((yearly_df.index[-1] + pd.DateOffset(years=1)) - yearly_df.index[-1]).days
    if inum.iloc[-1,0] < itot:
        yearly_df.drop(yearly_df.index[-1], inplace=True)
    # end if

    # Return yearly dataframe
    return yearly_df

# end def




#------------------------------------------------------------------------------
# Convert date column to index for dataframe
#   - First column must be date.
#------------------------------------------------------------------------------
def convert_date_index(df):

    df['_Date_'] = pd.to_datetime(df.iloc[:, 0])
    df.drop(columns=df.columns[0], axis=1,  inplace=True)
    df.set_index('_Date_', inplace=True)
    df.index.name = 'Date'

# end def




#------------------------------------------------------------------------------
# Report warning
#------------------------------------------------------------------------------
def report_warning(str1, str2=None, str3=None):

    print('\nWarning: {}'.format(str1))
    if not str2 is None:
        print('Check: {}'.format(str2))
    # end if
    if not str3 is None:
        print('Check: {}'.format(str3))
    # end if

# end def




#------------------------------------------------------------------------------
# Report error and exit
#------------------------------------------------------------------------------
def report_error(str1, str2=None, str3=None):

    print('\nError: {}'.format(str1))
    if not str2 is None:
        print('Check: {}'.format(str2))
    # end if
    if not str3 is None:
        print('Check: {}'.format(str3))
    # end if
    sys.exit()

# end def

