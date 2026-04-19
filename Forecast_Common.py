#==============================================================================
#
# Forecast - Common functions
#
#------------------------------------------------------------------------------
#
# Contains:
#
#   report_error()
#   report_warning()
#   is_file_open()
#
#==============================================================================




# Imports
import os, sys




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




#------------------------------------------------------------------------------
# Check if file is open
#------------------------------------------------------------------------------
def is_file_open(fname):
    try:
        with open(fname, 'a+') as f:
            fout = True
    except IOError:
        fout = False
    return fout
# end def

