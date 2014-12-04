# Functions to support the linear regression ipython notebook work.

import pandas as pd
import numpy as np


def add_dummy(df, categories, label, drop=False):#, interaction=None):
    """
df - dataframe in which to place new dummy variables
categories - categorical variable from which make dummy variables
label - string of how to label each dummy column.
drop - Boolean indicating whether to drop a column of dummies
"""

    # Get dataframe of dummy variables from categories
    dum = pd.get_dummies(categories)

    # Set index to match that of new dataframe
    dum = dum.set_index(df.index)

    # Label columns of dummy variables
    dum.columns = [label+'_'+str(val) for val in dum.columns]

    # Drop one column of dummy variable so that no column is 
    # a linear combination of another. Do this when using 
    # a constant in the linear model.
    if drop==True:
        dum.drop(dum.columns[0],axis=1,inplace=True)

    # Join new dummy dataframe to the dataframe of variables
    # for the regression
    df = df.join(dum)

    # Return new updated dataframe to the calling program
    return df


def add_interactions(df, variables):
    """
df - dataframe in which to place interaction terms
variables - list of names from which to create interaction terms
"""

    # Enumerate all variables in each group
    vardict = {}
    for var in variables:
        if var=='Hour':
            vardict[var] = ['Hour_'+str(val) for val in range(1,24)]
        if var=='Day':
            vardict[var] = ['Day_'+str(val) for val in range(1,7)]
        if var=='Season':
            vardict[var] = ['Season_'+str(val) for val in range(1,4)]
        if var=='Month':
            vardict[var] = ['Month_'+str(val+1) for val in range(1,12)]
        if var=='Weather':
            vardict[var] = ['Weather_'+str(val+1) for val in range(1,4)]
        if var=='days_elapsed':
            vardict[var] = 'days_elapsed'

    # Add interaction between all items in the variable dictionary
    if len(variables)==2:
        for value1 in vardict.values()[0]:
            for value2 in vardict.values()[1]:
                newname = value1+'_*_'+value2
                df[newname] = df[value1]*df[value2]

    if len(variables)==3:
        for value1 in vardict.values()[0]:
            for value2 in vardict.values()[1]:
                for value3 in vardict.values()[2]:
                    newname = value1+'_*_'+value2+'_*_'+value3
                    df[newname] = df[value1]*df[value2]*df[value3]

    if len(variables)==4:
        for value1 in vardict.values()[0]:
            for value2 in vardict.values()[1]:
                for value3 in vardict.values()[2]:
                    for value4 in vardict.values()[3]:
                        newname = value1+'_*_'+value2+'_*_'+value3+'_*_'+value4
                        df[newname] = df[value1]*df[value2]*df[value3]*df[value4]

    # Return dataframe to calling program
    return df



def make_matrix(df, monthly_scale={}, weather_scale={}, constant=False):
    """
Function to build matrix for statsmodels regression.
monthly_scale - list of values by which to scale input for each month
weather_scale - list of values by which to scale weather for each type
constant - if True, adds constant to regression.
"""

    # Define new dataframe to hold the matrix
    X = pd.DataFrame(index=df.index)

    # Add time variables to the predictor variables matrix
    months_elapsed = []
    for val in X.index:
        yeardiff = val.year - X.index[0].year
        monthdiff = val.month - X.index[0].month
        months_elapsed.append(12*yeardiff + monthdiff)

    X = add_dummy(X, months_elapsed, 'Months_Elapsed', drop=True)
    X = add_dummy(X, df.index.hour, 'Hour', drop=True)
    X = add_dummy(X, df.index.dayofweek, 'Day', drop=True)
    X = add_dummy(X, df.weather, 'Weather', drop=True)
    # Add holidays by forcing them to look like a Saturday
    X['Day_5'][df.holiday==1] = 1
        
    # Add interaction terms
    # After some experimentation, the major interaction term is Day_of_week*Hour_of_Day
    # This is the big one. Each day of the week has its own daily pattern of rides
    X = add_interactions(X, ['Day','Hour'])

    # Most of the weather data proves unreliable
    #X['temp'] = df.temp


    # Scale each row of X by the mean ridership that month.
    # This lets us scale our model to mean ridership each month, so the
    # months with fewer riders will have less pronounced daily cycles
    if monthly_scale!={}:
        for time,scale in monthly_scale.iteritems():
            this = (df.index.month==time.month) & (df.index.year==time.year)
            X[this] = scale*X[this]

    # Scale each row of X by the mean weather during that weather type
    # This lets us scale our model to mean ridership during different types
    # of weather.
    if weather_scale!={}:
        for weather,scale in weather_scale.iteritems():
            this = (df['weather']==weather)
            X[this] = scale*X[this]

    # Do not add constant because we already have mean offsets for each month of data.
    # A constant would be a linear combination of the monthly indicator variables.
    if constant==True:
        X = sm.add_constant(X,prepend=True)

    # Return dataframe to calling program
    return X



def score( obs, predict ):
    """
Calculate score on the predictions (predict) given the observations (obs).
"""

    rmsle = np.sqrt( np.sum( 1./len(obs) * (np.log(predict+1) - np.log(obs+1))**2 ) )
    return rmsle



def get_scale(df,types):
    """
Return dictionary of scale factors for all types in types.
"""

    outdict = {}
    for tt in types:

        if tt=='weather':
            weather_scale = {}
            weather_mean = df['count'].groupby(df['weather']).mean()
            for ii in range(len(weather_mean)):
                weather_scale[weather_mean.index[ii]] = weather_mean.iloc[ii] / df['count'].mean()
            outdict[tt] = weather_scale

        if tt=='monthly':
            monthly_scale = {}
            monthly_mean = df['count'].resample('1m',how=np.mean)
            for ii in range(len(monthly_mean)):
                monthly_scale[monthly_mean.index[ii]] = monthly_mean.iloc[ii] / df['count'].mean()
            outdict[tt] = monthly_scale

    return outdict
