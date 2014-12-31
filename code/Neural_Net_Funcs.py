# Functions to assist the use of neural networks for the kaggle competition.

import pandas as pd
import numpy as np

# Create matrix of input vectors for the neural network.
#    Input is m x n matrix.
#       m - number of observations
#       n - number of elements in input layer
def make_input_output(df,Age=True,Test=False):
    """
Creates input and output arrays for the neural network.
    The result is a tuple that contains two elements:
        1. The inputs: a [m x n] array
        2. The output: a [m x 1] array
            m - number of observations
            n - number of elements in input layer

INPUT
    df - dataframe read from csv of titanic train/test data
         ex. df = pd.read_csv("train.csv",index_col="PassengerId")
    Age - if True, include age in the input matrix. NaN rows are removed.
    Test - if True, then we are processing the test data, not the train
           data. There are no output labels, so replace with any values.
"""

    # Build the matrix as a pandas dataframe. Then, convert the
    # dataframe to a numpy array using pandas.dataframe.as_matrix()

    # Create dataframe
    matrix = pd.DataFrame(index=df.index)

    # Add column for the output array
    if Test==True:
        output = np.zeros(len(df))
    else:
        output = np.where(df['Survived']==1,df['Survived'],-1)
    matrix['Output'] = output

    # ---- Add inputs with linear activation functions
    linvars = ['SibSp','Parch','Fare']
    if Age==True:
        linvars.append('Age')

    # Add variables to input matrix, and scale values so that they
    # have mean=0 and range (max-min)=1.
    # Replace null values with median values if not processing Age
    for var in linvars:
        matrix[var] = (df[var]-df[var].mean()) / (df[var].max()-df[var].min())
        if var!='Age':
            matrix[var][matrix[var].isnull()] = np.median(matrix[var])


    # ---- Add categorical variables
    catvars = ['Sex','Pclass','Embarked']
    for var in catvars:

        # Get dummy variables, add index and column names
        dummies = pd.get_dummies(df[var])
        dummies = dummies.set_index(df.index)
        dummies.columns = [var+'_'+str(value) for value in dummies.columns]

        # Add dummy variables to the matrix
        # Shift dummy variables so that they have value in {-1,1}
        for col in dummies.columns:
            aa = np.where(dummies[col]==1,1,-1)
            matrix[col] = np.where(dummies[col]==1,1,-1)

    # Once all variables are added, remove rows with nan Age values
    # If Age==False, keep columns in which Age.isnull()==True
    if Age==False:
        matrix = matrix[df['Age'].isnull()]
    else:
        matrix = matrix[matrix['Age'].notnull()]


    # Get list of minimum and maximum values for each type of input (column)
    minmaxlist = []
    for col in matrix.columns:
        if col=='Output': continue

        # For each input column, set min/max values to a range twice that
        #    of the values in the training data. This is to allow for
        #    input values in the test set that are outside the range
        #    in the training set.
        # Only do this for linear inputs because categorical inputs
        #    will not have values outside of the input range.
        thismin = matrix[col].min()
        thismax = matrix[col].max()
        if col.split()[0] in linvars:
            delta = 0.5*(thismax-thismin)
        else:
            delta = 0.0
        thisminmax = [thismin-delta,thismax+delta]
        minmaxlist.append(thisminmax)

    # Extract output from the dataframe, and convert to numpy arrays
    passengerid = matrix.index
    output = matrix['Output'].values    # .values returns array
    output = output.reshape(len(output),1)
    matrix = matrix.drop('Output',axis=1)
    matrix = matrix.values

    # Return arrays to calling program
    return (matrix,output,minmaxlist,passengerid)



