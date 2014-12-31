# File to hold auxiliary functions for performing logistic regression with the Titanic data set.


import numpy as np
import pandas as pd


def cat2indicator(df, columns):
    """
Convert columns of categorical variables to multiple columns of
indicator variables.
"""

    # Create new dataframe to hold data
    df2 = df.copy()

    # Process each column included in columns argument
    for column in columns:

        # Make new column from each unique value in the categorical column
        for value in df2[column].unique():
            colname = column+'_'+str(value)
            newcol = np.zeros(len(df2))
            newcol[np.array(df[column]==value)] = 1
            df2[colname] = newcol

        # Drop original column of categorical variables
        df2.drop(column,axis=1,inplace=True)

    # Return dataframe to calling program
    return df2



def nametitles(df):
    """
Get title information from the name column in the dataframe.
Add indicator column for each title.
"""

    # Create new dataframe to hold data
    df2 = df.copy()

    # Get list of titles from the name list
    titles = []
    for name in df2.Name:
        thistitle = name.split('.')[0].split(' ')[-1]
        titles.append(thistitle)

    # Add this column to the dataframe
    df2['Title'] = titles

    ## For each title, add an indicator variable marking it as a title
    #for title in titles:
    #    barray = []
    #    for name in df2.Name:
    #        if title+'.' in name: barray.append(True)
    #        else: barray.append(False)
    #    newcol = 'Title_'+title
    #    df2[newcol] = barray
    #
    ## Drop name list from the dataframe
    #df2.drop('name',axis=1,inplace=True)

    # Return dataframe to calling program
    return df2



def add_interactions(df, X, variables):
    """
df - dataframe in which to place interaction terms
variables - list of names from which to create interaction terms
"""

    # Get dummy variables for each category
    #vardict = {}
    #for var in variables:
    #    # Get dummy variables for this category
    #    vardict[var] = pd.get_dummies(df[var])

    # Get dummy variables for each category
    vardict = {}
    for var in variables:
        thesecols = []
        for col in X:
            if (var==col[:len(var)]) & ('*' not in col):
                thesecols.append( col )
        vardict[var] = thesecols

    # Add interaction between all items in the variable dictionary
    if len(variables)==2:
        for value1 in vardict.values()[0]:
            for value2 in vardict.values()[1]:
                newname = value1+'_*_'+value2
                X[newname] = X[value1]*X[value2]


#    # Calculate ineraction terms between all items in the dictionary
#    if len(variables)==2:
#        for column1 in vardict[variables[0]].columns:
#            for column2 in vardict[variables[1]].columns:
#                newname = str(column1)+'_*_'+str(column2)
#                X[newname] = vardict[variables[0]][column1]*vardict[variables[1]][column2]
#        colname = str(vardict[variables[0]].columns[0]) + '_*_' + str(vardict[variables[1]].columns[0])
#
#    if len(variables)==3:
#        for column1 in vardict[variables[0]].columns:
#            for column2 in vardict[variables[1]].columns:
#                for column3 in vardict[variables[2]].columns:
#                    newname = str(column1)+'_*_'+str(column2)+'_*_'+str(column3)
#                    X[newname] = vardict[variables[0]][column1]*vardict[variables[1]][column2]*vardict[variables[2]][column2]
#        colname = str(vardict[variables[0]].columns[0]) + '_*_' + str(vardict[variables[1]].columns[0]) + '_*_' + str(vardict[variables[2]].columns[0])
#        
#
#    # Drop one column from those above
    #X.drop(colname,axis=1,inplace=True)

    # Return dataframe to calling program
    return X



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



def make_matrix(df,matchcols=None):
    """
Function to construct the matrix for the linear regression.
matchcols - column names to include in the new matrix, regardless
            of whether the new matrix may be singular.
"""

    # Fill embarcation location nan with a string
    df.Embarked = df.Embarked.fillna('nan')

    # Create name category from titles in the name column
    df = nametitles(df)

    # Define new dataframe to hold the matrix
    X = pd.DataFrame(index=df.index)

    # Add categorical variables to the matrix
    X = add_dummy(X, df.Embarked, 'Embarked', drop=True)
    X = add_dummy(X, df.Sex, 'Sex', drop=True)
    X = add_dummy(X, df.Pclass, 'Pclass', drop=True)
    X = add_dummy(X, df.Title, 'Title', drop=True)
    goodtitles = ['Master']
    #goodtitles = ['Mr','Mrs','Miss','Master']#,'Rev', 'Dr']#,
                  #'Jonkheer', 'Countess','Lady','Major','Capt','Sir']
    for column in X.columns:
        if column[:5]=='Title':
            if column[6:] not in goodtitles:
                X.drop(column,axis=1,inplace=True)

    # Make categorical variables from Fare and Age
    #df['Fare'][df.Fare.isnull()] = df.Fare.notnull().mean()
    #df.Fare = np.ceil( df.Fare / 10. ) * 10.
    #df.Fare[df.Fare>50] = 50.
    #X = add_dummy(X, df.Fare, 'Fare', drop=True)
    #df['Age'][df.Age.isnull()] = df.Age.notnull().mean()
    #df.Age = np.ceil( df.Age / 10. ) * 10.
    #df.Age[df.Age>60] = 60.
    #X = add_dummy(X, df.Age, 'Age', drop=True)

    # Add continuous variables to the dataframe
    #X['Fare'] = (df.Fare - df.Fare.mean()) / df.Fare.std()
    #X['Fare'][X.Fare.isnull()] = X.Fare.notnull().mean()
    #X['Fare2'] = X.Fare**2
    #X['Age'] = (df.Age - df.Age.mean()) / df.Age.std()
    #X['Age'][X.Age.isnull()] = X.Age.notnull().mean()
    #X['Age2'] = X.Age**2
    

    # Add interaction terms
    X = add_interactions(df, X, ['Pclass','Sex'])
    X = add_interactions(df, X, ['Sex','Embarked'])
    #X = add_interactions(df, X, ['Embarked','Pclass'] )
    #X = add_interactions(df, X, ['Age','Sex'])
    #X = add_interactions(df, X, ['Age','Pclass'])

    # Remove any columns that are a single constant or all zeros
    if matchcols is None:
        for col in X.columns:
            if (np.std(X[col])==0):
                X.drop(col,axis=1,inplace=True)
    else:
        # Remove columns not in matchcols
        for col in X.columns:
            if col not in matchcols:
                X.drop(col,axis=1,inplace=True)
        # Add matchcols not in columns
        for col in matchcols:
            if col not in X.columns:
                X[col] = np.zeros(len(X))
        # Order columns to match that of the input columns
        X = X.reindex_axis(matchcols, axis=1)

    # Add column of ones as a constant
    X.insert(0,'const',np.ones(len(X)))   #X = sm.add_constant(X,prepend=True)

    # Return dataframe for regression to calling program
    return X


def score(obs, prediction):
    """
Calculate score of the prediction.
"""

    return float(sum(obs==prediction)) / len(obs)
