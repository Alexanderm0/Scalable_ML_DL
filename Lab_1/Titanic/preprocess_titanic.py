from random import randint

import pandas as pd

def run_preprocess():
    # Read data
    titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")

    # find NaN valued columns
    hasNaNs = []
    for column in titanic_df.columns:
      print(column + ": " + str(sum(titanic_df[column].isna())) + ", type: "+ str(titanic_df[column].dtype))
      if sum(titanic_df[column].isna()) > 0:
          hasNaNs.append(column)

    # process age column, fill nans
    col = "Age"
    min_age = int(min(titanic_df[col]))
    max_age = int(max(titanic_df[col]))
    nanSpots = titanic_df[col].isna()
    temp = titanic_df[col]
    for i, nan in enumerate(nanSpots):
        if nan == True:
            temp[i] = randint(min_age, max_age)
    titanic_df[col] = temp

    # Fill embarked values
    titanic_df["Embarked"] = titanic_df["Embarked"].fillna('S')

    # Categorical variable encoding
    # binarize gender
    titanic_df['Sex'] = titanic_df['Sex'].map({'male': 1, 'female': 0})

    # one hot for embark and pclass
    cols = ['Embarked', 'Pclass']
    for col in cols:
        onehot = pd.get_dummies(titanic_df[col], prefix=col, dtype='int64')
        titanic_df = titanic_df.drop(col, axis=1)
        titanic_df = titanic_df.join(onehot)

    # Drop columns lacking predictive power
    drop_cols = ['Name', 'PassengerId', 'Ticket', 'Cabin']
    titanic_df = titanic_df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)
    titanic_df = titanic_df.rename(columns=str.lower)

    return titanic_df