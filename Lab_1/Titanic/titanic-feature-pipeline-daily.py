import os
import modal
from preprocess_titanic import run_preprocess
    
BACKFILL=False
LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def generate_passenger(survived, embarked_c, embarked_q, embarked_s, pclass_1, pclass_2, pclass_3):
    """
    Returns a single a single passenger as a single row in a DataFrame
    """
    import pandas as pd
    import random

       
    df = pd.DataFrame({ "sex": [random.randint(0, 1)],
                       "age": [random.uniform(100, 1)],
                       "sibsp": [random.randint(0, 8)],
                       "parch": [random.randint(0, 6)],
                       "fare": [random.uniform(500, 0)],
                       "embarked_c": [embarked_c],
                       "embarked_q": [embarked_q],
                       "embarked_s": [embarked_s],
                       "pclass_1": [pclass_1],
                       "pclass_2": [pclass_2],
                       "pclass_3": [pclass_3],
                      })
    df['survived'] = survived
    return df


def get_random_passenger():
    """
    Returns a DataFrame containing one random passenger
    """
    import pandas as pd
    import random

    survivedP = generate_passenger(1, 1, 0, 0, 1, 0, 0)
    survivedP2 = generate_passenger(1, 0, 0, 1, 0, 1, 0)
    notSurvivedP = generate_passenger(0, 0, 1, 0, 0, 1, 0)

    # randomly pick one of these 3 and write it to the featurestore
    pick_random = random.uniform(0,3)
    if pick_random >= 2:
        titanic_df = survivedP
        print("Passenger added")
    elif pick_random >= 1:
        titanic_df = survivedP2
        print("Passenger added")
    else:
        titanic_df = notSurvivedP
        print("Passenger added")

    return titanic_df



def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    if BACKFILL == True:
        titanic_df = run_preprocess()
    else:
        titanic_df = get_random_passenger()

    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_modal",
        version=1,
        primary_key=['sex','age','sibsp','parch','fare','embarked_c','embarked_q','embarked_s',
                     'pclass_1','pclass_2','pclass_3',''], 
        description="Titanic dataset")
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
