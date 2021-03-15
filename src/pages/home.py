import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st
from src.utils.data_handler import fetch_model, fetch_credit_data
from src.db.database import RedisDB

db = RedisDB()


def plot_corr(df, key_='pairplot'):
    plot = db.get_value(key_)
    if plot:
        print('Getting graph from DB')
        return pickle.loads(plot)
    print('Generating graph')
    sns.set_context("paper", rc={"axes.labelsize": 16})
    gfg = sns.pairplot(df, hue='label')
    db.set_value(key_, pickle.dumps(gfg))
    return gfg


def write():
    st.title('Model prediction for Credit Risk')

    st.header("Data Analysis")
    st.write(""" 
    In order to model the credit risk, machine learning algorithms are applied to learn partterns from 
    the data. The dataset used was the [UCI German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)).
    It has 20 categorical and numerical features useful to train the models. 
    """)
    st.subheader("Data Description")
    st.markdown(""" 
    The features follow the description below:

    1. Status of existing checking account, in Deutsche Mark.
    2. Duration in months
    3. Credit history (credits taken, paid back duly, delays, critical accounts)
    4. Purpose of the credit (car, television,…)
    5. Credit amount
    6. Status of savings account/bonds, in Deutsche Mark.
    7. Present employment, in number of years.
    8. Installment rate in percentage of disposable income
    9. Personal status (married, single,…) and sex
    10. Other debtors / guarantors
    11. Present residence since X years
    12. Property (e.g. real estate)
    13. Age in years
    14. Other installment plans (banks, stores)
    15. Housing (rent, own,…)
    16. Number of existing credits at this bank
    17. Job
    18. Number of people being liable to provide maintenance for
    19. Telephone (yes,no)
    20. Foreign worker (yes,no)
    """)
    
    df = fetch_credit_data()
    st.write(""" 
       In the table below we see all samples from the dataset. 
       Each sample with its repective true label.
    """)

    st.dataframe(data=df)

    # st.subheader("Features Correlation")
    # a = plot_corr(df)
    # st.pyplot(a)

    st.subheader("Predict Random Sample")
    st.write("""
        In this section the model is loaded to predict a random sample from the credit data.        
    """)
    button_action = st.button(label='Predict Sample')

    last_trial = fetch_model()

    if button_action or 1:
        if not last_trial:
            st.text('No model was trained yet.')
        else:
            st.write("Selected Sample:")
            sample = df.sample(1)
            x_sample, y_sample = sample.drop('label', axis=1), sample['label']
            st.dataframe(data=sample)
            fitted_model = last_trial['model_instance']
            prediction = fitted_model.predict(X=x_sample)
            pred_success = prediction[0] == y_sample.iloc[0]
            if pred_success:
                st.success('Model suceed!')
            else:
                st.error('Model got wrong.')
            st.markdown(f"""
            - **Predicted Value**: {prediction[0]}  
            - **Actual Value**: {y_sample.iloc[0]}""")
