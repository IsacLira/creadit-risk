import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
from src.model_training import train_model
from src.utils.data_handler import fetch_model
from ..evaluation import compute_scores


def plot_scores(scores):
    cv_scores = pd.DataFrame(scores)    
    cv_scores = cv_scores.stack().reset_index()\
        .rename(columns={'level_1': 'Model', 0:'Accuracy'})
    cv_scores['Model'] = cv_scores['Model'].map(
                {'mlp': 'MLP', 
                'log_reg': 'Logistic Regression',
                'rf': 'Random Forest',
                'xgboost': 'XGBoost',
                'catboost': 'CatBoost',
                'lightgbm': 'LightGBM'
                }
    )
    im = sns.boxplot(x='Model', y='Accuracy', data=cv_scores)
    plt.xticks(rotation=45)
    im.tick_params(labelsize=10)
    plt.title('Accuracy CV Score', fontsize=14)
    st.pyplot(plt)

def write():
    st.title('Model Monitor')
    last_trial = fetch_model()
    
    st.subheader("Model Info")
    st.markdown(
        """
        There are 5 model being optimized:
        * Random Forest
        * XGBoost
        * CatBoot
        * MLP
        * Logistic Regression
            
        """ 
    )

    st.subheader('Trained Model')    
    st.write("""
        After run the bayesian optmiser over each model, the best one is selected with respect to the
        accuracy score. Below the parameters found for the best model. (You can minimize this info for a better readability)
    """)
    
    if not bool(last_trial):
        st.warning('No model was trained yet.')
    else: 
        st.json(last_trial['model_instance'].get_params())
    st.sidebar.title('Train a new model')

    st.sidebar.info("""
    The models will be optmized again and best model from this 
    trial will replace the previous model in the case of its accuracy be higher.""")
    button_action = st.sidebar.button(label='Train Model')
    if button_action: 
        with st.spinner(f"Training a new model. Please wait..."):
            last_trial = train_model()        
        st.sidebar.success("A new model was trained succefully!")        

    st.subheader("Model Performance")
    st.write("""
    The 10-fold cross validation accuracy is measured for each model. The perfomances are shown in the graph below.
    """)

    if bool(last_trial):
        plot_scores(last_trial['cv-scores'])
    else:
        st.warning('Run the model training first.')
    
    st.subheader('Confusion Matrix')
    st.write(f"""
    In the followin graph, we have the confusion matrix computed for the best model ({last_trial['clf_name'][0]}) on the valition set.
    """)
    partitions = last_trial['partitions']
    x_val, y_val = partitions['val']
    plot_confusion_matrix(last_trial['model_instance'], 
                          x_val, y_val,
                          cmap=plt.cm.Blues,
                          normalize=None)    
    st.pyplot(plt)                    