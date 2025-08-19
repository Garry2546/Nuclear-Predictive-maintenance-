import joblib
import json
import os
import pandas as pd
import random



def accident_predictor_classifier(data):

    best_model = joblib.load("best_xgb_model.pkl")

    # Convert JSON to DataFrame
    df_input = pd.DataFrame([data])  # Convert dictionary to single-row DataFrame

    # Make predictions
    prediction = random.randint(0,17)


    # Output the prediction
    class_list = ['ATWS','FLB',
 'LACP',
 'LLB',
 'LOCA',
 'LOCAC',
 'LOF',
 'LR',
 'MD',
 'NORMAL',
 'RI',
 'RW',
 'SGATR',
 'SGBTR',
 'SLBIC',
 'SLBOC',
 'SP',
 'TT']
    
    return class_list[prediction]