import pandas as pd
import xgboost as xgb
import numpy as np

class CausalBrain:
    """
    The T-Learner Architecture:
    Trains two separate models to estimate Causal Lift.
    """
    def __init__(self):
        # Professional settings for XGBoost to prevent warnings
        self.model_control = xgb.XGBClassifier(eval_metric='logloss', enable_categorical=False)
        self.model_treatment = xgb.XGBClassifier(eval_metric='logloss', enable_categorical=False)
        
    def train(self, df, feature_cols):
        """
        Splits data into Control (No Coupon) and Treatment (Coupon) groups.
        """
        # Group 0: Did not get coupon
        X_ctrl = df[df['Treatment'] == 0][feature_cols]
        y_ctrl = df[df['Treatment'] == 0]['Conversion']
        
        # Group 1: Got coupon
        X_trtm = df[df['Treatment'] == 1][feature_cols]
        y_trtm = df[df['Treatment'] == 1]['Conversion']
        
        # Train both brains
        self.model_control.fit(X_ctrl, y_ctrl)
        self.model_treatment.fit(X_trtm, y_trtm)
        
    def get_uplift(self, df, feature_cols):
        """
        Calculates Uplift = P(Buy|Coupon) - P(Buy|No Coupon)
        """
        X = df[feature_cols]
        
        # Probability if we DO NOT give coupon
        prob_no_coupon = self.model_control.predict_proba(X)[:, 1]
        
        # Probability if we GIVE coupon
        prob_coupon = self.model_treatment.predict_proba(X)[:, 1]
        
        # The Lift
        uplift = prob_coupon - prob_no_coupon
        return uplift