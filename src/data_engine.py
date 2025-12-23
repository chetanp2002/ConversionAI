import numpy as np
import pandas as pd

class DataGenerator:
    """
    Simulates e-commerce customer data with hidden causal patterns.
    """
    def __init__(self, n_samples=5000):
        self.n = n_samples
    
    def get_data(self):
        np.random.seed(42)
        
        # 1. Create Features (RFM - Recency, Frequency, Monetary + Demographics)
        data = {
            'Recency': np.random.randint(1, 365, self.n),      # Days since last visit
            'Frequency': np.random.randint(1, 20, self.n),     # Visits per year
            'Amount': np.random.normal(100, 30, self.n),       # Avg spend ($)
            'Age': np.random.randint(18, 70, self.n),
            'Income': np.random.normal(50000, 15000, self.n)
        }
        df = pd.DataFrame(data)
        
        # 2. Hidden Logic (Segments)
        # Rule: Young people with low income are 'Persuadable' (Need discount)
        # Rule: High Frequency users are 'Sure Things' (Don't need discount)
        conditions = [
            (df['Frequency'] > 15),                             # Loyal -> Sure Thing
            (df['Recency'] > 300),                              # Dormant -> Lost Cause
            (df['Age'] < 30) & (df['Income'] < 60000)           # Young/Broke -> Persuadable
        ]
        choices = ['Sure Thing', 'Lost Cause', 'Persuadable']
        df['Segment'] = np.select(conditions, choices, default='Sleeping Dog')
        
        # 3. Simulate Experiment (A/B Test)
        # 50% people got coupon (Treatment=1), 50% didn't (Treatment=0)
        df['Treatment'] = np.random.binomial(1, 0.5, self.n)
        
        # 4. Simulate Outcome (Purchase) based on Segment Logic
        purchase_probs = []
        for i, row in df.iterrows():
            if row['Segment'] == 'Sure Thing':
                prob = 0.9 # Buys anyway
            elif row['Segment'] == 'Lost Cause':
                prob = 0.01 # Never buys
            elif row['Segment'] == 'Persuadable':
                # BIG difference: 80% if coupon, 10% if no coupon
                prob = 0.8 if row['Treatment'] == 1 else 0.1
            else: # Sleeping Dog
                # Hates coupons: 70% if no coupon, 20% if coupon
                prob = 0.2 if row['Treatment'] == 1 else 0.7
            
            purchase_probs.append(prob)
            
        df['Conversion'] = np.random.binomial(1, purchase_probs)
        return df
    
# obj= DataGenerator()
# print(obj.get_data())