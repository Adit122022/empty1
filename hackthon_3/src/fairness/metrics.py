import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

class FairnessAnalyzer:
    def __init__(self, data: pd.DataFrame, target_col: str, protected_cols: list):
        self.data = data
        self.target_col = target_col
        self.protected_cols = protected_cols
        
    def calculate_disparate_impact(self, y_pred, sensitive_col, privileged_group, unprivileged_group):
        """
        Calculate Disparate Impact Ratio:
        (P(Y=1 | D=unprivileged) / P(Y=1 | D=privileged))
        """
        df = self.data.copy()
        df['y_pred'] = y_pred
        
        # Calculate selection rate for privileged group
        privileged_indices = df[sensitive_col] == privileged_group
        unprivileged_indices = df[sensitive_col] == unprivileged_group
        
        # Selection rate (Positive prediction rate)
        sr_privileged = df.loc[privileged_indices, 'y_pred'].mean()
        sr_unprivileged = df.loc[unprivileged_indices, 'y_pred'].mean()
        
        if sr_privileged == 0:
            return 0.0 # Avoid division by zero
            
        return sr_unprivileged / sr_privileged

    def calculate_equal_opportunity_difference(self, y_true, y_pred, sensitive_col, privileged_group, unprivileged_group):
        """
        Calculate Equal Opportunity Difference:
        TPR_unprivileged - TPR_privileged
        """
        df = self.data.copy()
        df['y_true'] = y_true
        df['y_pred'] = y_pred
        
        def get_tpr(indices):
            subset = df.loc[indices]
            if len(subset) == 0:
                return 0
            tn, fp, fn, tp = confusion_matrix(subset['y_true'], subset['y_pred'], labels=[0, 1]).ravel()
            return tp / (tp + fn) if (tp + fn) > 0 else 0

        privileged_indices = df[sensitive_col] == privileged_group
        unprivileged_indices = df[sensitive_col] == unprivileged_group
        
        tpr_privileged = get_tpr(privileged_indices)
        tpr_unprivileged = get_tpr(unprivileged_indices)
        
        return tpr_unprivileged - tpr_privileged
    
    def generate_report(self, y_true, y_pred):
        report = {}
        for col in self.protected_cols:
            # Assume privileged group is the majority group for simplicity or hardcoded
            # In a real scenario, this should be configurable
            groups = self.data[col].unique()
            if len(groups) < 2:
                continue
                
            # Heuristic: largest group is privileged
            privileged_group = self.data[col].mode()[0]
            unprivileged_groups = [g for g in groups if g != privileged_group]
            
            for unpriv in unprivileged_groups:
                di = self.calculate_disparate_impact(y_pred, col, privileged_group, unpriv)
                eod = self.calculate_equal_opportunity_difference(y_true, y_pred, col, privileged_group, unpriv)
                
                report[f"{col}_{unpriv}_vs_{privileged_group}"] = {
                    "Disparate Impact": di,
                    "Equal Opportunity Diff": eod
                }
        return report

if __name__ == "__main__":
    # Test
    data = pd.DataFrame({
        'gender': ['Male', 'Male', 'Female', 'Female', 'Male', 'Female'],
        'target': [1, 1, 0, 1, 0, 0]
    })
    analyzer = FairnessAnalyzer(data, 'target', ['gender'])
    y_pred = [1, 1, 0, 0, 0, 0]
    print(analyzer.generate_report(data['target'], y_pred))
