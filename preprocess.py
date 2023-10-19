


import pandas as pd


class DataPreprocessor:
    def __init__(self, columns_to_encode : list, columns_to_drop : list):
        self.columns_to_encode = columns_to_encode
        self.columns_to_drop = columns_to_drop 
        self.data = None
    
    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)
        return self
    
    def clean(self):
        if self.data is not None:
            self.data = self.data.drop(columns= self.columns_to_drop) 
            self.data = self.data.replace("?", pd.NA)
            self.data = self.data.dropna()
            self.data = pd.get_dummies(self.data, columns = self.columns_to_encode, drop_first=True)
        return self
    
    def get_data(self):
        return self.data

#create a class that has the method we will use to select columns that correlate with the target 
    
class ColumnSelector:
    def __init__(self, data : pd.DataFrame, threshold : float, corr_column : str):
        self.data = data
        self.threshold = threshold
        self.corr_column = corr_column
        
    def selector(self):
        selected_columns = self.data.corr()[self.corr_column].abs() > self.threshold
        selected_columns = selected_columns.index[selected_columns].tolist()
        return self.data[selected_columns]
