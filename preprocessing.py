import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from visualization import *



def extract_digit(column):
    extracted = []
    for string in column:
        parts = string.split()
        for part in parts:
            if part.isdigit():
                extracted.append(int(part))
                break
    return extracted


def extract(column):
    extracted = []
    for string in column:
        parts = string.split("-")
        for part in parts:
            if part.isdigit():
                extracted.append(int(part))
                break
    return extracted

def Feature_Encoder(X,cols):
    feature_encoder = {}
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
        feature_encoder[c] = {
            'encoder': lbl,
            'classes': lbl.classes_.tolist() 
        }
    with open( 'feature_encoder.pkl', 'wb') as f:
        pickle.dump(feature_encoder, f)
    return X

def oneHotEncoding(X,cols):
    X_encoded= pd.get_dummies(X, columns=cols)
    new_columns = []
    for col in X_encoded.columns:
        if col not in X.columns:
            new_columns.append(col)      
    with open('encoded_columns.pkl', 'wb') as f:
        pickle.dump(new_columns, f)
    return X_encoded

def count_outliers(df, columns):
    outlier_counts = {}
    for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            outlier_counts[column] = outliers.shape[0]
    for column, count in outlier_counts.items():
       print(column +' : '+ str(count) +" outliers")

def drop_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

def replace_outliers_With_LU(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df.loc[df[column] < lower_bound, column] = lower_bound
        df.loc[df[column] > upper_bound, column] = upper_bound
    return df

def replace_outliers_with_mean(X, columns):
    parameters = {}
    for column in columns:
        Q1_train = X[column].quantile(0.25)
        Q3_train = X[column].quantile(0.75)
        IQR_train = Q3_train - Q1_train
        lower_bound_train = Q1_train - 1.5 * IQR_train
        upper_bound_train = Q3_train + 1.5 * IQR_train
        mean_value_train = X[column].mean()
        
        parameters[column] = {
            'lower_bound': lower_bound_train,
            'upper_bound': upper_bound_train,
            'mean': mean_value_train
        }
        with open('outlier_parameters.pkl', 'wb') as params_file:
          pickle.dump(parameters, params_file)
        
        X.loc[X[column] < lower_bound_train, column] = mean_value_train
        X.loc[X[column] > upper_bound_train, column] = mean_value_train
   
    return X

def handle_duplicates(df):
    duplicate_mask = df.duplicated()
    num_duplicates = duplicate_mask.sum()
    # print('Number of duplicate rows : ' + str(num_duplicates))
    df_cleaned = df.drop_duplicates()
    return df_cleaned
    
def standardize_data(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    return df_scaled

def mean_normalize_data(X):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    return X_train_scaled


def Skewness(X):
    skewness_params = {}
    for col in X.columns:
        skewness = X[col].skew()
        skewness_params[col] = skewness
        if skewness > 0.5:
            X[col] = np.log1p(X[col])
        elif skewness < -0.5:
            X[col] = X[col] ** 2
    with open('skewness_params.pkl', 'wb') as skew_file:
        pickle.dump(skewness_params, skew_file)    
    return X




def preprocess(X,y):
  # print(df.isnull().sum())
  cols=('msoffice','Touchscreen')
  X=Feature_Encoder(X,cols)
  
  columns=["brand","processor_brand","processor_name","processor_gnrtn","ram_type","weight"]
  X=oneHotEncoding(X,columns)

  X['warranty'] = X['warranty'].replace("No warranty","0")
  
  digit_columns = ['warranty', 'rating', 'ssd', 'hdd', 'graphic_card_gb', 'ram_gb']
  for col in digit_columns:
     if col=='rating':
         y =extract_digit(y)
     if col in X.columns:
         X[col] = extract_digit(X[col])
         
  X['os'] = extract(X['os'])
  
  X=Skewness(X)
  X=mean_normalize_data(X)

  X= replace_outliers_with_mean(X,X.columns)
  
  y = pd.Series(y)
  corr = X.corrwith(y)
  
  top_feature = corr[abs(corr) > 0.15].index.tolist()
  
  with open('selected_features.pkl', 'wb') as f:
      pickle.dump(top_feature, f) 
      
  return X,y,top_feature