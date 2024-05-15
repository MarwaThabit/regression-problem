import pickle
import numpy as np
import pandas as pd
from  preprocessing import extract,extract_digit
from sklearn import metrics
from preprocessing import extract_digit

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    

def label_encoder_for_features(X):
    with open('feature_encoder.pkl', 'rb') as f:
        feature_encoder = pickle.load(f)
        
    for column, info in feature_encoder.items():
        lbl = info['encoder']
        transformed_items = []
        
        for item in X[column]:
            if item in lbl.classes_:
                transformed_item = lbl.transform([item])[0]
            else:
                 transformed_item = max(lbl.transform(lbl.classes_)) + 1
            transformed_items.append(transformed_item)
        
        X[column] = transformed_items
    return X

def normalize_with_scaler(X):
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    
    X_normalized = scaler.transform(X)
    X_normalized = pd.DataFrame(X_normalized, columns=X.columns)
    return X_normalized

def skewness(X):
    with open('skewness_params.pkl', 'rb') as skew_file:
        skewness_params = pickle.load(skew_file)
    for col in X.columns:
        skewness = skewness_params.get(col, 0)
        if skewness > 0.5:
            X[col] = np.log1p(X[col])
        elif skewness < -0.5:
            X[col] = X[col] ** 2
    return X

def OneHotEncoding(X):
    with open('encoded_columns.pkl', 'rb') as f:
        encoder_info = pickle.load(f)
        
    columns=["brand","processor_brand","processor_name","processor_gnrtn","ram_type","weight"]
    X_encoded= pd.get_dummies(X, columns=columns)

    for col in encoder_info:
        if col not in X_encoded.columns:
            X_encoded[col] = 0
            
    new_columns = []
    for col in X_encoded.columns:
        if col not in X.columns:
            new_columns.append(col)
      
    if set(encoder_info) != set(new_columns):
        for col in new_columns:
           if col not in encoder_info:
              X_encoded.drop(columns=[col], inplace=True)
              
    desired_order=['ram_gb', 'ssd', 'hdd', 'os', 'graphic_card_gb', 'warranty',
       'Touchscreen', 'msoffice', 'Price', 'Number of Ratings',
       'Number of Reviews','brand_APPLE', 'brand_ASUS', 'brand_Avita',
       'brand_DELL', 'brand_HP', 'brand_Lenovo', 'brand_MSI', 'brand_acer',
       'processor_brand_AMD', 'processor_brand_Intel', 'processor_brand_M1',
       'processor_name_Celeron Dual', 'processor_name_Core i3',
       'processor_name_Core i5', 'processor_name_Core i7',
       'processor_name_Core i9', 'processor_name_M1','processor_name_Pentium Quad','processor_name_Ryzen 3',
       'processor_name_Ryzen 5', 'processor_name_Ryzen 7',
       'processor_name_Ryzen 9', 'processor_gnrtn_10th',
       'processor_gnrtn_11th', 'processor_gnrtn_12th','processor_gnrtn_4th',
       'processor_gnrtn_7th', 'processor_gnrtn_8th', 'processor_gnrtn_9th',
       'processor_gnrtn_Not Available', 'ram_type_DDR3', 'ram_type_DDR4',
       'ram_type_DDR5', 'ram_type_LPDDR3', 'ram_type_LPDDR4',
       'ram_type_LPDDR4X', 'weight_Casual', 'weight_Gaming',
       'weight_ThinNlight']        
    X_encoded=X_encoded[desired_order]
    return X_encoded


def replace_outliers_with_mean(X):
    with open('outlier_parameters.pkl', 'rb') as params_file:
        outlier_params = pickle.load(params_file)
    for column, params in outlier_params.items():
        mean_value = params['mean']
        lower_bound = params['lower_bound']
        upper_bound = params['upper_bound']
        X.loc[X[column] < lower_bound, column] = mean_value
        X.loc[X[column] > upper_bound, column] = mean_value
    return X


def feature_selector(X):
    with open('selected_features.pkl', 'rb') as f:
        selected_features = pickle.load(f)
    X = X[selected_features]
    return X

def fill_missing_with_mode(X):
    with open('train_modes.pkl', "rb") as f:
        modes_dict = pickle.load(f)
        
    missing_mask = X.isna()
    for col in X.columns:
            X[col] = np.where(missing_mask[col], modes_dict[col], X[col])
    return X

def preprocessing(X,y):
    # X.iloc[1, 0] = np.nan
    # print(X.isna().sum())
    X=fill_missing_with_mode(X)
   
    # print(X.isna().sum())
    
    # X.iloc[1, 2] = 'marwa'
    # X.iloc[3, 2] = 'marwan'
    
    X = label_encoder_for_features(X)
 
    X = OneHotEncoding(X)
 
    if not(isinstance(y, list)):
        y=extract_digit(y)
    
   
    if(X['warranty']=='No warranty').any():
      X['warranty'] = X['warranty'].replace("No warranty","0")
      
    digit_columns = ['warranty','ssd', 'hdd', 'graphic_card_gb', 'ram_gb']
    for col in digit_columns:
       if col in X.columns:
            X[col] = extract_digit(X[col])
    X['os'] = extract(X['os'])

    X =skewness(X)
  
    X = normalize_with_scaler(X)
  
    X= replace_outliers_with_mean(X)
        
    return X ,y

def evaluate_models(X,Y,path,name):
    loaded_model = load_model(path)   
    prediction = loaded_model.predict(X)
    mse_test = metrics.mean_squared_error(Y, prediction)
    r2_test = metrics.r2_score(Y, prediction)
    
    length = len(name) + len("Regression")+4
    margin = (50 - length) // 2
    print(" " * margin + name + "Regression")
    print(" "*(margin-3),'=' * length)
    print('MSE for test dataset :', mse_test,'\n')
    print('R2_Score for test dataset : '+str( round(r2_test*100,2))+'%')
    print('=' * 50)

def Polynomial_model(X,Y):
    with open('polynomial_regression_model.pkl', 'rb') as f:
        poly_model = pickle.load(f)
          
    with open('poly_features.pkl', 'rb') as features_file:
        poly_features = pickle.load(features_file)
    
    X = poly_features.transform(X)
    prediction = poly_model.predict(X)
    mse_test = metrics.mean_squared_error(Y, prediction)
    r2_test = metrics.r2_score(Y, prediction)
    
    length = len("Polynomial Regression")+4
    margin = (50 - length) // 2
    print(" " * margin +"Polynomial Regression")
    print(" "*(margin-3),'=' * length)
    print('MSE for test dataset :', mse_test,'\n')
    print('R2_Score for test dataset : '+str( round(r2_test*100,2))+'%')
    print('=' * 50)

def predict(x,Y,x2):
    X,Y=preprocessing(x,Y)
    X2,Y=preprocessing(x2,Y)
    
    X=feature_selector(X)
   
    Polynomial_model(X,Y)
    evaluate_models(X,Y,'multiple_linear_regression_model.pkl' ,'Multiple Linear')
    evaluate_models(X2,Y,'lasso_regression_model.pkl','Lasso') 
    evaluate_models(X,Y,'ridge_regression_model.pkl', 'Ridge') 

with open('x_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
    
with open('x_test1.pkl', 'rb') as f:
    X_test1 = pickle.load(f)

with open('Y_test.pkl', 'rb') as f:
    Y_test = pickle.load(f)  


predict(X_test,Y_test,X_test1)

##################<for exam>##################
# test_data = pd.read_csv("Unseen_data.csv") 
# y=test_data['rating']
# x =test_data.drop(columns=["rating"])

# predict(x,y)
