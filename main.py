import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import *
from model import*


def save_train_modes(X):
   modes = X.mode().iloc[0]

   modes_dict = modes.to_dict()

   with open('train_modes.pkl', "wb") as f:
        pickle.dump(modes_dict, f) 
        
def main():
    
  df=pd.read_csv('ElecDeviceRatingPrediction.csv')
  y=df['rating'] 
  df=df.drop(columns=["rating"])
  x=df
  
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,shuffle=False)
  save_train_modes(X_train)
  X_train,y_train,top_feature=preprocess(X_train,y_train)
  
  #for Lasso regression
  X_train1=X_train
  X_test1=X_test
  
  X_train=X_train[top_feature] 
 
  lassoRegression(X_train1,y_train)
  ridgeRegression(X_train,y_train)
  MultipleLinearRegression(X_train,y_train)
  PolynomialRegression(X_train,y_train,2)
  

  with open('x_test.pkl', 'wb') as f:
      pickle.dump(X_test, f)

  with open('x_test1.pkl', 'wb') as f:
        pickle.dump(X_test1, f)
        
  with open('Y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
      

main()  