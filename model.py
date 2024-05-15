from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso,Ridge
import pickle

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
 
def MultipleLinearRegression(x_train,y_train):
    cls = linear_model.LinearRegression()
    # print(x_train.shape)
    # print(y_train.shape)
    cls.fit(x_train,y_train)
    cls.predict(x_train)
    save_model(cls, 'multiple_linear_regression_model.pkl')
    

def lassoRegression(X_train,y_train):
    lasso_model = Lasso(alpha=0.0001) 
    # print(X_train.shape)
    # print(y_train.shape)
    lasso_model.fit(X_train, y_train)
    save_model(lasso_model, 'lasso_regression_model.pkl')
    lasso_model.predict(X_train)
 

def ridgeRegression(X_train,y_train):
    ridge_model = Ridge(alpha=0.001) 
    # print(X_train.shape)
    # print(y_train.shape)
    ridge_model.fit(X_train, y_train)
    save_model(ridge_model, 'ridge_regression_model.pkl')
    ridge_model.predict(X_train)
   


def PolynomialRegression(X_train,y_train,degree):
    poly_features = PolynomialFeatures(degree)
    # print(X_train.shape)
    # print(y_train.shape)
    X_train_poly = poly_features.fit_transform(X_train)
    poly_model = linear_model.LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    save_model(poly_model, 'polynomial_regression_model.pkl')
    with open('poly_features.pkl', 'wb') as features_file:
        pickle.dump(poly_features, features_file)
    poly_model.predict(X_train_poly)   
    