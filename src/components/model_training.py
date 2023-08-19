import os
import sys
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.svm import SVR
from dataclasses import dataclass
from src.utils import save_object
from src.utils import evaluate_model
from src.exception import custom_exception
from src.logger import logging

@dataclass
class modeltrainerconfig:
    modeltrainerconfig=os.path.join('Artifacts','model.pkl')
    
class model_trainer:
    def __init__(self):
        self.modeltrainerpath=modeltrainerconfig()
        
    def model_training(self,train_arr,test_arr):
        try:
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
                )
            logging.info(f'data are seprated')
            
            models={
                "LinearRegression":LinearRegression(),'Ridge':Ridge(),'Lasso':Lasso(),
                'KNeighborsRegressor':KNeighborsRegressor(),'DecisionTreeRegressor':DecisionTreeRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),'XGBRegressor':XGBRegressor(),'ExtraTreesRegressor':ExtraTreesRegressor(),
                "CatBoostRegressor":CatBoostRegressor()
            }
            params={
                "LinearRegression":{},
                'Ridge':{},'Lasso':{},
                'KNeighborsRegressor':{},
                
                'DecisionTreeRegressor': {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "RandomForestRegressor":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
               
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                'ExtraTreesRegressor':{},
                "CatBoostRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                
            }
            evaluatemodels:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,param=params)
            
            best_model_score=max(sorted(evaluatemodels.values()))
            logging.info(f' best model score {best_model_score} ')
            best_model_name=list(evaluatemodels.keys())[list(evaluatemodels.values()).index(best_model_score)]
            
            best_model=models[best_model_name]
            logging.info(f'best model is {best_model}')
            
            if best_model_score < 0.6 :
                raise custom_exception('no best model found')
            logging.info('best model found for training and testing data')
                
            
            predicted_values=best_model.predict(x_test)
            
            r_square=r2_score(y_test,predicted_values)
            
            save_object(
                file_path=self.modeltrainerpath.modeltrainerconfig,
                obj=best_model
            )
            
            return r_square
        

        except Exception as e:
            raise custom_exception(e,sys)