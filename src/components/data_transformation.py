import os 
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.logger import logging
from src.exception import custom_exception
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.utils import save_object


@dataclass
class data_transform_config:
    data_transform_path=os.path.join('Artifacts','preprocessing.pkl')
    
class data_transform_obj:
    def __init__(self):
        self.data_transformation=data_transform_config()
        
    def data_transform(self):
        try:
            num_features=['reading score', 'writing score']
            cat_features=[
                 "gender",
                 "race/ethnicity",
                 "parental level of education",
                 "lunch",
                 "test preparation course",
            ]
            logging.info('pipeline process starts')
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    ('preprocessing',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            logging.info('pipeline process ends')
            logging.info('preprocessing data')
            preprocessing=ColumnTransformer([
                ('num_pipeline',num_pipeline,num_features),
                ('cat_pipeline',cat_pipeline,cat_features),
            ]
            )
            return preprocessing
        except Exception as e:
            raise custom_exception(e,sys)

    def data_transformer(self,train_path,test_path):
        try:
            
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info(f'data read sucessfully ')
            preprocess_obj=self.data_transform()
            target_feature='math score'
            num_columns=['reading score', 'writing score']
            cat_features=['gender','race/ethnicity','parental level of education',
                          'test preparation course',
                          'lunch']
            
            input_features_train=train_df.drop(target_feature,axis=1)
            target_feature_train=train_df[target_feature]
            
            logging.info(f'input feature{train_df.columns}')
            
            input_features_test=test_df.drop(target_feature,axis=1)
            target_feature_test=test_df[target_feature]
            
            input_features_train_arr=preprocess_obj.fit_transform(input_features_train)
            input_features_test_arr=preprocess_obj.transform(input_features_test)
            
            train_arr=np.c_[
                input_features_train_arr,np.array(target_feature_train)
            ]
            test_arr=np.c_[input_features_test_arr,np.array(target_feature_test)]
            save_object(
                file_path=self.data_transformation.data_transform_path,
                obj=preprocess_obj
            )
            logging.info(f'saved the preprocessing object{input_features_train_arr}')
            
            return (
                train_arr,
                test_arr,
                self.data_transformation.data_transform_path
            )
            
        except Exception as e:
            raise custom_exception(e,sys)