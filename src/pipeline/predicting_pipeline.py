import sys
import pickle
from src.utils import load_object
from src.exception import custom_exception
import pandas as pd
import os
from src.logger import logging
class predict:
    def __init__(self):
        pass
    def predictdata(self,features):
        try:
            model_path=os.path.join('Artifacts','model.pkl')
            preprocessor_path=os.path.join('Artifacts','preprocessing.pkl')
            model=load_object(file_path=model_path)
            logging.info(f'features are {features}')
            prepro=pickle.load(open('Artifacts/preprocessing.pkl','rb'))
            logging.info(f'preprocess starts  {prepro} ')
            preprocess_data=prepro.transform(features)
            logging.info(f'preprocess data is {preprocess_data}')
            results=model.predict(preprocess_data)
            
            return results
        except Exception as e:
            raise custom_exception(e,sys)

class predict_data:
    
    def __init__ (self,gender:str,
                  race_ethnicity:str,
                  parental_level_of_education,
                  lunch:str,
                  test_preparation_course: str,
                  reading_score: int,
                  writing_score: int):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score
     
    def data_as_dataframe(self):
        try:
            column={
                'gender':[self.gender],
                
                'race/ethnicity':[self.race_ethnicity],
                
                'parental level of education':[self.parental_level_of_education],
                
                'lunch':[self.lunch],

                'test preparation course':[self.test_preparation_course],

                'reading score':[self.reading_score],

                'writing score':[self.writing_score]
                
            }
            
            
            return pd.DataFrame(column)
             
        except Exception as e:
            raise custom_exception(e,sys)
        
         
        
        
        
        