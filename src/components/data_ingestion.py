import os
import sys
import pandas as pd
from src.exception import custom_exception
from src.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import data_transform_obj
logging.info('enter the data ingestion part')
from src.components.model_training import model_trainer

@dataclass
class dataingestionconfig:
    train_data_path=os.path.join('Artifacts','train.csv')
    test_data_path=os.path.join('Artifacts','test.csv')
    raw_data_path=os.path.join('Artifacts','data.csv')
    
class dataingestion:
    def __init__(self):
        self.dataingest=dataingestionconfig()
        
    def dataingestion_paths(self):
        logging.info('enter the data ingestion part 2')
        df=pd.read_csv(r"notebook\data\students\StudentsPerformance.csv")
        logging.info('read the data sucessfully')
        os.makedirs(os.path.dirname(self.dataingest.train_data_path),exist_ok=True)
        try:
            df.to_csv(self.dataingest.raw_data_path,header=True,index=False)
            train_data,test_data=train_test_split(df,test_size=0.3,random_state=42)
            
            train_data.to_csv(self.dataingest.train_data_path,header=True,index=False)
            test_data.to_csv(self.dataingest.test_data_path,header=True,index=False)
            logging.info('data ingestion is completed')
            
            return(
                self.dataingest.train_data_path,
                self.dataingest.test_data_path
            )
            
            
        except Exception as e:
            raise custom_exception(e,sys)
        
if __name__=='__main__':
    obj=dataingestion()
    train_path,test_path=obj.dataingestion_paths()
    data_trans=data_transform_obj()
    train_array,test_array,_=data_trans.data_transformer(train_path,test_path)
    model_train=model_trainer()
    print(model_train.model_training(train_arr=train_array,test_arr=test_array))   
    