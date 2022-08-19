import numpy as np
import pandas as pd
import env
import os

def conn(db):
    '''
    This function uses my info from my env file to create a 
    connection url to access the MySQL CodeUp database.
    '''
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

def get_telco_data():
    '''
    This function checks if the telco data is saved locally. 
    If it is not local, this function reads the telco data from 
    the CodeUp MySQL database and return it in a DataFrame.    '''
    filename = 'telco.csv'
    if os.path.isfile(filename):
        return pd.read_csv(filename).iloc[:,1:]
    else:
        query = '''
            SELECT * FROM customers
            JOIN contract_types 
                USING (contract_type_id)
            JOIN payment_types 
                USING (payment_type_id)
            JOIN internet_service_types 
                USING (internet_service_type_id);  
        '''
        df = pd.read_sql(query, conn('telco_churn')) 
        df.to_csv(filename)
        return df

