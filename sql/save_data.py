#!/usr/bin/env python3
import os
import pandas as pd
from pyhive import presto

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DEST = os.path.join(CURRENT_PATH, "..", "data")

def run(label):
    cursor = presto.connect('presto.smartnews.internal',8081).cursor()
    with open(f"{label}.sql", 'r') as file:
        sql = file.read()
        print(sql)
    cursor.execute(sql)
    column_names = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(cursor.fetchall(), columns=column_names)
    df.to_csv(f'{DATA_DEST}/{label}.csv', index=False, header=False)

if __name__ == '__main__':
    label = "au_pay"
    run(label)
