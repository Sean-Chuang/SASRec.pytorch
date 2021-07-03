#!/usr/bin/env python3
import os
import pandas as pd
import pickle as pkl
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

    uids = sorted(df['user_id'].unique())
    user2id = dict(zip(uids, range(1, len(uids) + 1)))
    iids = sorted(df['item_id'].unique())
    item2id = dict(zip(iids, range(1, len(iids) + 1)))

    # save in dictionary
    vocab_path = "vocab.pkl"
    pkl.dump((user2id, item2id), open(vocab_path, 'wb'))
    print(f"Users size: {len(user2id)}")
    print(f"Items size: {len(item2id)}")

    df['user_id'] = df['user_id'].apply(lambda x: user2id.get(x, 0))
    df['item_id'] = df['item_id'].apply(lambda x: item2id.get(x, 0))
    df = df.reset_index(drop=True)
    print(df.head(), '\n------------------\n')

    df.to_csv(f'{DATA_DEST}/{label}.txt', index=False, header=False, sep=' ')

if __name__ == '__main__':
    label = "au_pay"
    run(label)
