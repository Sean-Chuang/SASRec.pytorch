#!/usr/bin/env python3
import os
import pandas as pd
import pickle as pkl
from pyhive import presto

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DEST = os.path.join(CURRENT_PATH, "..", "data")

def query_train_data(label, dt, recency=30, sn_user_only=False):
    cursor = presto.connect('presto.smartnews.internal',8081).cursor()
    param = {
        "label": label,
        "dt": dt,
        "recency": recency,
        "sn_user_condition": "and length(user_id) = 42" if sn_user_only else ""
    }

    with open(f"train_data.sql", 'r') as file:
        sql = f"{file.read()}".format(**param)
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
    return item2id


def query_user_history(label, dt, item2id, maxlen=20):
    cursor = presto.connect('presto.smartnews.internal',8081).cursor()
    param = {
        "label": label,
        "dt": dt,
        "history_max": maxlen,
        "recency": 15
    }

    with open(f"user_history.sql", 'r') as file:
        sql = f"{file.read()}".format(**param)
        print(sql)

    cursor.execute(sql)
    column_names = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(cursor.fetchall(), columns=column_names)

    df['rtg_item'] = df['rtg_item'].apply(lambda x: ','.join([str(item2id.get(e, 0)) for e in x]))
    df = df.reset_index(drop=True)
    print(df.head(), '\n------------------\n')

    df.to_csv(f'{DATA_DEST}/{label}_user_history.txt', index=False, header=False, sep='\t')


if __name__ == '__main__':
    label = "adidas"
    dt = "2021-07-02"
    train_recency = 30
    maxlen = 20
    item2id = query_train_data(label, dt, train_recency, False)
    query_user_history(label, dt, item2id, maxlen)
