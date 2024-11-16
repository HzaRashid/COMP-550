from ast import literal_eval
import pandas as pd
import numpy as np
import os


''' load data '''
data = pd.read_csv(
    filepath_or_buffer=os.path.join(os.path.dirname(__file__), 'pa2data.tsv'),
    converters={'label':literal_eval},
    sep='\t'
    )

train_data = data[data['id'].str.startswith('d001')]
test_data = data[~data['id'].str.startswith('d001')]


if __name__ == "__main__":
    print(len(train_data['label'].apply(tuple).unique()))
    print(len(data['label'].apply(tuple).unique()))
    print(len(data))

    
    print('get rows with unique labels')
    data['label'] = data['label'].apply(tuple)
    unique_by_label = data.drop_duplicates('label')
    print('unique by label, length: ', len(unique_by_label))


    df_all = data.merge(unique_by_label.drop_duplicates(), on=['id', 'lemma', 'context', 'index', 'label'], 
                   how='left', indicator=True)
    df_all = df_all.sort_values('_merge').reset_index(drop=True)
    # print(len(df_all['label'].unique()))
    # print(df_all.reset_index(drop=True))

    train_split = df_all[df_all['_merge'] == 'both']
    # print(len(train_split))
    print(train_split.sort_values(['label']).reset_index(drop=True))
    # print(train_split[train_split['label'].apply(lambda x: len(x) > 1)])


    train_validation = pd.concat([train_split, df_all[df_all['_merge'] == 'left_only'][:377]])

    test_split = df_all[df_all['_merge'] == 'left_only'][377:]

    train_split = train_validation[:1000]
    validation_split = train_validation[:200]
    
    print(train_validation)

    print()
    print()
    print(train_split)
    print(validation_split)
    print(test_split)



    # print(train_validation.duplicated())
    # ids = train_validation['id']

    # print(df_all[df_all['label'].apply(lambda x: len(x) > 1)].drop_duplicates('label'))