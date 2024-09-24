import pandas as pd

fact_data_saint_petersburg = pd.read_csv(filepath_or_buffer='~/Desktop/comp550/data/facts.txt', 
                        sep='\t',
                        header=None
                        )

fact_data = pd.read_csv(filepath_or_buffer='~/Downloads/facts.tsv', 
                        sep='\t',
                        header=None
                        )
fake_data = pd.read_csv(filepath_or_buffer='~/Downloads/fakes.tsv',
                        sep='\t',
                        header=None
                        )

print(fact_data_saint_petersburg.head())
print(fact_data.head())


# fact_data[fact_data.columns[0]] = fact_data[fact_data.columns[0]].apply(lambda x: x.replace('\\', ''))
# fake_data[fake_data.columns[0]] = fake_data[fake_data.columns[0]].apply(lambda x: x.replace('\\', ''))
# fact_data.to_csv(path_or_buf='~/Downloads/facts.tsv', sep='\t', index=False, header=None)
# fake_data.to_csv(path_or_buf='~/Downloads/fakes.tsv', sep='\t', index=False, header=None)