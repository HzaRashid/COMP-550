import pandas as pd

fact_data_saint_petersburg = pd.read_csv(filepath_or_buffer='~/Desktop/comp550/data/facts.txt', 
                        sep='\t',
                        header=None
                        )

fake_data_saint_petersburg = pd.read_csv(filepath_or_buffer='~/Desktop/comp550/data/fakes.txt', 
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

# print(fact_data_saint_petersburg.head())
# print(fact_data.head())

all_facts = pd.concat([fact_data_saint_petersburg, fact_data])
print(all_facts.head())
print(all_facts.tail())

all_fakes = pd.concat([fake_data_saint_petersburg, fake_data])
print(all_fakes.head())

# # export
# all_facts.to_csv(
#     path_or_buf='./data/test_facts.txt', 
#     sep='\t',
#     index=False,
#     header=None
#     )

# all_fakes.to_csv(
#     path_or_buf='./data/test_fakes.txt', 
#     sep='\t',
#     index=False,
#     header=None
#     )


# fact_data[fact_data.columns[0]] = fact_data[fact_data.columns[0]].apply(lambda x: x.replace('\\', ''))
# fake_data[fake_data.columns[0]] = fake_data[fake_data.columns[0]].apply(lambda x: x.replace('\\', ''))
# fact_data.to_csv(path_or_buf='~/Downloads/facts.tsv', sep='\t', index=False, header=None)
# fake_data.to_csv(path_or_buf='~/Downloads/fakes.tsv', sep='\t', index=False, header=None)




# # test 

# test_facts = pd.read_csv(
#     filepath_or_buffer='./data/test_facts.txt',
#     sep='\t',
#     header=None
# )

# print(test_facts[test_facts.columns[0]][2])
# print(test_facts.head())
# print(test_facts.tail())


# test_fakes = pd.read_csv(
#     filepath_or_buffer='./data/test_fakes.txt',
#     sep='\t',
#     header=None
# )

# print(test_fakes[test_fakes.columns[0]][2])
# print(test_fakes.head())
# print(test_fakes.tail())