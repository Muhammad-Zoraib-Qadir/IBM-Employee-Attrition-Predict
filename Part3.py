import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
file_path = 'TV Shows - Association Rule Learning.csv' 
data = pd.read_csv(file_path)
transactions = [list(filter(pd.notna, row)) for row in data.values]
transaction_encoder = TransactionEncoder()
encoded_data = transaction_encoder.fit(transactions).transform(transactions)
transaction_df = pd.DataFrame(encoded_data, columns=transaction_encoder.columns_)
frequent_itemsets = apriori(transaction_df, min_support=0.01, use_colnames=True)
frequent_itemsets['num_itemsets'] = frequent_itemsets['itemsets'].apply(len)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5, num_itemsets=len(frequent_itemsets))
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules)
