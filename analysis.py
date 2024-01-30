import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
import os

#|%%--%%| <lWWauUuS4N|DyFT9evIMi>

df = pd.read_csv(os.path.join(os.getcwd(),'penguins_binary_classification.csv'))

pd.set_option("display.max_columns", None) #set options to show all columns

#|%%--%%| <DyFT9evIMi|fLxzS78OHm>
df.info()  

#|%%--%%| <fLxzS78OHm|ZAdhVKZ3RE>

print(df.describe(include='all'))
# df.head()

# Check if there is null
n = df[df.isna().any(axis=1)]
print("Size: ", len(n))

# Drop null in place
# df.dropna(inplace=True)
# print("Size: ", len(df))


#|%%--%%| <ZAdhVKZ3RE|z9Gf8hDObN>
# Make some seaborn plots with hue

g = sns.PairGrid(df,hue='species')
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()

#|%%--%%| <z9Gf8hDObN|6Ej9wCBwhk>
r"""°°°
# Prepare the Dataset
- Encode Features
- Split dataset appropriately
- Will year be useful or not? let's experiment with Chi squared, and if it leads us correctly
°°°"""
#|%%--%%| <6Ej9wCBwhk|Lbs6HC98yY>
from sklearn.preprocessing import LabelEncoder
# one-hot encoding for features
print(df['island'].unique())

df_encoded = pd.get_dummies(df,columns=['island',],dtype=float)
print(df_encoded)
#|%%--%%| <Lbs6HC98yY|PtF6SoLPly>

# df = pd.concat([df,encoded_islands], axis=1)
print(df.head(10))

#replace Gentoo with 1 and Adelie with 0

#|%%--%%| <PtF6SoLPly|JgrPOtGxIa>
r"""°°°
# Lets decide which features are best!
- Chi squared tests
- random forest top k important features
- logistic regression to test
- shap values to evaluate
°°°"""
#|%%--%%| <JgrPOtGxIa|2EsGEyK89h>



