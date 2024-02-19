import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
import os

df = pd.read_csv(os.path.join(os.getcwd(),'penguins_binary_classification.csv'))

#|%%--%%| <S8ws6fP3Fg|DyFT9evIMi>


pd.set_option("display.max_columns", None) #set options to show all columns

#|%%--%%| <DyFT9evIMi|sRUwf9v249>
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Taking only numeric components for PCA
X0 = df.select_dtypes(include='number')
Y0 = df['species']


scaler = StandardScaler()
x_train_s = scaler.fit_transform(X0)

pca = PCA(n_components=2)
x_reduced = pca.fit_transform(x_train_s)


#|%%--%%| <sRUwf9v249|qnzY1J1rl1>
# Using plotly
import plotly.express as px
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
features = list(X0.columns)

# Create DataFrame from reduced data
reduced_df = pd.DataFrame(data=x_reduced, columns=['PC1', 'PC2'])

# Add species column to reduced_df
reduced_df['species'] = df['species']

fig = px.scatter(reduced_df, x='PC1', y='PC2', color='species')

for i, feature in enumerate(features):
    fig.add_annotation(
        ax=0, ay=0,
        axref="x", ayref="y",
        x=loadings[i, 0],
        y=loadings[i, 1],
        showarrow=True,
        arrowsize=2,
        arrowhead=2,
        xanchor="right",
        yanchor="top"
    )
    fig.add_annotation(
        x=loadings[i, 0],
        y=loadings[i, 1],
        ax=0, ay=0,
        xanchor="center",
        yanchor="bottom",
        text=feature,
        yshift=5,
    )
fig.show()

#|%%--%%| <qnzY1J1rl1|nPTCrDCLs6>

print('Explained Variance Ratio: ', pca.explained_variance_ratio_)
print('Explained Variance Ratio CumSum: ', pca.explained_variance_ratio_.cumsum())
print('Singular Values: ', pca.singular_values_)

#|%%--%%| <nPTCrDCLs6|Ofsbt43mkP>

print('Components ', pca.components_)
print('shape: ', pca.components_.shape)
print('Components Transpased ', pca.components_.T)
print('EV ',pca.explained_variance_)

#|%%--%%| <Ofsbt43mkP|uPKMasAyPU>
import prince

pca = prince.PCA(n_components=2,
                 engine='sklearn',
                 random_state=1337)
# automatically one-hot encodes
pca = pca.fit(X0)

print(pca.eigenvalues_summary)
print(pca.row_coordinates(X0).head())
print(pca.column_coordinates_)


#|%%--%%| <uPKMasAyPU|RlzfjsHzEj>
# plot using plotly express
import plotly.express as px
import prince
# loadings = eigenvectors / sqrt(eigenvalues)
loadings = pca.column_coordinates_ * np.sqrt(pca.eigenvalues_)

# Create DataFrame from reduced data
# reduced_df = pd.DataFrame(data=pca.row_coordinates(X0), columns=['PC1', 'PC2'])
reduced_df = pca.row_coordinates(X0)

# Add species column to reduced_df
reduced_df['species'] = df['species']
reduced_df = reduced_df.rename(columns={0:'PC1',1:'PC2'})
print(reduced_df.head())

fig = px.scatter(reduced_df, x='PC1', y='PC2', color='species')

for i, feature in enumerate(features):
    fig.add_annotation(
        ax=0, ay=0,
        axref="x", ayref="y",
        x=loadings.iloc[i, 0],
        y=loadings.iloc[i, 1],
        showarrow=True,
        arrowsize=2,
        arrowhead=2,
        xanchor="right",
        yanchor="top"
    )
    fig.add_annotation(
        x=loadings.iloc[i, 0],
        y=loadings.iloc[i, 1],
        ax=0, ay=0,
        xanchor="center",
        yanchor="bottom",
        text=feature,
        yshift=5,
    )
fig.show()


