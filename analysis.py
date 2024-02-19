import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
import os

df = pd.read_csv(os.path.join(os.getcwd(),'penguins_binary_classification.csv'))

#|%%--%%| <S8ws6fP3Fg|DyFT9evIMi>


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


#|%%--%%| <ZAdhVKZ3RE|kKwUV7se4q>

# Plot to see any imbalanced
plt.figure()
df.groupby('species').count().iloc[:,0].plot(kind='bar')
plt.title('Balance of Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()

#|%%--%%| <kKwUV7se4q|z9Gf8hDObN>
# Make some seaborn plots with hue

g = sns.PairGrid(df,hue='species')
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()

#|%%--%%| <z9Gf8hDObN|EP67o6Lvsa>
# Investigate balance
a = df.groupby(by='species').count()
print(a)

# Adelie 151
# Gentoo           123
#|%%--%%| <EP67o6Lvsa|6Ej9wCBwhk>
r"""°°°
# Prepare the Dataset
- Encode Features
- Split dataset appropriately
- Will year be useful or not? let's experiment with Chi squared, and if it leads us correctly
°°°"""
#|%%--%%| <6Ej9wCBwhk|Lbs6HC98yY>
from sklearn.preprocessing import OneHotEncoder
# one-hot encoding for features
print(df['island'].unique())

df_encoded = pd.get_dummies(df,columns=['island',],dtype=float)
print(df_encoded)
#|%%--%%| <Lbs6HC98yY|PtF6SoLPly>

# df = pd.concat([df,encoded_islands], axis=1)
# print(df.head(10))

#replace Gentoo with 1 and Adelie with 0
df_encoded['species'][df_encoded['species'] == 'Adelie'] = 1
df_encoded['species'][df_encoded['species'] == 'Gentoo'] = 0
df_encoded['species'] = df_encoded['species'].astype('float')

print(df_encoded.head())
#|%%--%%| <PtF6SoLPly|JgrPOtGxIa>
r"""°°°
# Lets decide which features are best!
- Chi squared tests
- random forest top k important features
- logistic regression to test
- shap values to evaluate
°°°"""
#|%%--%%| <JgrPOtGxIa|f33NgBtkKj>

print(df_encoded.columns)

#|%%--%%| <f33NgBtkKj|2EsGEyK89h>
from sklearn.model_selection import train_test_split

features = ['bill_length_mm', 'bill_depth_mm','flipper_length_mm',
       'body_mass_g', 'year', 'island_Biscoe', 'island_Dream','island_Torgersen']
## Prepare data set
temp = df.sample(frac=1).reset_index(drop=True)
X = temp[features]
Y = temp['species']

X_train,X_test, y_train, y_test = train_test_split(X,Y, test_size=0.20, random_state=42)

#|%%--%%| <2EsGEyK89h|AOklVHGyqp>
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# Iterate through features, run logistic regression on one feature
def transform_features(features):
    transformed_features = features
    return transformed_features

def evaluate_features(trans_df,target):
    result_df = pd.DataFrame()
    for f in list(trans_df.columns):
        print(f)
        if f == target: continue
        X = trans_df[[f]]
        Y = trans_df[target]
        X_train,X_test, y_train, y_test = train_test_split(X,Y, test_size=0.20, random_state=42)
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        model = LogisticRegression(random_state=42)
        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)
        report = classification_report(y_pred,y_test,output_dict=True)
        prec_0 = report['0.0']['precision']
        prec_1 = report['1.0']['precision']
        acc = report['accuracy']
        coef = model.coef_.item()
        intercept = model.intercept_.item()
        temp_df = pd.DataFrame({'Feature':f,
                                '0_precision':prec_0,
                                '1_precision':prec_1,
                                'acc':acc,
                                'coefficent':coef,
                                'intercept':intercept}, index=[0])
        result_df = pd.concat([result_df,temp_df])
    return result_df


total_result = evaluate_features(df_encoded,'species')

print(total_result)


#|%%--%%| <AOklVHGyqp|If0queyVBw>

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

def logit_pvalue(model, x):
    """ Calculate z-scores for scikit-learn LogisticRegression.
    parameters:
        model: fitted sklearn.linear_model.LogisticRegression with intercept and large C
        x:     matrix on which the model was fit
    This function uses asymtptics for maximum likelihood estimates.
    """
    p = model.predict_proba(x)
    n = len(p)
    m = len(model.coef_[0]) + 1
    coefs = np.concatenate([model.intercept_, model.coef_[0]])
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
    ans = np.zeros((m, m))
    for i in range(n):
        ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i,1] * p[i, 0]
    vcov = np.linalg.inv(np.matrix(ans))
    se = np.sqrt(np.diag(vcov))
    t =  coefs/se  
    p = (1 - norm.cdf(abs(t))) * 2
    return p

# test p-values
x = np.arange(10)[:, np.newaxis]
y = np.array([0,0,0,1,0,0,1,1,1,1])
model = LogisticRegression(C=1e30).fit(x, y)
print(logit_pvalue(model, x))
#|%%--%%| <If0queyVBw|7hl1VxAjwS>

print(df_encoded['species'].unique())
print(df_encoded['bill_length_mm'].unique())
#|%%--%%| <7hl1VxAjwS|TR2lUAHbGa>
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

result_df = pd.DataFrame()
f = 'bill_length_mm'
target = 'species'
X = df_encoded[[f]]
print(X.shape)
Y = df_encoded[[target]]
print(Y.shape)
X_train,X_test, y_train, y_test = train_test_split(X,Y, test_size=0.20, random_state=42)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

model = LogisticRegression(random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
report = classification_report(y_pred,y_test,output_dict=True)
prec_0 = report['0.0']['precision']
prec_1 = report['1.0']['precision']
acc = report['accuracy']
temp_df = pd.DataFrame({'Feature':f,
                        '0_precision':prec_0,
                        '1_precision':prec_1,
                        'acc':acc}, index=[0])
result_df = pd.concat([result_df,temp_df])

print(logit_pvalue(model, X_test))


#|%%--%%| <TR2lUAHbGa|B5ywvC7aXX>
import prince

famd= prince.FAMD(n_components=2,
                  n_iter=3,
                  copy=True,
                  check_input=True,
                  random_state=42,
                  engine="sklearn",
                  handle_unknown="error"
                  )

famd = famd.fit(df)

famd.eigenvalues_summary

# Coordinates

# famd.row_coordinates(df).head()
famd.column_coordinates_

import altair as alt
# alt.renderers.enable("notebook")

chart=famd.plot(df,
          x_component=0,
          y_component=1
          )
#|%%--%%| <B5ywvC7aXX|sRUwf9v249>
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Taking only numeric components for PCA
X0 = df.select_dtypes(include='number')
Y0 = df['species']


scaler = StandardScaler()
x_train_s = scaler.fit_transform(X0)

pca = PCA(n_components=2)
x_reduced = pca.fit_transform(x_train_s)


#|%%--%%| <sRUwf9v249|XtmSpfSZpE>
import numpy as np
# plot PCA
colors = Y0.map({'Adelie':'r','Gentoo':'b'})

# Get Axis names
print(pd.DataFrame(pca.components_,columns=X0.columns,index = ['PC-1','PC-2']))

_, ax = plt.subplots()
scatter = ax.scatter(x_reduced[:,0],x_reduced[:,1], c=colors)

# Create legend
# legend1 = ax.legend(scatter.legend_elements(),
#              ('Adelie','Gentoo'),
#               loc='lower right',
#               title='species')
# plt.show()
# Get unique labels and create legend handles

labels = list(y_train.unique())
legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=color, markersize=10) 
                  for color in ['r', 'b']]

# Create legend
ax.legend(handles=legend_handles, labels=labels , loc='lower right', title='species')
evr = pca.explained_variance_ratio_
ax.set_xlabel(f'PC-1 {evr[0]:.2%} Explained Variance')
ax.set_ylabel(f'PC-2 {evr[1]:.2%} Explained Variance')

plt.show()
#|%%--%%| <XtmSpfSZpE|qnzY1J1rl1>
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

#|%%--%%| <nPTCrDCLs6|uPKMasAyPU>
import prince
## Taking only categorical components for MCA (Multiple Coorespondance Analysis)

X0 = df.select_dtypes(include='object')
X0 = X0.loc[:,X0.columns != 'species']
Y0 = df['species']

mca = prince.MCA(n_components=2,
                 engine='sklearn',
                 random_state=1337)
# automatically one-hot encodes
mca = mca.fit(X0)

print(mca.eigenvalues_summary)
print(mca.row_coordinates(X0).head())
print(mca.column_coordinates(X0).head())



#|%%--%%| <uPKMasAyPU|r35WMTXeC4>

# take row_coordinates and plot
# take column_coordinates, and plot arrows
colors = Y0.map({'Adelie':'r','Gentoo':'b'})
coords = mca.row_coordinates(X0)
vects = mca.column_coordinates(X0)
_, ax2 = plt.subplots()
scat2 = ax2.scatter(coords.iloc[:,0],coords.iloc[:,1],color=colors)
scat3 = ax2.scatter(vects.iloc[:,0],vects.iloc[:,1])
plt.show()

#|%%--%%| <r35WMTXeC4|iKqHxCWKAQ>

print(mca.explained_variance_ratio)
print(len(vects))
print(vects)

#|%%--%%| <iKqHxCWKAQ|RlzfjsHzEj>
# plot using plotly express
import plotly.express as px
import prince
# loadings = eigenvectors / sqrt(eigenvalues)


