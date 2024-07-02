import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os 

df = pd.read_csv('./data/raw/dvc-s3-data.csv')

X = df.drop(columns=['PLACED'])
y = df['PLACED']


# scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Creating result with pca

df_pca = pd.DataFrame(data=X_pca,columns=['PC1','PC2','PC3'])
df_pca['PLACED'] = y.values

df_pca.to_csv(os.path.join('data','processed','stud_perf_pca.csv'),index=False)
