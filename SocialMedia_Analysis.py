import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# -----------------------------
# 1) Daten laden & bereinigen
# -----------------------------
df = pd.read_csv('./data/Time-Wasters on Social Media.csv')
df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
df['total_time_spent'] = df['total_time_spent'] / 60  # Minuten → Stunden

# -----------------------------
# 2) Auswahl relevanter Spalten
# -----------------------------
cols = [
    'userid', 'age', 'gender', 'platform', 'total_time_spent',
    'number_of_sessions', 'addiction_level', 'self_control',
    'satisfaction', 'productivityloss'
]
df = df[cols]

# -----------------------------
# 3) Deskriptive Statistik
# -----------------------------
print("\nDeskriptive Statistik (numerisch):")
print(df.describe())

# -----------------------------
# 4) Plattformvergleich: Nutzungsdauer
# -----------------------------
print("\nPlattformvergleich (⏱ Ø Stunden/Woche):")
platform_means = df.groupby('platform')['total_time_spent'].mean().sort_values(ascending=False)
print(platform_means)

plt.figure(figsize=(10, 6))
ax = sns.boxplot(x='platform', y='total_time_spent', data=df, palette='Set2')
plt.title("Nutzungsdauer pro Plattform")
plt.xlabel("Plattform")
plt.ylabel("Total Time Spent (Stunden/Tag)")
plt.xticks(rotation=45, ha='right')

# Ø beschriften
for i, mean in enumerate(platform_means):
    ax.text(i, mean + 0.1, f"{mean:.1f}", ha='center', fontsize=9, color='black')

plt.tight_layout()
plt.show()

# -----------------------------
# 5) Clusteranalyse: Aktivitätsmuster
# -----------------------------
features = ['total_time_spent', 'number_of_sessions']
X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df.loc[X.index, 'activity_cluster'] = clusters

sil_score = silhouette_score(X_scaled, clusters)
print(f"\nClusteranalyse (Aktivitätsdaten) abgeschlossen – Silhouette-Score: {sil_score:.2f}")

# PCA-Plot mit Cluster-Beschriftung
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)

# Beschreibungen zu den Clustern
cluster_labels = {
    0: "Kontrollierte Vielnutzer",
    1: "Unzufriedene Gelegenheitssurfer",
    2: "Impulsive Dauernutzer"
}

# PCA-Datenframe zur leichteren Annotation
pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
pca_df['cluster'] = clusters
pca_df['label'] = pca_df['cluster'].map(cluster_labels)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='label', palette='Set1')
plt.title("Nutzercluster basierend auf Aktivität (PCA + Interpretation)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.legend(title="Clusterbeschreibung")
plt.tight_layout()
plt.show()

# -----------------------------
# 6) Altersgruppenanalyse
# -----------------------------
bins = [0, 25, 35, 50, 65, 100]
labels = ['18–25', '26–35', '36–50', '51–65', '65+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

age_group_means = df.groupby('age_group')['total_time_spent'].mean().round(2)
print("\nDurchschnittliche Nutzungsdauer pro Altersgruppe:")
print(age_group_means)

plt.figure(figsize=(8, 5))
sns.barplot(x=age_group_means.index, y=age_group_means.values, palette='muted')
plt.title("⏱ Durchschnittliche Social-Media-Nutzung pro Altersgruppe")
plt.ylabel("Total Time Spent (Stunden/Tag)")
plt.xlabel("Altersgruppe")
plt.tight_layout()
plt.show()
