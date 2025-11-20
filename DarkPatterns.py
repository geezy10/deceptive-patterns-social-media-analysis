import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1) Daten laden
# -----------------------------
df = pd.read_csv('./data/dark-patterns-v2.csv')

print("Spaltenübersicht:")
print(df.columns)
print(df.head())

# -----------------------------
# 2) Grundlegende Infos
# -----------------------------
print("\nAnzahl Zeilen (Dark Pattern Instanzen):", df.shape[0])
print("\nAnzahl unterschiedlicher Websites:", df['Website Page'].nunique() if 'Website Page' in df.columns else 'Spalte fehlt')
print("\nAnzahl verschiedener Dark Pattern Typen:", df['Pattern Type'].nunique() if 'Pattern Type' in df.columns else 'Spalte fehlt')

# -----------------------------
# 3) Dark Pattern Kategorien: Häufigkeiten
# -----------------------------
sns.set_theme(style="whitegrid")

if 'Pattern Category' in df.columns:
    pattern_counts = df['Pattern Category'].value_counts().reset_index()
    pattern_counts.columns = ['Pattern Category', 'Count']

    print("\nHäufigkeiten pro Dark Pattern Kategorie:\n", pattern_counts)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x='Pattern Category',
        y='Count',
        data=pattern_counts,
        order=pattern_counts['Pattern Category'],
        palette='viridis'
    )

    # Balken-Beschriftungen
    for index, row in pattern_counts.iterrows():
        ax.text(index, row['Count'] + 5, row['Count'], color='black', ha="center")

    plt.xticks(rotation=45, ha='right')
    plt.title('Häufigkeit pro Dark Pattern Kategorie', fontsize=14)
    plt.ylabel('Anzahl Instanzen')
    plt.xlabel('Pattern Kategorie')
    plt.tight_layout()
    plt.show()

# -----------------------------
# 4) Anteil betroffener Websites
# -----------------------------
if 'Website Page' in df.columns:
    total_websites = 11000  # lt. Paper
    websites_with_patterns = df['Website Page'].nunique()
    share = websites_with_patterns / total_websites * 100
    print(f"\n{websites_with_patterns} von ~{total_websites} Websites ({share:.2f}%) haben Dark Patterns.")

# -----------------------------
# 5) Deceptive Patterns robust filtern
# -----------------------------
if 'Deceptive?' in df.columns:
    df['Deceptive?'] = df['Deceptive?'].astype(str).str.strip().str.lower()
    deceptive_df = df[df['Deceptive?'].isin(['true', 'yes', '1'])]

    if not deceptive_df.empty:
        deceptive_counts = deceptive_df['Pattern Category'].value_counts().reset_index()
        deceptive_counts.columns = ['Pattern Category', 'Count']

        print("\nDeceptive Patterns pro Kategorie:\n", deceptive_counts)

        plt.figure(figsize=(10, 5))
        ax = sns.barplot(
            x='Pattern Category',
            y='Count',
            data=deceptive_counts,
            order=deceptive_counts['Pattern Category'],
            palette='magma'
        )

        # Balken-Beschriftungen
        for index, row in deceptive_counts.iterrows():
            ax.text(index, row['Count'] + 1, row['Count'], color='black', ha="center")

        plt.xticks(rotation=45, ha='right')
        plt.title('Häufigkeit deceptive Dark Patterns pro Kategorie', fontsize=14)
        plt.ylabel('Anzahl Instanzen')
        plt.xlabel('Pattern Kategorie')
        plt.tight_layout()
        plt.show()

        total_deceptive_sites = deceptive_df['Website Page'].nunique()
        print(f"\nInsgesamt {total_deceptive_sites} Websites nutzen deceptive Patterns.")
    else:
        print("\nKeine als 'deceptive' markierten Instanzen gefunden. Prüfe den Spalteninhalt!")
