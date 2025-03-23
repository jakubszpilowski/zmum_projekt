import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Podana ścieżka do pliku CSV
file_path = r""  # Zmień tę ścieżkę na właściwą

# Wczytanie danych
df = pd.read_csv(file_path)

# Sprawdzenie typów danych
print("Podstawowe informacje o zbiorze danych:")
print(df.info())

# Wyświetlenie pierwszych kilku wierszy danych
print("\nPierwsze 5 wierszy danych:")
print(df.head())

# 1. **Analiza cech numerycznych**
numeric_features = df.select_dtypes(include=[np.number])
print("\nCechy numeryczne:")
print(numeric_features.describe())  # Podstawowe statystyki dla cech numerycznych

# 2. **Korelacje między cechami numerycznymi**
correlation_matrix = numeric_features.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt='.2f')
plt.title('Macierz korelacji między cechami numerycznymi')
plt.tight_layout()
plt.show()

# 3. **Analiza cech kategorycznych**
categorical_features = df.select_dtypes(include=['object'])

# Wyświetlenie unikalnych wartości dla cech kategorycznych
print("\nCechy kategoryczne:")
print(categorical_features.nunique())

# Przykład kodowania cech kategorycznych
# Jeśli mamy cechę kategoryczną 'Type', np. w danych Pokémonów
if 'Type' in categorical_features.columns:
    le = LabelEncoder()
    df['Type_encoded'] = le.fit_transform(df['Type'])
    print("\nZakodowane wartości 'Type':")
    print(df[['Type', 'Type_encoded']].head())


# 5. **Zestawienie cech**
# Jeśli mamy powiązane cechy, możemy je zestawić (np. HP vs Attack)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='HP', y='Attack', color='purple')
plt.title('Zestawienie cech: HP vs Attack')
plt.xlabel('HP')
plt.ylabel('Attack')
plt.tight_layout()
plt.show()