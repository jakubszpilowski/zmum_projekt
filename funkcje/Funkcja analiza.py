import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Podana ścieżka do pliku CSV
file_path = r""

# Sprawdzenie, czy plik istnieje
if not os.path.exists(file_path):
    print(f"Plik o ścieżce {file_path} nie istnieje!")
else:
    try:
        # Wczytanie danych z pliku CSV
        df = pd.read_csv(file_path)
        print("Plik CSV wczytany pomyślnie.")

        # Sprawdzamy, czy dane zostały wczytane poprawnie
        print("\nPierwsze 5 wierszy danych:")
        print(df.head())
        print("\nPodstawowe informacje o zbiorze danych:")
        print(df.info())
        print("\nPodstawowe statystyki opisowe:")
        print(df.describe())

    except Exception as e:
        print(f"Wystąpił błąd przy wczytywaniu pliku: {e}")
        exit()

    # 1. Prosty sposób 1 - Wykrywanie brakujących danych
    missing_data_simple = df.isnull().sum()

    # 2. Prosty sposób 2 - Wykrywanie danych spoza zakresu (np. HP < 0 lub > 255)
    out_of_range = (df[['HP', 'Attack', 'Defense', 'Special_Attack', 'Special_Defense', 'Speed', 'Total']] < 0) | (
                df[['HP', 'Attack', 'Defense', 'Special_Attack', 'Special_Defense', 'Speed', 'Total']] > 255)

    # 3. Trudniejszy sposób - Wykrywanie wartości odstających (outliers) przy użyciu IQR (Interquartile Range)
    def detect_outliers_iqr(data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data < lower_bound) | (data > upper_bound)

    outliers = df[['HP', 'Attack', 'Defense', 'Special_Attack', 'Special_Defense', 'Speed', 'Total']].apply(
        detect_outliers_iqr)

    # Tworzenie trzech zbiorów danych na podstawie wyników analizy
    df_missing = df[df.isnull().any(axis=1)]
    df_out_of_range = df[out_of_range.any(axis=1)]
    df_outliers = df[outliers.any(axis=1)]

    # Zapisanie wyników do plików CSV
    df_missing.to_csv('missing_data.csv', index=False)
    df_out_of_range.to_csv('out_of_range_data.csv', index=False)
    df_outliers.to_csv('outliers_data.csv', index=False)

    # Wyświetlenie podsumowania wyników
    print("\nZawartość zbioru danych z brakującymi wartościami:")
    print(df_missing.head())
    print("\nZawartość zbioru danych z wartościami poza zakresem:")
    print(df_out_of_range.head())
    print("\nZawartość zbioru danych z wykrytymi wartościami odstającymi:")
    print(df_outliers.head())

    # Wizualizacja wyników:

    # Wykres 1: Wykres brakujących danych
    plt.figure(figsize=(12, 6))
    missing_data_simple.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Liczba brakujących danych w każdej kolumnie', fontsize=16, fontweight='bold')
    plt.ylabel('Liczba brakujących wartości', fontsize=14)
    plt.xlabel('Kolumny', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Wykres 2: Wykres wartości spoza zakresu (np. HP < 0 lub > 255)
    out_of_range_sum = out_of_range.sum()
    plt.figure(figsize=(12, 6))
    out_of_range_sum.plot(kind='bar', color='salmon', edgecolor='black')
    plt.title('Liczba wartości spoza zakresu w każdej kolumnie', fontsize=16, fontweight='bold')
    plt.ylabel('Liczba wartości spoza zakresu', fontsize=14)
    plt.xlabel('Kolumny', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Wykres 3: Wykres pudełkowy (boxplot) dla wartości liczbowych, aby zidentyfikować wartości odstające
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[['HP', 'Attack', 'Defense', 'Special_Attack', 'Special_Defense', 'Speed', 'Total']],
                palette="Set2")
    plt.title('Wykres pudełkowy (boxplot) dla danych liczbowych', fontsize=16, fontweight='bold')
    plt.ylabel('Liczba brakujących wartości', fontsize=14)
    plt.xlabel('Kolumny', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Wykres 4: Histogram dla jednej z kolumn (np. HP), aby pokazać rozkład danych
    plt.figure(figsize=(12, 6))
    df['HP'].plot(kind='hist', bins=20, color='lightgreen', edgecolor='black', rwidth=0.9)
    plt.title('Histogram rozkładu wartości HP', fontsize=16, fontweight='bold')
    plt.xlabel('HP', fontsize=14)
    plt.ylabel('Częstotliwość', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Wykres 5: Wykres rozrzutu (scatter plot) dla dwóch kolumn (np. HP vs Attack)
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=df['HP'], y=df['Attack'], color='purple', s=100, alpha=0.6)
    plt.title('Wykres rozrzutu: HP vs Attack', fontsize=16, fontweight='bold')
    plt.xlabel('HP', fontsize=14)
    plt.ylabel('Attack', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()