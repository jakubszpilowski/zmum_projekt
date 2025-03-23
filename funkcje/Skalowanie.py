import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

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

    # Przekształcenie danych: Skalowanie (Min-Max) i Standaryzacja

    # Inicjalizacja skalerów
    min_max_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    # Min-Max Scaling
    df_missing_minmax = df_missing.copy()
    df_out_of_range_minmax = df_out_of_range.copy()
    df_outliers_minmax = df_outliers.copy()

    df_missing_minmax[['HP', 'Attack', 'Defense', 'Special_Attack', 'Special_Defense', 'Speed',
                       'Total']] = min_max_scaler.fit_transform(
        df_missing[['HP', 'Attack', 'Defense', 'Special_Attack', 'Special_Defense', 'Speed', 'Total']])
    df_out_of_range_minmax[['HP', 'Attack', 'Defense', 'Special_Attack', 'Special_Defense', 'Speed',
                            'Total']] = min_max_scaler.fit_transform(
        df_out_of_range[['HP', 'Attack', 'Defense', 'Special_Attack', 'Special_Defense', 'Speed', 'Total']])
    df_outliers_minmax[['HP', 'Attack', 'Defense', 'Special_Attack', 'Special_Defense', 'Speed',
                        'Total']] = min_max_scaler.fit_transform(
        df_outliers[['HP', 'Attack', 'Defense', 'Special_Attack', 'Special_Defense', 'Speed', 'Total']])

    # Standardization (Standaryzacja)
    df_missing_standard = df_missing.copy()
    df_out_of_range_standard = df_out_of_range.copy()
    df_outliers_standard = df_outliers.copy()

    df_missing_standard[['HP', 'Attack', 'Defense', 'Special_Attack', 'Special_Defense', 'Speed',
                         'Total']] = standard_scaler.fit_transform(
        df_missing[['HP', 'Attack', 'Defense', 'Special_Attack', 'Special_Defense', 'Speed', 'Total']])
    df_out_of_range_standard[['HP', 'Attack', 'Defense', 'Special_Attack', 'Special_Defense', 'Speed',
                              'Total']] = standard_scaler.fit_transform(
        df_out_of_range[['HP', 'Attack', 'Defense', 'Special_Attack', 'Special_Defense', 'Speed', 'Total']])
    df_outliers_standard[['HP', 'Attack', 'Defense', 'Special_Attack', 'Special_Defense', 'Speed',
                          'Total']] = standard_scaler.fit_transform(
        df_outliers[['HP', 'Attack', 'Defense', 'Special_Attack', 'Special_Defense', 'Speed', 'Total']])

    # Zapisanie wyników do plików CSV
    df_missing_minmax.to_csv('missing_data_minmax.csv', index=False)
    df_out_of_range_minmax.to_csv('out_of_range_data_minmax.csv', index=False)
    df_outliers_minmax.to_csv('outliers_data_minmax.csv', index=False)

    df_missing_standard.to_csv('missing_data_standard.csv', index=False)
    df_out_of_range_standard.to_csv('out_of_range_data_standard.csv', index=False)
    df_outliers_standard.to_csv('outliers_data_standard.csv', index=False)


    # Wizualizacja wyników przed i po skalowaniu

    def plot_comparison(data_before, data_after_minmax, data_after_standard, column_name):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Przed
        axes[0].hist(data_before[column_name], bins=20, color='lightblue', edgecolor='black')
        axes[0].set_title(f'Przed Skalowaniem: {column_name}')
        axes[0].set_xlabel(column_name)
        axes[0].set_ylabel('Częstotliwość')

        # Po Min-Max Scaling
        axes[1].hist(data_after_minmax[column_name], bins=20, color='lightgreen', edgecolor='black')
        axes[1].set_title(f'Po Min-Max Scaling: {column_name}')
        axes[1].set_xlabel(column_name)
        axes[1].set_ylabel('Częstotliwość')

        # Po Standaryzacji
        axes[2].hist(data_after_standard[column_name], bins=20, color='lightcoral', edgecolor='black')
        axes[2].set_title(f'Po Standaryzacji: {column_name}')
        axes[2].set_xlabel(column_name)
        axes[2].set_ylabel('Częstotliwość')

        # Dostosowanie wykresu
        plt.tight_layout()
        plt.show()


    # Wykresy porównawcze dla wybranych kolumn
    plot_comparison(df_missing, df_missing_minmax, df_missing_standard, 'HP')
    plot_comparison(df_out_of_range, df_out_of_range_minmax, df_out_of_range_standard, 'Attack')
    plot_comparison(df_outliers, df_outliers_minmax, df_outliers_standard, 'Defense')
