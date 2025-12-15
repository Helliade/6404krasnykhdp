import os
import pandas as pd
from implementation.data_reader import DataReader
from implementation.data_processors import DataProcessor
from implementation.visualization import Visualization

def main():
    """Основная функция выполнения анализа."""
    try:
        # Пути к файлам
        csv_file_path = "global_emissions.csv"
        parquet_file_path = "data/global_emissions.parquet"
        
        # Проверяем существование CSV файла
        if not os.path.exists(csv_file_path):
            print(f"Файл {csv_file_path} не найден!")
            print("Пожалуйста, скачайте данные с https://corgis-edu.github.io/corgis/csv/global_emissions/")
            return
        
        # Инициализация читателя данных
        data_reader = DataReader(csv_file_path, parquet_file_path)
        
        print("\n=== Анализ выбросов парниковых газов ===")
        
        # Сравнение скорости чтения
        data_reader.compare_read_speed()
        
        # ПАЙПЛАЙН: Чтение -> Извлечение -> Агрегация
        time_series_data = DataProcessor.aggregate(
            DataProcessor.extract_emission_data(
                data_reader.read_csv_chunks(chunksize=1000)
            )
        )
        
        # Задание 1: Агрегация данных. 3 самые «зеленые» и 3 самые «грязные» страны
        country_data = DataProcessor.aggregate_by_country([time_series_data]) # !!!!!!!!!!!!!!!!!!!!!!!!!!!
        analysis_results = DataProcessor.analyze_emissions(country_data)
        
        # Задание 2: Дисперсия и доверительный интервал. 3 страны с наибольшим и 3 с наименьшим разбросом суммы выбросов
        variance_results = DataProcessor.analyze_emissions_variance(time_series_data)
        confidence_results = DataProcessor.analyze_emissions_with_confidence(time_series_data)
        
        # Задание 3: Временные ряды и скользящее среднее. Общие ВВП (GDP) и общие выбросы
        temporal_results = DataProcessor.analyze_temporal_trends(time_series_data, window_size=5)
        
        # Дополнительное задание: Корреляция между населением страны и количеством выбрасываемых парниковых газов
        correlation = DataProcessor.calculate_correlation(data_reader, ['Country.Population', 'Emissions.Production.CO2.Total'])
        analysis_results['correlation_population_emissions'] = correlation
        
        # Вывод результатов
        print_results(analysis_results, variance_results, temporal_results)
        
        # ВИЗУАЛИЗАЦИЯ ВСЕХ ГРАФИКОВ
        print("\n=== Построение графиков ===")
        
        Visualization.plot_main_analysis(analysis_results, variance_results)
        if not confidence_results.empty:
            Visualization.plot_variance_analysis(variance_results, confidence_results)
        if not temporal_results.empty:
            Visualization.plot_temporal_analysis(temporal_results)
        
        # Дополнительный scatter plot для корреляции
        Visualization.plot_correlation_scatter(data_reader, 'Country.Population', 'Emissions.Production.CO2.Total')
        
        print("\nВсе задания выполнены успешно!")
        
    except Exception as e:
        print(f"Критическая ошибка в основном потоке выполнения: {e}")
        import traceback
        traceback.print_exc()


def print_results(results: pd.DataFrame, variance_results: pd.DataFrame, temporal_results: pd.DataFrame) -> None:
    """Выводит результаты анализа в консоль."""
    if results.empty:
        print("Нет результатов для вывода")
        return
        
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ АНАЛИЗА ВЫБРОСОВ ПАРНИКОВЫХ ГАЗОВ")
    print("="*60)
    
    try:
        print("\n1. САМЫЕ 'ЗЕЛЕНЫЕ' СТРАНЫ (наименьшие выбросы на душу населения):")
        top_green = results.iloc[0]['top_green_countries']
        for i, (_, row) in enumerate(top_green.iterrows(), 1):
            print(f"   {i}. {row['country']}: {row['avg_emissions_per_capita']:.6f} т CO₂/чел")
        
        print("\n2. САМЫЕ 'ГРЯЗНЫЕ' СТРАНЫ (наибольшие выбросы на душу населения):")
        top_dirty = results.iloc[0]['top_dirty_countries']
        for i, (_, row) in enumerate(top_dirty.iterrows(), 1):
            print(f"   {i}. {row['country']}: {row['avg_emissions_per_capita']:.6f} т CO₂/чел")
        
        if not variance_results.empty:
            print("\n3. СТРАНЫ С НАИМЕНЬШИМ РАЗБРОСОМ ВЫБРОСОВ:")
            low_var = variance_results.iloc[0]['lowest_variance_countries'].copy()
            for i, (_, row) in enumerate(low_var.iterrows(), 1):
                print(f"   {i}. {row['country']}: {row['variance']:.2e}")
            
            print("\n4. СТРАНЫ С НАИБОЛЬШИМ РАЗБРОСОМ ВЫБРОСОВ:")
            high_var = variance_results.iloc[0]['highest_variance_countries'].copy()
            for i, (_, row) in enumerate(high_var.iterrows(), 1):
                print(f"   {i}. {row['country']}: {row['variance']:.2e}")
        
        print("\n5. ОБЩИЕ ПОКАЗАТЕЛИ ЗА ВЕСЬ ПЕРИОД:")
        print(f"   Общий ВВП: {results.iloc[0]['total_gdp']:,.2f} $")
        print(f"   Общие выбросы CO₂: {results.iloc[0]['total_emissions']:,.2f} тонн")
        
        if not temporal_results.empty:
            print("\n6. ДАННЫЕ ВРЕМЕННЫХ РЯДОВ:")
            print(f"   Период наблюдений: {temporal_results['year'].min()}-{temporal_results['year'].max()}")
            print(f"   Количество лет: {len(temporal_results)}")
        
        if 'correlation_population_emissions' in results.columns:
            corr_value = results.iloc[0]['correlation_population_emissions']
            if corr_value is not None:
                correlation_strength = "сильная" if abs(corr_value) > 0.7 else \
                                     "умеренная" if abs(corr_value) > 0.3 else "слабая"
                print(f"\n7. КОРРЕЛЯЦИЯ НАСЕЛЕНИЕ-ВЫБРОСЫ: {corr_value:.3f} ({correlation_strength})")
                
    except Exception as e:
        print(f"Ошибка при выводе результатов: {e}")


if __name__ == "__main__":
    main()