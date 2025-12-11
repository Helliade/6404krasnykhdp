import pandas as pd
import numpy as np
from typing import Generator

class DataProcessor:
    """Класс с генераторами для обработки данных."""
    
    @staticmethod
    def extract_emission_data(chunk_stream: Generator[pd.DataFrame, None, None]) -> Generator[pd.DataFrame, None, None]:
        """Извлекает и агрегирует данные по выбросам из реальных данных CORGIS."""
        for chunk in chunk_stream:
            try:
                # Создаем копию для безопасного изменения
                processed_chunk = chunk.copy()
                
                # Основные поля из реальных данных
                processed_chunk['country'] = processed_chunk.get('Country.Name', '')
                processed_chunk['year'] = processed_chunk.get('Year', 0)
                processed_chunk['population'] = processed_chunk.get('Country.Population', 0)
                processed_chunk['gdp'] = processed_chunk.get('Country.GDP', 0)
                processed_chunk['emissions'] = processed_chunk.get('Emissions.Production.CO2.Total', 0) * 1_000_000
                
                # Выбросы на душу населения (тонны/человека)
                processed_chunk['emissions_per_capita'] = (
                    processed_chunk['emissions'] / processed_chunk['population']).where(processed_chunk['population'] > 0, 0)
                
                # Фильтруем только нужные столбцы и валидные строки
                result_columns = ['country', 'year', 'emissions', 'population', 'gdp', 'emissions_per_capita']
                valid_data = processed_chunk[result_columns].dropna()
                
                yield valid_data
                
            except Exception as e:
                print(f"Ошибка при обработке чанка: {e}")
                continue
    
    @staticmethod
    def aggregate_by_country(data_stream: Generator[pd.DataFrame, None, None]) -> pd.DataFrame:
        """Агрегирует данные по странам в DataFrame."""
        result_agg = None
        
        for data_frame in data_stream:
            if data_frame.empty:
                continue
                
            # Агрегируем текущий чанк
            chunk_agg = data_frame.groupby('country').agg({
                'emissions': 'sum',
                'population': 'sum',
                'gdp': 'sum',
                'emissions_per_capita': ['sum', 'count']  # sum для сложения, count для подсчета
            })
            
            if result_agg is None:
                result_agg = chunk_agg
            else:
                result_agg = result_agg.add(chunk_agg, fill_value=0)  # отсутствующие значения считаются 0
        
        if result_agg is None:
            return pd.DataFrame()
        
        # Распрямляем мультииндекс столбцов
        result_agg.columns = [
            'emissions_sum', 
            'population_sum', 
            'gdp_sum', 
            'emissions_per_capita_sum', 
            'emissions_per_capita_count'
        ]
        
        # Вычисляем среднее для emissions_per_capita
        result_agg['avg_emissions_per_capita'] = (
            result_agg['emissions_per_capita_sum'] / result_agg['emissions_per_capita_count']
        ).fillna(0)
        
        # Сбрасываем индекс и переименовываем столбцы
        result_agg = result_agg.reset_index()
        
        final_result = pd.DataFrame({
            'country': result_agg['country'],
            'total_emissions': result_agg['emissions_sum'],
            'total_population': result_agg['population_sum'],
            'total_gdp': result_agg['gdp_sum'],
            'avg_emissions_per_capita': result_agg['avg_emissions_per_capita']
        })
        
        return final_result
    
    @staticmethod
    def get_time_series_data(data_stream: Generator[pd.DataFrame, None, None]) -> pd.DataFrame:
        """Возвращает данные временных рядов для анализа."""
        all_data = []
        
        for data_frame in data_stream:            
            all_data.append(data_frame)
        if not all_data:
            return pd.DataFrame()
        
        # Объединяем все данные
        combined_df = pd.concat(all_data, ignore_index=True)
        
        return combined_df
    
    @staticmethod
    def analyze_emissions(country_data: pd.DataFrame) -> pd.DataFrame:
        """Выполняет анализ данных по выбросам и ВВП и возвращает DataFrame с результатами."""
        if country_data.empty:
            raise ValueError("Не найдено стран с валидными данными для анализа")
        
        # Фильтруем страны с валидными данными
        valid_countries = country_data[
            (country_data['avg_emissions_per_capita'] > 0)
        ].copy()
        
        print(f"Анализируем данные по {len(valid_countries)} странам")
        
        # Создаем результат анализа как DataFrame (из словаря)
        analysis_results = pd.DataFrame({
            'total_gdp': [valid_countries['total_gdp'].sum()],
            'total_emissions': [valid_countries['total_emissions'].sum()]
        })
        
        top_green = valid_countries.nsmallest(3, 'avg_emissions_per_capita')[['country', 'avg_emissions_per_capita']]
        analysis_results['top_green_countries'] = [top_green]
        
        top_dirty = valid_countries.nlargest(3, 'avg_emissions_per_capita')[['country', 'avg_emissions_per_capita']]
        analysis_results['top_dirty_countries'] = [top_dirty]
        
        return analysis_results
    
    @staticmethod
    def analyze_emissions_variance(time_series_data: pd.DataFrame) -> pd.DataFrame:
        """Анализ дисперсии выбросов по странам."""
        if time_series_data.empty:
            return pd.DataFrame()
        
        # Вычисляем дисперсию по годам для каждой страны
        variance_by_country = time_series_data.groupby('country')['emissions'].var().reset_index()
        variance_by_country.columns = ['country', 'variance']
        
        # Фильтруем страны с достаточным количеством данных
        valid_variance = variance_by_country[variance_by_country['variance'] > 0].copy()
        
        lowest_variance = valid_variance.nsmallest(3, 'variance')[['country', 'variance']]
        highest_variance = valid_variance.nlargest(3, 'variance')[['country', 'variance']]
        
        # Создаем результат анализа
        variance_results = pd.DataFrame({
            'lowest_variance_countries': [lowest_variance],
            'highest_variance_countries': [highest_variance]
        })
        
        return variance_results
    
    @staticmethod
    def analyze_emissions_with_confidence(time_series_data: pd.DataFrame) -> pd.DataFrame:
        """Анализ выбросов с доверительными интервалами."""
        if time_series_data.empty:
            return pd.DataFrame()
        
        confidence_results = []
        
        for country in time_series_data['country'].unique():
            country_data = time_series_data[time_series_data['country'] == country]
            
            if len(country_data) > 1:
                emissions_series = country_data['emissions']
                mean_emissions = emissions_series.mean()
                lower, upper = DataProcessor.calculate_confidence_interval(emissions_series)
                
                confidence_results.append({
                    'country': country,
                    'mean_emissions': mean_emissions,
                    'confidence_lower': lower,
                    'confidence_upper': upper
                })
        
        return pd.DataFrame(confidence_results)
    
    @staticmethod
    def calculate_confidence_interval(data: pd.Series, confidence: float = 0.95) -> pd.Series:
        """Вычисляет доверительный интервал для данных."""
        if len(data) < 2:
            return pd.Series([0, 0])
        
        mean = data.mean()
        std_err = data.std(ddof=1) / np.sqrt(len(data))
        z_score = 1.96  # Для 95% доверительного уровня
        
        margin_of_error = z_score * std_err
        return pd.Series([mean - margin_of_error, mean + margin_of_error])
    
    @staticmethod
    def calculate_moving_average(data: pd.Series, window: int = 3) -> pd.Series:
        """Вычисляет скользящее среднее для временного ряда."""
        if len(data) < window:
            return data
        
        return data.rolling(window=window, min_periods=1).mean()
    
    @staticmethod
    def analyze_temporal_trends(time_series_data: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
        """Анализирует временные тенденции ВВП и выбросов."""
        if time_series_data.empty:
            return pd.DataFrame()
        
        # Агрегируем общие показатели по годам
        yearly_totals = time_series_data.groupby('year').agg({
            'gdp': 'sum',
            'emissions': 'sum'
        }).reset_index().sort_values('year')
        
        # Вычисляем скользящее среднее
        yearly_totals['gdp_moving_avg'] = DataProcessor.calculate_moving_average(yearly_totals['gdp'], window_size)
        yearly_totals['emissions_moving_avg'] = DataProcessor.calculate_moving_average(yearly_totals['emissions'], window_size)
        
        return yearly_totals
    
    @staticmethod
    def calculate_correlation(parquet_reader, columns: list) -> float:
        """Вычисляет корреляцию между населением и выбросами."""
        try:
            df = parquet_reader.read_parquet_columns(columns)
            
            if len(columns) == 2 and all(col in df.columns for col in columns):
                population_col, emissions_col = columns
                
                # Убираем нулевые и отрицательные значения
                valid_data = df[
                    (df[population_col] > 0) & 
                    (df[emissions_col] > 0)
                ]
                
                if len(valid_data) > 1:
                    correlation = valid_data[population_col].corr(valid_data[emissions_col])
                    print(f"Рассчитана корреляция между {population_col} и {emissions_col}: {correlation:.3f}")
                    print(f"Количество точек данных: {len(valid_data)}")
                    return correlation
                else:
                    print("Недостаточно данных для расчета корреляции после фильтрации")
                    return 0.0
            
            print("Предупреждение: Не найдены необходимые столбцы для расчета корреляции")
            return 0.0
            
        except Exception as e:
            print(f"Ошибка при расчете корреляции: {e}")
            return 0.0