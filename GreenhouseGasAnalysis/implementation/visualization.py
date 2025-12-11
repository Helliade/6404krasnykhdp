import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Visualization:
    """Класс для визуализации результатов анализа с DataFrame."""
    
    @staticmethod
    def plot_main_analysis(results: pd.DataFrame, variance_results: pd.DataFrame) -> None:
        """Строит графики для основных заданий."""
        if results.empty:
            print("Нет данных для построения графиков")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Анализ выбросов парниковых газов по странам', fontsize=16, fontweight='bold')
        
        # Задание 1: Самые "зеленые" и "грязные" страны
        Visualization._plot_green_dirty_countries(ax1, results)
        
        # Задание 2: Разброс выбросов
        Visualization._plot_emission_variance(ax2, variance_results)
        
        # Задание 3: Общие показатели
        Visualization._plot_total_metrics(ax3, results)
        
        # Дополнительное задание: Корреляция
        Visualization._plot_correlation_placeholder(ax4, results)
        
        plt.tight_layout()
        plt.savefig('emissions_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def _plot_green_dirty_countries(ax: plt.Axes, results: pd.DataFrame) -> None:
        """График самых 'зеленых' и 'грязных' стран."""
        try:
            top_green = results.iloc[0]['top_green_countries']
            top_dirty = results.iloc[0]['top_dirty_countries']
            
            if top_green.empty or top_dirty.empty:
                ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)
                return
            
            # Объединяем данные
            combined_df = pd.concat([
                top_green.assign(type='green'),
                top_dirty.assign(type='dirty')
            ], ignore_index=True)
            
            colors = ['#2ecc71' if t == 'green' else '#e74c3c' for t in combined_df['type']]
            
            bars = ax.bar(combined_df['country'], combined_df['avg_emissions_per_capita'], 
                         color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_title('Самые "зеленые" и "грязные" страны\n(выбросы CO₂ на душу населения)', fontweight='bold')
            ax.set_ylabel('Выбросы CO₂ (тонн/человека)')
            
            # ДОБАВЛЕНО: логарифмическая ось Y для этого графика
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, axis='y', which='both')
            
            ax.tick_params(axis='x', rotation=45)
            
            # Добавляем подписи значений
            for bar, value in zip(bars, combined_df['avg_emissions_per_capita']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{value:.4f} т/чел', ha='center', va='bottom', fontweight='bold')
            
            ax.legend(['Низкие выбросы', 'Высокие выбросы'], loc='upper right')
            
        except Exception as e:
            print(f"Ошибка при построении графика зеленых/грязных стран: {e}")
            ax.text(0.5, 0.5, 'Ошибка данных', ha='center', va='center', transform=ax.transAxes)
    
    @staticmethod
    def _plot_emission_variance(ax: plt.Axes, variance_results: pd.DataFrame) -> None:
        """График разброса выбросов по странам."""
        try:
            if variance_results.empty:
                ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)
                return
            
            low_var = variance_results.iloc[0]['lowest_variance_countries']
            high_var = variance_results.iloc[0]['highest_variance_countries']
            
            if low_var.empty or high_var.empty:
                ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)
                return
            
            # Объединяем данные
            combined_df = pd.concat([
                low_var.assign(type='low'),
                high_var.assign(type='high')
            ], ignore_index=True)
            
            colors = ['#3498db' if t == 'low' else '#f39c12' for t in combined_df['type']]
            
            bars = ax.bar(combined_df['country'], combined_df['variance'], 
                         color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_title('Страны с наименьшим и наибольшим\nразбросом выбросов CO₂', fontweight='bold')
            ax.set_ylabel('Дисперсия выбросов CO₂ (тонны²)')
            
            # ДОБАВЛЕНО: логарифмическая ось Y для этого графика
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, axis='y', which='both')
            
            ax.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, combined_df['variance']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{value:.2e} т²', ha='center', va='bottom', fontweight='bold')
            
            ax.legend(['Стабильные', 'Волатильные'], loc='upper right')
            
        except Exception as e:
            print(f"Ошибка при построении графика дисперсии: {e}")
            ax.text(0.5, 0.5, 'Ошибка данных', ha='center', va='center', transform=ax.transAxes)
    
    @staticmethod
    def _plot_total_metrics(ax: plt.Axes, results: pd.DataFrame) -> None:
        """График общих показателей ВВП и выбросов."""
        try:
            metrics = ['Общий ВВП', 'Общие выбросы CO₂']
            values = [results.iloc[0]['total_gdp'], results.iloc[0]['total_emissions']]
            colors = ['#9b59b6', '#34495e']
            
            bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_title('Общие показатели за весь период наблюдений', fontweight='bold')
            ax.set_ylabel('Значение')
            
            # ДОБАВЛЕНО: логарифмическая ось Y для этого графика
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, axis='y', which='both')
            
            # Форматируем большие числа
            for bar, value in zip(bars, values):
                if value > 1e12:
                    formatted_value = f'{value/1e12:.1f} трлн $'
                elif value > 1e9:
                    formatted_value = f'{value/1e9:.1f} млрд $'
                elif value > 1e6:
                    formatted_value = f'{value/1e6:.1f} млн т'
                else:
                    formatted_value = f'{value:,.0f}'
                
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       formatted_value, ha='center', va='bottom', fontweight='bold')
                
        except Exception as e:
            print(f"Ошибка при построении графика общих показателей: {e}")
            ax.text(0.5, 0.5, 'Ошибка данных', ha='center', va='center', transform=ax.transAxes)
    
    @staticmethod
    def _plot_correlation_placeholder(ax: plt.Axes, results: pd.DataFrame) -> None:
        """Отображает информацию о корреляции."""
        try:
            correlation_value = results.get('correlation_population_emissions', pd.Series([0.0])).iloc[0]
            
            # Интерпретация корреляции
            if abs(correlation_value) < 0.3:
                interpretation = "Слабая корреляция"
                color = "gray"
            elif abs(correlation_value) < 0.7:
                interpretation = "Умеренная корреляция"
                color = "orange"
            else:
                interpretation = "Сильная корреляция"
                color = "red" if correlation_value > 0 else "blue"
            
            ax.text(0.5, 0.7, f'Корреляция:', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14, fontweight='bold')
            ax.text(0.5, 0.5, f'{correlation_value:.3f}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=24, fontweight='bold', color=color)
            ax.text(0.5, 0.3, interpretation, ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, style='italic')
            ax.set_title('Корреляция: Население (чел) vs Выбросы CO₂ (т)', fontweight='bold')
            ax.set_facecolor('#f8f9fa')
            
            # Добавляем рамку
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(2)
                
        except Exception as e:
            print(f"Ошибка при построении графика корреляции: {e}")
            ax.text(0.5, 0.5, 'Ошибка данных', ha='center', va='center', transform=ax.transAxes)

    @staticmethod
    def plot_correlation_scatter(parquet_reader, population_col: str = 'Country.Population', 
                               emissions_col: str = 'Emissions.Production.CO2.Total') -> None:
        """Scatter plot для корреляции между населением и выбросами."""
        try:
            df = parquet_reader.read_parquet_columns([population_col, emissions_col])
            
            # Фильтруем валидные данные
            valid_data = df[
                (df[population_col] > 0) & 
                (df[emissions_col] > 0)
            ]
            
            if len(valid_data) == 0:
                print("Нет валидных данных для построения scatter plot")
                return
            
            plt.figure(figsize=(12, 8))
            
            # Scatter plot с прозрачностью
            plt.scatter(valid_data[population_col], valid_data[emissions_col], 
                       alpha=0.6, s=30, color='steelblue', edgecolors='black', linewidth=0.1)
            
            plt.xlabel('Население страны (человек)', fontsize=12, fontweight='bold')
            plt.ylabel('Выбросы CO₂ (миллионы тонн)', fontsize=12, fontweight='bold')
            plt.title('Корреляция между населением страны и выбросами CO₂\n(Scatter Plot)', 
                     fontsize=14, fontweight='bold')
            
            # ДОБАВЛЕНО: логарифмические оси X и Y для этого графика
            plt.xscale('log')
            plt.yscale('log')
            
            plt.grid(True, alpha=0.3, which='both')
            
            # Добавляем линию тренда (в логарифмическом пространстве)
            if len(valid_data) > 1:
                # Преобразуем в логарифмы для линейной регрессии
                log_pop = np.log10(valid_data[population_col])
                log_emi = np.log10(valid_data[emissions_col])
                z = np.polyfit(log_pop, log_emi, 1)
                p = np.poly1d(z)
                
                # Генерируем точки для линии тренда
                x_trend = np.logspace(np.log10(valid_data[population_col].min()),
                                     np.log10(valid_data[population_col].max()), 100)
                y_trend = 10**p(np.log10(x_trend))
                
                plt.plot(x_trend, y_trend, "r--", alpha=0.8, linewidth=2, label='Линия тренда')
            
            correlation = valid_data[population_col].corr(valid_data[emissions_col])
            
            # Блок с информацией
            info_text = f'Коэффициент корреляции: {correlation:.3f}\n'
            info_text += f'Количество наблюдений: {len(valid_data):,}\n'
            if len(valid_data) > 1:
                info_text += f'Уравнение тренда: log(y) = {z[0]:.3f}·log(x) + {z[1]:.3f}\n'
            info_text += f'*Логарифмические шкалы по обеим осям'
            
            plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                    fontfamily='monospace', verticalalignment='top')
            
            plt.legend()
            plt.tight_layout()
            plt.savefig('correlation_scatter.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Ошибка при построении scatter plot: {e}")

    @staticmethod
    def plot_variance_analysis(variance_results: pd.DataFrame, confidence_results: pd.DataFrame) -> None:
        """Анализ дисперсии с доверительными интервалами."""
        if variance_results.empty:
            print("Нет данных для анализа дисперсии")
            return
            
        try:
            # Получаем список стран из variance_results
            variance_res_country = pd.concat([
                variance_results.iloc[0]['highest_variance_countries']['country'],
                variance_results.iloc[0]['lowest_variance_countries']['country']]).tolist()

            # Фильтруем по списку стран
            sorted_results = (
                confidence_results.set_index('country')
                .loc[variance_res_country]  # выбираем в нужном порядке
                .reset_index())  # возвращаем country обратно в столбцы
            
            countries = sorted_results['country'].tolist()
            means = sorted_results['mean_emissions'].tolist()
            lowers = sorted_results['confidence_lower'].tolist()
            uppers = sorted_results['confidence_upper'].tolist()
            
            plt.figure(figsize=(12, 8))
            
            # Bar plot с доверительными интервалами
            x_pos = np.arange(len(countries))
            bars = plt.bar(x_pos, means, yerr=[np.array(means)-np.array(lowers), 
                                             np.array(uppers)-np.array(means)], 
                          capsize=5, alpha=0.7, color='coral', edgecolor='darkred')
            
            plt.title('Средние выбросы с 95% доверительными интервалами\n(Топ-3 max и топ-3 min)', 
                     fontsize=14, fontweight='bold')
            plt.ylabel('Выбросы CO₂ (тонны)')
            plt.xlabel('Страны')
            
            # ДОБАВЛЕНО: логарифмическая ось Y для этого графика
            plt.yscale('log')
            plt.grid(True, alpha=0.3, which='both')
            
            plt.xticks(x_pos, countries, rotation=45)
            
            # Добавляем значения на столбцы
            for bar, mean in zip(bars, means):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                        f'{mean:.0f} т', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('variance_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Ошибка при построении анализа дисперсии: {e}")

    @staticmethod
    def plot_temporal_analysis(temporal_results: pd.DataFrame) -> None:
        """Line plot для анализа временных рядов со скользящим средним."""
        if temporal_results.empty:
            print("Нет данных временного анализа для построения графиков")
            return
            
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            fig.suptitle('Временные ряды общих показателей ВВП и выбросов CO₂', fontsize=16, fontweight='bold')
            
            # График ВВП
            ax1.plot(temporal_results['year'], temporal_results['gdp'], 'o-', alpha=0.7, 
                    color='green', label='Общий ВВП', linewidth=2)
            ax1.plot(temporal_results['year'], temporal_results['gdp_moving_avg'], '--', 
                    color='darkgreen', label='Скользящее среднее ВВП', linewidth=2)
            ax1.set_title('Общий ВВП за период наблюдений', fontweight='bold')
            ax1.set_ylabel('ВВП ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # График выбросов
            ax2.plot(temporal_results['year'], temporal_results['emissions'], 'o-', alpha=0.7, 
                    color='red', label='Общие выбросы CO₂', linewidth=2)
            ax2.plot(temporal_results['year'], temporal_results['emissions_moving_avg'], '--', 
                    color='darkred', label='Скользящее среднее выбросов', linewidth=2)
            ax2.set_title('Общие выбросы CO₂ за период наблюдений', fontweight='bold')
            ax2.set_ylabel('Выбросы CO₂ (тонны)')
            ax2.set_xlabel('Год')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('temporal_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Ошибка при построении графиков временного анализа: {e}")