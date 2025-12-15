import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from typing import Generator
import time

class DataReader:
    """Класс для чтения данных."""
    
    def __init__(self, csv_path: str, parquet_path: str = "data/global_emissions.parquet"):
        self.csv_path = csv_path
        self.parquet_path = parquet_path
        self._ensure_parquet_file()
    
    def _ensure_parquet_file(self) -> None:
        """Создает parquet-файл если он не существует."""
        if not os.path.exists(self.parquet_path):
            os.makedirs(os.path.dirname(self.parquet_path), exist_ok=True)
            print("Создание parquet-файла...")
            self._convert_csv_to_parquet()
    
    def _convert_csv_to_parquet(self) -> None:
        """Конвертирует CSV в Parquet формат."""
        # Читаем весь CSV для конвертации
        df = pd.read_csv(self.csv_path)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, self.parquet_path)
    
    def read_csv_chunks(self, chunksize: int = 10000) -> Generator[pd.DataFrame, None, None]: #элемент, принимаемое значение, возвращаемое значение
        """Генератор для чтения CSV файла по частям."""
        print("Чтение CSV файла по частям...")
        for chunk in pd.read_csv(self.csv_path, chunksize=chunksize):
            yield chunk
    
    def read_parquet_full(self) -> pd.DataFrame:
        """Чтение всего parquet-файла."""
        print("Чтение всего parquet-файла...")
        return pq.read_table(self.parquet_path)#.to_pandas()
    
    def read_parquet_columns(self, columns: list) -> pd.DataFrame:
        """Чтение только определенных столбцов из parquet-файла."""
        print(f"Чтение столбцов {columns} из parquet-файла...")
        return pq.read_table(self.parquet_path, columns=columns).to_pandas()
    
    def compare_read_speed(self) -> dict:
        """Сравнение скорости чтения CSV и Parquet."""
        print("\n=== Сравнение скорости чтения ===")
        
        # Чтение CSV
        start_time = time.time()
        df_csv = pd.read_csv(self.csv_path)
        csv_time = time.time() - start_time
        
        # Чтение Parquet
        start_time = time.time()
        df_parquet = self.read_parquet_full()
        parquet_time = time.time() - start_time
        
        print(f"CSV время чтения: {csv_time:.2f} сек")
        print(f"Parquet время чтения: {parquet_time:.2f} сек")
        print(f"Parquet быстрее в {csv_time/parquet_time:.2f} раз")