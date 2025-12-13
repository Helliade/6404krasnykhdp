import sys
import os
import asyncio
import time

# Добавляем корневую директорию в путь для импортов
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from implementation.api_processor import AsyncCatImageProcessor

def get_image_count() -> int:
    """Запрашивает у пользователя количество изображений."""
    while True:
        try:
            limit = int(input("Введите количество изображений для обработки (1-10): "))
            if 1 <= limit <= 10:
                return limit
            else:
                print("Пожалуйста, введите число от 1 до 10")
        except ValueError:
            print("Пожалуйста, введите корректное число")

def get_worker_count() -> int:
    """Запрашивает у пользователя количество воркеров."""
    import os
    cpu_count = os.cpu_count() or 2
    
    while True:
        try:
            workers = input(f"Введите количество воркеров (1-{cpu_count}, по умолчанию {cpu_count}): ").strip()
            
            if not workers:  # Если пустая строка - используем значение по умолчанию
                return cpu_count
            
            workers = int(workers)
            if 1 <= workers <= cpu_count:
                return workers
            else:
                print(f"Пожалуйста, введите число от 1 до {cpu_count}")
        except ValueError:
            print("Пожалуйста, введите корректное число")

async def main():
    """Основная функция с пользовательским вводом."""
    
    # Запрос параметров у пользователя
    print("НАСТРОЙКА ПАРАМЕТРОВ ОБРАБОТКИ")
    print("-" * 40)
    
    limit = get_image_count()
    workers = get_worker_count()
    
    print("\n" + "ЗАПУСК ПАЙПЛАЙНА")
    print("-" * 80)
    
    start_time = time.time()
    
    async with AsyncCatImageProcessor(max_workers=workers) as processor: # + создается сессия для работы с сервером (100 соед, 5 минут)
        print(f"Начало обработки {limit} изображений...")
        
        results = []
        count = 0
        
        async for img_index in processor.processing_pipeline(limit=limit):
            count += 1
            results.append(img_index)
            print(f"  ✅ Изображение {img_index} обработано ({count}/{limit})")
    
    total_time = time.time() - start_time
    
    print(f"\n" + "-" * 80)
    print("РЕЗУЛЬТАТЫ ОБРАБОТКИ:")
    print("-" * 80)
    print(f"Общее время: {total_time:.2f} секунд")
    print(f"Изображений обработано: {len(results)}")
    print(f"Воркеров использовано: {workers}")
    print(f"Порядок обработки: {results}")
    print(f"Среднее время на изображение: {total_time/len(results):.2f} сек")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())