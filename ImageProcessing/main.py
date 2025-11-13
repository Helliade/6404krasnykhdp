"""
main.py

Лабораторная работа No2 - Обработка изображений животных через API.

Программа скачивает изображения кошек, обрабатывает их и сохраняет результаты.
"""

import sys
import os
#from decorators import timer_decorator
#import inspect

# Добавляем корневую директорию в путь для импортов
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from implementation.api_processor import CatImageProcessor

def main():
    """Основная функция программы."""
    try:
        # Инициализация процессора
        processor = CatImageProcessor()

        #CatImageProcessor._url_to_array = timer_decorator(CatImageProcessor._url_to_array)
        #print(inspect.getsource(CatImageProcessor._url_to_array))
        
        # Запрос количества изображений у пользователя
        while True:
            try:
                limit = int(input("Введите количество изображений для обработки (1-10): "))
                if 1 <= limit <= 10:
                    break
                else:
                    print("Пожалуйста, введите число от 1 до 10")
            except ValueError:
                print("Пожалуйста, введите корректное число")
        
        # Основной процесс
        print("=" * 50)
        cat_images = processor.download_images(limit)
        
        print("=" * 50)
        processor.process_and_save_images(cat_images)
        
        print("=" * 50)
        processor.demonstrate_operations(cat_images)
        
        print("=" * 50)
        print("Программа успешно завершена!")
        
        # Вывод информации о скачанных изображениях
        print("\nСкачанные изображения:")
        for i, img in enumerate(cat_images, 1):
            print(f"{i}. {img}")
            
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    