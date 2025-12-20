# test_import.py
import sys
sys.path.insert(0, '.')  # добавляем текущую директорию в путь

try:
    import ImageProcessing
    print("✅ Успешный импорт ImageProcessing")
    print("Путь:", ImageProcessing.__file__)
except ImportError as e:
    print("❌ Ошибка импорта:", e)