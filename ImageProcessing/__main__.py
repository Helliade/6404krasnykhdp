"""
Точка входа для запуска пакета как модуля: python -m ImageProcessing
"""

from .main import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())