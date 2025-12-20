"""
Конфигурация пакета для установки.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="image-processing-cats",
    version="1.0.0",
    author="Красных Дарья 6404",
    author_email="dasha_krasnykh@mail.ru",
    description="Пакет для обработки изображений кошек с использованием OpenCV и асинхронного программирования",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Helliade/ImageProcessing",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "aiohttp>=3.8.0",
        "aiofiles>=0.7.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "process-cats=main:main",
        ],
    },
)