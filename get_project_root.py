import os
from pathlib import Path

# Автоматически находим корень проекта (папку 'dnacsv')
def get_project_root():
    return next(p for p in Path(__file__).resolve().parents if p.name == "dnasegnercrosstalk")


import os

def get_project_root():
    current_path = os.path.abspath(__file__)
    while True:
        current_path, folder = os.path.split(current_path)
        if folder == "dnasegnercrosstalk":
            return os.path.join(current_path, folder)
        if folder == "":
            raise RuntimeError("❌ Не удалось найти папку 'dnasegnercrosstalk' вверх по дереву директорий")

import os
import sys

def get_project_root():
    if hasattr(sys, '_getframe'):
        try:
            current_path = os.path.abspath(__file__)
        except NameError:
            # Если __file__ не определён (например, в Jupyter), берём текущую рабочую директорию
            current_path = os.getcwd()
    else:
        current_path = os.getcwd()

    while True:
        current_path, folder = os.path.split(current_path)
        if folder == "dnasegnercrosstalk":
            return os.path.join(current_path, folder)
        if folder == "":
            raise RuntimeError("❌ Не удалось найти папку 'dnasegnercrosstalk' вверх по дереву директорий")