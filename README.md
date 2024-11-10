# ElectrocorticogramClassification
Решение команды ThreeNearestNeighbours (MIPT, MISIS)

## Запуск на Windows
Для запуска достаточно распаковать архив, перейти во вложенную папку и запустить RUN.BAT.

Если по каким-то причинам проргамма не запустилась, достаточно сделать следующее:
1. Установить в систему [Python 3.13](https://www.python.org/downloads/release/python-3130/)
2. Скачать этот [репозиторий](https://github.com/Paspasuy/ElectrocorticogramClassification) 
3. Открыть эту папку в командной строке (открыть ее в проводнике нажать правой кнопкой мыши, затем нажать соотв. пункт меню)
4. Ввести команду `python -m pip install -r requirements.txt`
5. Запустить программу: `python main.py`

## Запуск на Linux
Инструкция выше работает и под Linux (Важно: в этом случае советуем заменить зависимость `torch` на `torch-cpu`, т.к. эта библиотека меньше весит. Уберите `torch` из requirements, и установите вручную: `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`)


![Иллюстрация](screenshots/scr1.png)