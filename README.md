# steam-success

Семестровый проект по дисциплине «Технологии и инструментарий машинного обучения»
(РТУ МИРЭА, ИМБО-02-24, Кизеев Е.М.).

Задача — по карточке игры Steam на момент релиза оценить вероятность того, что игра
наберёт не менее 50 тысяч владельцев. Данные — открытый набор
[Steam Store Games](https://www.kaggle.com/datasets/nikdavis/steam-store-games) (файл `steam.csv`).

| Тег   | Практическая работа |
|-------|---------------------|
| `pr2` | Разведочный анализ данных, проектирование системы и пайплайна |

## Как воспроизвести

```bash
poetry install
# steam.csv из набора Kaggle положить в data/raw/ (хеш проверит DVC:
# md5 61db6ef6b06166d0c100ee78b2a5244a, 5 787 977 байт, переводы строк LF)
dvc repro            # split -> validate -> eda
poetry run pytest    # модульные тесты проверок качества
```

## Структура

```
configs/params.yaml        параметры всех стадий
data/raw/steam.csv.dvc     исходный файл под DVC
src/steam_success/data     разбиение (split), схемы pandera и проверки К1–К13 (validate)
src/steam_success/eda      разведочный анализ
src/steam_success/features история издателя с зазором 12 месяцев
reports/                   quality.json, eda_summary.json, рисунки
tests/                     модульные тесты
```
