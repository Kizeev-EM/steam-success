# steam-success

Семестровый проект по дисциплине «Технологии и инструментарий машинного обучения»
(РТУ МИРЭА, ИМБО-02-24, Кизеев Е.М.).

Задача — по карточке игры Steam на момент релиза оценить вероятность того, что игра
наберёт не менее 50 тысяч владельцев. Данные — открытый набор
[Steam Store Games](https://www.kaggle.com/datasets/nikdavis/steam-store-games) (файл `steam.csv`).

| Тег   | Практическая работа |
|-------|---------------------|
| `pr2` | Разведочный анализ данных, проектирование системы и пайплайна |
| `pr3` | Обучение и сравнение моделей, выбор и сохранение итоговой модели |

## Как воспроизвести

```bash
poetry install
# steam.csv из набора Kaggle положить в data/raw/ (хеш проверит DVC:
# md5 61db6ef6b06166d0c100ee78b2a5244a, 5 787 977 байт, переводы строк LF)
dvc repro            # split -> validate -> eda -> features -> experiments -> tune -> train
poetry run pytest    # модульные тесты
```

Полный прогон занимает около 15 минут, из них 12 — подбор гиперпараметров (стадия `tune`).
Эксперименты записываются в локальную базу MLflow `mlflow.db`; посмотреть их можно командой
`mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Конвейер

| Стадия        | Что делает | Результат |
|---------------|------------|-----------|
| `split`       | Разбиение по дате: исходная часть до 30.04.2017 и 4 квартальные порции | `data/interim`, `data/incoming` |
| `validate`    | Проверки качества К1–К13 (pandera) | `reports/quality.json` |
| `eda`         | Разведочный анализ, гипотезы Г1–Г5 | `reports/eda_summary.json` |
| `features`    | Любой файл карточек -> таблица признаков, проверка повторяемости | `data/processed`, `reports/features_check.json` |
| `experiments` | Базовые решения и модели пяти семейств, временная кросс-валидация | `reports/experiments.csv` |
| `tune`        | Подбор гиперпараметров LightGBM, случайного леса и логистической регрессии | `reports/tuning_*.csv` |
| `train`       | Выбор по правилу, обучение и сохранение итоговой модели, регистрация в MLflow | `models/model_v1.joblib`, `models/model_v1.json` |

## Использование модели

```python
import joblib
from steam_success.features.build import prepare_games

model = joblib.load("models/model_v1.joblib")
games, _, _ = prepare_games("data/incoming/batch_01.csv")  # проверка, очистка, история издателя
proba = model.predict_proba(games)[:, 1]                     # вероятность успеха
```

## Структура

```
configs/params.yaml        параметры всех стадий
data/raw/steam.csv.dvc     исходный файл под DVC
src/steam_success/data     разбиение (split), схемы pandera и проверки К1–К13 (validate)
src/steam_success/eda      разведочный анализ
src/steam_success/features очистка, история издателя с зазором 12 месяцев, CardFeaturizer
src/steam_success/models   схема валидации, модели, эксперименты, подбор, обучение
reports/                   таблицы экспериментов, отчёты о качестве, рисунки
models/model_v1.json       описание итоговой модели (сама модель — под DVC)
tests/                     модульные тесты
```
