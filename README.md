# House Prices — ML-пайплайн

Регрессия цены дома: кросс-валидация, OOF-предсказания и готовый сабмишн.
Весь код — в `src/`, настройки — в `config.yaml`, запуск — через `main.py`.
Устроено так же, как пайплайн для Titanic, только задача — регрессия.

## Структура

```
.
├── config.yaml            все настройки: данные, признаки, модель, валидация
├── main.py                точка входа: train / predict
├── src/
│   ├── data.py            чтение csv, удаление выбросов, разбиение на фолды
│   ├── features.py        признаки House Prices + пропуски и кодирование категорий
│   ├── models.py          создание модели по имени из конфига
│   └── training.py        цикл по фолдам, метрики, сохранение результатов
├── notebooks/
│   └── 01_eda.ipynb       разведочный анализ данных
├── data/raw/              сюда кладутся train.csv и test.csv
├── outputs/               результаты запусков (создаётся автоматически)
├── RESULTS.md             журнал экспериментов
└── requirements.txt
```

## Данные

Датасет не хранится в репозитории. Скачайте его со страницы соревнования
[House Prices — Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
и положите файлы в `data/raw/`:

```
data/raw/
├── train.csv
├── test.csv
└── data_description.txt   описание всех 79 колонок — пригодится
```

Либо через Kaggle API (нужен токен в `~/.kaggle/kaggle.json`):

```bash
kaggle competitions download -c house-prices-advanced-regression-techniques -p data/raw
unzip data/raw/house-prices-advanced-regression-techniques.zip -d data/raw
```

## Запуск

```bash
pip install -r requirements.txt
python main.py train
```

Все команды запускаются из корня проекта.

Результат — папка `outputs/<дата>_<модель>/`:

```
metrics.json    метрики по фолдам и по OOF + конфиг запуска
oof.csv         out-of-fold предсказания в долларах (честная оценка качества)
submission.csv  готовый файл для Kaggle (Id, SalePrice)
models/         обученные модели по каждому фолду
```

## Эксперименты

Постоянные настройки живут в `config.yaml`, разовые удобнее задавать из командной строки:

```bash
python main.py train --set model.name=lasso --set features.encoding=onehot --set features.scale=true
python main.py train --set model.name=lasso --set features.encoding=onehot --set features.scale=true --set model.params.lasso.alpha=0.001
python main.py train --set model.name=lgbm --set model.params.lgbm.num_leaves=31
python main.py train --set features.engineering=false     # без доменных признаков
python main.py predict                                   # предсказать последним обученным запуском
python main.py predict --run-dir outputs/2026-09-26_12-00-00_lasso
```

Доступные модели: `ridge`, `lasso`, `elasticnet`, `rf`, `histgb` (только scikit-learn),
`lgbm`, `xgb`, `catboost`. Линейным нужны `features.encoding=onehot` и
`features.scale=true`, деревьям — `ordinal` без масштабирования (стоит по умолчанию).

Параметры моделей лежат в конфиге по имени модели (`model.params.lasso`,
`model.params.lgbm`, ...), поэтому при `--set model.name=...` берутся только
параметры выбранной модели.

Результаты сравнений записывайте в `RESULTS.md`.

## Результаты

Лучший результат: Lasso (onehot + scale), OOF rmse_log **0.1094** ± 0.006,
MAE ≈ 13 тыс. $. Деревья заметно хуже: histgb 0.1211, RandomForest 0.1291.
Полная таблица экспериментов — в [RESULTS.md](RESULTS.md).

## Анализ данных

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Ноутбук смотрит на распределение цены и зачем её логарифмировать, на пропуски
(и почему большинство из них — «нет гаража/подвала», а не «неизвестно»),
на связь цены с качеством, площадью и районом, на выбросы, затем показывает,
что дают признаки из `src/features.py`, и считает baseline тем же кодом,
что и `main.py`.

## Как это работает

**Метрика — RMSE по логарифму цены.** Так считает Kaggle: ошибка в 10% на доме
за 100 тыс. и за 500 тыс. весит одинаково. Поэтому модель учится на
`log1p(SalePrice)` (`target_log: true`) — через `TransformedTargetRegressor`,
так что `predict` сразу возвращает доллары и сохранённые модели самодостаточны.

**Выбросы удаляются только из train.** Два огромных дома (>4000 кв. футов),
проданных за бесценок, сильно портят линейные модели. Тест не трогается.

**NaN чаще всего значит «нет».** `PoolQC`, `GarageType`, `BsmtQual` и т.п. пусты,
когда бассейна, гаража или подвала нет. Оценки качества (`Ex`…`Po`) превращаются
в числа 5…1, а отсутствие — в 0; для категорий пропуск становится своей категорией `missing`.

**Препроцессинг обучается внутри фолда.** Импутер, энкодер и скейлер создаются заново
на каждом train-фолде (`clone` в `src/training.py`), поэтому оценка честная.

**Признаки без состояния.** `add_features` считает новые колонки только из
существующих и ничего не запоминает про выборку, поэтому применяется к train и
test сразу, без риска утечки.

**Колонки определяются по типу данных.** Добавили признак в `add_features` — он
сам попадёт в числовую или категориальную ветку препроцессора.

**Предсказания усредняются по фолдам** — в логарифмах, как и метрика.

## Как расширять

| Что нужно | Куда смотреть |
|---|---|
| новый признак | `src/features.py`, функция `add_features` |
| новая модель | `src/models.py`, функция `build_model`, + параметры в `config.yaml` |
| новая метрика | `src/training.py`, функция `compute_metrics` |
| другой датасет | `config.yaml` → `data`, и `features.engineering: false` |
