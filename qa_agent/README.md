# QA Agent - Автоматизированный контроль качества разметки

QA Agent - система автоматического мониторинга и контроля качества разметки в TagMe проектах.

## Возможности

- **Автоматическое отслеживание задач**: Обнаружение новых задач в проектах по расписанию
- **Скачивание результатов**: Автоматическое скачивание assignments через TagMe SDK
- **Расчет метрик качества**: 
  - Согласие между разметчиками (inter-rater agreement)
  - Выявление проблемных заданий
  - Статистика по разметчикам
- **Экспорт результатов**: Сохранение в JSON, CSV и Excel форматах
- **LLM-комментирование**: Генерация комментариев для проблемных заданий (опционально)
- **Итоговые отчеты**: Автоматическая генерация сводок по качеству

## Архитектура

QA Agent работает в несколько этапов (stages):

### 1. Stage "quality" (`scripts/calc_quality.py` или `calc_quality_spotter.py`)
- Скачивает assignments по задаче
- Вычисляет метрики качества разметки
- **Для проекта Spotter** (`calc_quality_spotter.py`):
  - Обрабатывает временные метки (аудиосегменты)
  - Находит правильный ответ через consensus
  - Учитывает толерантность по времени (0.15 сек)
  - Фильтрует задания с низким перекрытием/согласованностью
- Сохраняет результаты **временно** в `data/{project_id}/temp/{run_date}/`:
  - `spotter_XXXXXXXX_YYYYMMDD_HHMMSS.xlsx` - Excel с 3 листами:
    - **Data** - все данные с метриками
    - **Markers Stats** - статистика по разметчикам
    - **Summary** - общая сводка
  - `spotter_XXXXXXXX_YYYYMMDD_HHMMSS_stats.json` - статистика в JSON
  - **`spotter_XXXXXXXX_YYYYMMDD_HHMMSS_BAD_assignments.xlsx`** - только плохие assignments (score=0)
  - **`spotter_XXXXXXXX_YYYYMMDD_HHMMSS_BAD_assignments.json`** - плохие assignments для LLM обработки
  
  ⚠️ **ВАЖНО**: Данные сейчас сохраняются локально для отладки. 
  В будущем они будут сохраняться на ФИР и не храниться в проекте.

### 2. Stage "comments" (`scripts/generate_comments.py`)
- **РЕАЛИЗОВАНО** ✅
- Генерирует комментарии к плохим заданиям через LLM (GigaChat)
- **Использует TagMe прокси** для обращения к GigaChat (через `vsellm_client`)
- **Параллельная обработка**: до 5 одновременных запросов к LLM
- Требует конфиг `~/configs/ml_config.cfg` с токеном для прокси
- Использует правила из `rules/short_rules.md`
- Сохраняет результаты:
  - `*_WITH_COMMENTS.xlsx` - Excel с комментариями
  - `*_WITH_COMMENTS.json` - JSON с комментариями
- **НЕ ОТПРАВЛЯЕТ комментарии в TagMe** (только генерирует локально)

### 3. Stage "summary" (`scripts/generate_summary.py`)
- **РЕАЛИЗОВАНО** ✅
- Создает персональные сводки для разметчиков
- Группирует ошибки по marker_id
- Генерирует обратную связь через LLM (GigaChat)
- **После успешной генерации** сохраняет обработанные assignment_id в трекер
- Сохраняет результаты:
  - `*_WITH_COMMENTS_AND_SUMMARY.xlsx` - финальный Excel с двумя листами:
    - **Comments** - все комментарии
    - **Personal Summaries** - персональные сводки для разметчиков
- **НЕ ОТПРАВЛЯЕТ** сводки в TagMe (только генерирует локально)

## Отслеживание обработанных assignments

QA Agent отслеживает какие assignments уже были обработаны, чтобы не дублировать работу при повторных запусках.

### Как это работает:

1. **При первом запуске** (31 мая):
   - Скачано 1000 assignments
   - Обработано все 1000
   - Сохранено 1000 assignment_id в `state/processed_assignments/{project_id}.json`

2. **При повторном запуске** (1 июня):
   - Скачано 1500 assignments (задача продолжает размечаться)
   - Отфильтровано 1000 (уже обработаны)
   - **Обработано только 500 новых**
   - Сохранено еще 500 assignment_id

### Сброс отслеживания:

Чтобы обработать все assignments заново:
```bash
rm state/processed_assignments/{project_id}.json
```

Подробнее см. [docs/PROCESSED_ASSIGNMENTS.md](docs/PROCESSED_ASSIGNMENTS.md)

## Установка

```bash
cd ml_scripts/qa_agent

# Установка зависимостей
pip install -r requirements.txt

# Настройка конфигурации TagMe
# Создайте ~/.crowd.cfg с credentials
```

## Конфигурация

### Конфигурация проектов (`configs/projects_config.json`)

```json
{
  "projects": [
    {
      "project_id": "12345",
      "project_name": "My Annotation Project",
      "organization_id": "67890",
      "stand_name": "sigma",
      "enabled": true,
      "daily_feedback_enabled": true,
      "feedback_scope": "problem_only",
      "paths": {
        "tagme_config": "~/.crowd.cfg"
      }
    }
  ]
}
```

**Параметры проекта:**
- `project_id` - ID проекта в TagMe
- `stand_name` - стенд TagMe (cloud/sigma/sberdev)
- `enabled` - включить/выключить мониторинг проекта
- `daily_feedback_enabled` - запускать ежедневную проверку
- `feedback_scope` - режим фидбека:
  - `problem_only` - только проблемные задания
  - `all` - все задания

### Состояние отслеживания (`state/tracking_state.json`)

Автоматически управляется qa_manager. Содержит:
- Список отслеживаемых задач по проектам
- История запусков pipeline
- Статистика обработки

## Использование

### 1. Запуск полного pipeline (production mode)

```bash
python qa_manager.py --mode production --date 2024-05-31
```

Обрабатывает все включенные проекты за указанную дату.

### 2. Тестовый запуск одного скрипта

```bash
# Только расчет качества для конкретной задачи
python scripts/calc_quality.py \
  --task-id "task_12345" \
  --task-name "Test Task" \
  --config ~/.crowd.cfg \
  --data-dir ./data \
  --run-date 2024-05-31
```

### 3. Интерактивный режим

```python
import asyncio
from pathlib import Path
from crowd_sdk.tagme import TagmeClientAdvanced
from scripts import calc_quality

async def test_quality():
    client = TagmeClientAdvanced("~/.crowd.cfg")
    
    result = await calc_quality.run(
        client=client,
        task_id="your_task_id",
        task_name="Test Task",
        run_date="2024-05-31",
        data_dir=Path("./data")
    )
    
    print(result)

asyncio.run(test_quality())
```

## Структура данных

### Формат результатов качества

**`quality_metrics.json`:**
```json
{
  "task_id": "task_12345",
  "calculated_at": "2024-05-31T10:00:00",
  "summary": {
    "total_assignments": 150,
    "unique_tasks": 50,
    "average_overlap": 3.0,
    "average_agreement": 0.92,
    "problem_tasks_count": 4,
    "problem_tasks_rate": 0.08
  },
  "problem_tasks": [
    {
      "task_id": "internal_task_1",
      "overlap": 3,
      "average_agreement": 0.33,
      "field_metrics": {...}
    }
  ],
  "marker_quality": {
    "marker_001": {
      "total_assignments": 30,
      "agreements": 28,
      "disagreements": 2,
      "accuracy": 0.93
    }
  }
}
```

### Excel отчет (`quality_report.xlsx`)

Содержит несколько листов:
1. **Summary** - общие метрики по задаче
2. **Marker Quality** - статистика по разметчикам
3. **Problem Tasks** - список проблемных заданий
4. **Raw Assignments** - исходные данные (если не слишком большой)

## Метрики качества

### Agreement Rate (Процент согласия)
Доля разметчиков, которые дали одинаковый ответ на задание.

- `agreement = 1.0` (100%) - полное согласие
- `agreement < 0.5` (50%) - задание считается проблемным

### Marker Accuracy (Точность разметчика)
Процент совпадений ответов разметчика с мажоритарным ответом.

### Problem Tasks (Проблемные задания)
Задания с низким согласием между разметчиками (по умолчанию < 50%).

## Расширение функциональности

### Добавление новых метрик

Редактируйте `utils/quality_metrics.py`:

```python
def calculate_custom_metric(assignments):
    # Ваша логика расчета
    return metric_value
```

### Кастомизация отчетов

Модифицируйте метод `_create_quality_excel_report` в `scripts/calc_quality.py`.

## Troubleshooting

### Ошибка: "No assignments found"
- Проверьте, что задача существует и имеет assignments
- Убедитесь, что credentials в `~/.crowd.cfg` корректны
- Проверьте `stand_name` в конфигурации проекта

### Ошибка при сохранении Excel
- Убедитесь, что установлен `openpyxl`: `pip install openpyxl`
- Проверьте права на запись в директорию `data/`

### Медленная работа
- Используйте фильтрацию заданий по датам
- Ограничьте количество одновременно обрабатываемых проектов
- Увеличьте интервал между запросами к API

## TODO

- [ ] Реализовать stage "comments" с LLM
- [ ] Реализовать stage "summary" 
- [ ] Добавить Cohen's Kappa для пар разметчиков
- [ ] Интеграция с Slack/Telegram для уведомлений
- [ ] Dashboard для визуализации метрик
- [ ] Поддержка различных типов разметки (NER, classification, etc.)

## Структура проекта

```
qa_agent/
├── README.md                     # Документация
├── requirements.txt              # Зависимости
├── qa_manager.py                 # Главный менеджер pipeline
├── configs/
│   └── projects_config.json      # Конфигурация проектов
├── state/
│   └── tracking_state.json       # Состояние отслеживания
├── scripts/
│   ├── calc_quality.py           # Stage 1: расчет качества
│   ├── generate_comments.py      # Stage 2: генерация комментариев (TODO)
│   └── generate_summary.py       # Stage 3: итоговая сводка (TODO)
├── utils/
│   ├── utils.py                  # Общие утилиты
│   └── quality_metrics.py        # Функции расчета метрик
├── prompts/
│   ├── comment_prompt.md         # Prompt для комментариев
│   └── summary_prompt.md         # Prompt для сводки
├── rules/
│   └── short_rules.md            # Правила разметки
├── data/                         # Результаты (игнорируется git)
│   └── {project_id}/
│       └── {task_id}/
│           └── {run_date}/
│               ├── assignments.json
│               ├── assignments.csv
│               ├── assignments.xlsx
│               ├── quality_metrics.json
│               ├── quality_report.xlsx
│               └── problem_tasks.json
└── docs/
    └── CROWD_SDK.md              # Документация Crowd SDK
```
