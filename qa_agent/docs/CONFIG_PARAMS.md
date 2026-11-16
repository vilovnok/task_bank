# Параметры конфигурации проектов

## feedback_scope

Определяет режим обработки разметчиков:

- **`"all"`** - обрабатывать всех разметчиков (используйте для Spotter проекта)
- **`"problem_only"`** - обрабатывать только проблемных разметчиков (низкое качество)

## quality_script

Путь к скрипту расчета качества (относительно директории qa_agent):

- **`"scripts/calc_quality.py"`** - универсальный скрипт для любых типов разметки
- **`"scripts/calc_quality_spotter.py"`** - специализированный скрипт для проекта "Выделение споттерных слов на аудио"
  - Обрабатывает временные метки
  - Использует consensus для определения правильного ответа
  - Учитывает толерантность по времени (0.15 сек)

## Пример конфигурации для проекта Spotter

```json
{
  "enabled": true,
  "project_id": "d8f48f67-8632-4201-9313-fde85732bba9",
  "project_name": "Выделение споттерных слов на аудио",
  "stand_name": "cloud",
  "organization_id": "1bf17a92-158c-4ff6-8e16-83a51386ab09",
  
  "tracking_start_date": "2026-05-26",
  
  "daily_feedback_enabled": true,
  "feedback_scope": "all",
  
  "paths": {
    "quality_script": "scripts/calc_quality_spotter.py",
    "short_rules_file": "rules/short_rules.md",
    "comment_prompt_file": "prompts/comment_prompt.md",
    "summary_prompt_file": "prompts/summary_prompt.md"
  },
  
  "task_selection": {
    "include_active_tasks": true,
    "include_done_tasks": true,
    "skip_task_kinds": ["training", "exam", "education", "qualification"]
  }
}
```

## Важные моменты

### Временное хранение данных

⚠️ **СЕЙЧАС**: Результаты сохраняются локально в `data/{project_id}/temp/{run_date}/` для отладки

📦 **В БУДУЩЕМ**: Данные будут сохраняться на ФИР и не будут храниться в проекте

### Отправка комментариев

🚫 **СЕЙЧАС**: Комментарии НЕ отправляются в TagMe (stages 2-3 не реализованы)

✅ **В БУДУЩЕМ**: Будет реализована отправка комментариев через LLM
