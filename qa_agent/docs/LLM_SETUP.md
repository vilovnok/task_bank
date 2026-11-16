# Настройка LLM для генерации комментариев

## Обращение к GigaChat через TagMe прокси

QA Agent использует **TagMe прокси** для обращения к LLM (GigaChat), а не прямое подключение через mTLS сертификаты.

### Преимущества прокси подхода:

- ✅ Не нужны mTLS сертификаты
- ✅ Единая точка авторизации через TagMe токен
- ✅ Упрощенная конфигурация
- ✅ Поддержка множества моделей

## Конфигурация

### 1. Создайте конфиг файл

Путь по умолчанию: `~/configs/ml_config.cfg`

```ini
[DEFAULT]
oauth_token = YOUR_TAGME_TOKEN_HERE
```

### 2. Получение токена

Токен можно получить из вашего TagMe аккаунта:
1. Войдите в TagMe
2. Перейдите в настройки профиля
3. Скопируйте OAuth токен

### 3. Настройка в projects_config.json

Добавьте путь к конфигу в настройки проекта:

```json
{
  "paths": {
    "tagme_config": "~/configs/ml_config.cfg",
    "quality_script": "scripts/calc_quality_spotter.py",
    "short_rules_file": "rules/short_rules.md",
    "comment_prompt_file": "prompts/comment_prompt.md",
    "summary_prompt_file": "prompts/summary_prompt.md"
  }
}
```

## Доступные модели

Модель настраивается в `scripts/generate_comments.py`:

```python
LLM_MODEL = "GigaChat-2-Max"  # По умолчанию
```

Доступные модели через TagMe прокси:
- `GigaChat` - базовая модель
- `GigaChat-2-Max` - улучшенная модель (рекомендуется)
- `GigaChat-3-Pro` - профессиональная модель
- `GigaChat-3-Ultra` - максимальная модель
- `DeepSeek-R1` - модель с рассуждениями

## Параметры LLM

В `scripts/generate_comments.py`:

```python
LLM_MODEL = "GigaChat-2-Max"
LLM_TIMEOUT_SEC = 180  # Таймаут запроса
LLM_TEMPERATURE = 0.2  # Креативность (0.0-1.0)
MAX_CONCURRENT_REQUESTS = 5  # Максимум одновременных запросов
```

### Настройка параллелизма

По умолчанию используется **5 одновременных запросов** к LLM для ускорения обработки.

**Преимущества:**
- ⚡ Ускорение в ~5 раз при большом количестве плохих assignments
- 📊 Эффективное использование API
- 🔄 Автоматическое управление очередью запросов

**Рекомендации:**
- Для малого количества (<50) assignments: оставить 5
- Для большого количества (>500): можно увеличить до 10
- При ошибках rate-limit: уменьшить до 2-3

Изменить можно в начале `scripts/generate_comments.py`:
```python
MAX_CONCURRENT_REQUESTS = 10  # Увеличить для быстрой обработки
```

## Архитектура

```
qa_agent/generate_comments.py
    ↓ (использует)
ml_scripts/tagme_bot/utils/vsellm_client.py
    ↓ (вызывает)
TagMe Proxy API (https://tagme.sberdevices.ru/x/ai/llm/v1/chat/completions)
    ↓ (перенаправляет)
GigaChat / DeepSeek / другие LLM
```

## Формат запроса через vsellm_client

```python
from utils.vsellm_client import call_vsellm

result = call_vsellm(
    model="GigaChat-2-Max",
    user_text="Ваш промпт здесь",
    temperature=0.2,
    timeout_sec=180,
    proxy_cfg_path="~/configs/ml_config.cfg"
)

answer = result["answer"]
usage = result["usage"]  # {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
```

## Troubleshooting

### Ошибка: "Failed to load config"
- Проверьте путь к конфигу: `~/configs/ml_config.cfg`
- Убедитесь что файл существует и содержит `oauth_token`

### Ошибка: "401 Unauthorized"
- Проверьте токен в конфиге
- Токен мог истечь - получите новый из TagMe

### Ошибка: "Connection timeout"
- Увеличьте `LLM_TIMEOUT_SEC`
- Проверьте доступность TagMe прокси

### Ошибка: "Model not found"
- Проверьте что модель доступна через TagMe прокси
- Список моделей см. в `vsellm_client.py` → `TAGME_MODELS`

### Ошибка: "Rate limit exceeded" или "429 Too Many Requests"
- Уменьшите `MAX_CONCURRENT_REQUESTS` (например, до 2-3)
- Это ограничение API TagMe прокси на количество одновременных запросов

### Слишком медленная генерация
- Увеличьте `MAX_CONCURRENT_REQUESTS` (например, до 10)
- Проверьте скорость интернета
- Проверьте время отклика TagMe прокси

## Мониторинг использования токенов

Статистика токенов сохраняется в результатах:

```json
{
  "total_prompt_tokens": 15000,
  "total_completion_tokens": 5000,
  "total_tokens": 20000,
  "comments_generated": 150
}
```

В Excel файлах:
- `llm_prompt_tokens` - токены в промпте
- `llm_completion_tokens` - токены в ответе
- `llm_total_tokens` - всего токенов
- `llm_model` - использованная модель
