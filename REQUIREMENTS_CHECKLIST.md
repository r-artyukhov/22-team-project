# Чек-лист требований проекта

## Основные требования

### POST /forward endpoint

**Требование:** Route типа POST на /forward для детекции аномалий в логах

**Реализация:**
- Принимает JSON с массивом лог-записей
- Файл: `app/main.py:81-129`
- Использует модель Isolation Forest для детекции аномалий
- Возвращает score, is_anomaly, threshold, num_events

**Пример использования:**
```bash
curl -X POST http://localhost:8000/forward \
  -H "Content-Type: application/json" \
  -d '{
    "logs": [
      {"message": "Receiving block blk_123", "component": "DataNode", "level": "INFO"}
    ]
  }'
```

**Формат ответа:**
```json
{
  "score": -0.58,
  "is_anomaly": false,
  "threshold": -0.58,
  "num_events": 1
}
```

### Обработка ошибок

**Требование:** Правильные коды ответов для разных ситуаций

**Реализация:**
- **422 Validation Error** - неверный формат запроса (пустой список логов)
- **503 Service Unavailable** - модель не загружена
- **500 Internal Server Error** - ошибка обработки
- **200 OK** - успешная обработка

---

## GET /history

**Требование:** Показывать историю всех запросов из базы данных

**Реализация:**
- Endpoint: `app/main.py:132-145`
- Модель БД: `app/models.py:RequestHistory`
- Хранит:
  - Тип запроса (log_anomaly_detection)
  - Время обработки
  - Количество событий (input_data_size)
  - Код статуса
  - Результат
  - Сообщение об ошибке
  - Время создания
- Требует JWT авторизацию администратора
- Схема ответа: `app/schemas.py:HistoryResponse`

**Пример использования:**
```bash
curl -X GET http://localhost:8000/history \
  -H "Authorization: Bearer JWT_TOKEN"
```

---

## DELETE /history

**Требование:** Удаление истории с подтверждающим токеном

**Реализация:**
- Endpoint: `app/main.py:148-155`
- Требует admin token в заголовке `Authorization`
- Верификация токена: `app/auth.py:verify_admin_token`
- Токен настраивается через `.env` файл (`ADMIN_TOKEN`)
- Возвращает 204 No Content при успехе
- Возвращает 401/403 при неверном токене

**Пример использования:**
```bash
curl -X DELETE http://localhost:8000/history \
  -H "Authorization: Bearer admin_secret_token_12345"
```

---

## GET /stats

**Требование:** Статистика запросов с метриками

**Реализация:**
- Endpoint: `app/main.py:158-201`
- **Время обработки**:
  - Среднее (mean)
  - Медиана (50-й перцентиль)
  - 95-й перцентиль
  - 99-й перцентиль

- **Характеристики входных запросов**:
  - Средний размер данных (количество событий)

- Использует NumPy для расчета квантилей
- Требует JWT авторизацию администратора
- Схема ответа: `app/schemas.py:StatsResponse`

**Пример использования:**
```bash
curl -X GET http://localhost:8000/stats \
  -H "Authorization: Bearer JWT_TOKEN"
```

**Пример ответа:**
```json
{
  "total_requests": 100,
  "mean_processing_time": 0.025,
  "median_processing_time": 0.023,
  "percentile_95_processing_time": 0.045,
  "percentile_99_processing_time": 0.089,
  "average_input_size": 8.5
}
```

---

## JWT авторизация

**Требование:** Механизм авторизации при помощи JWT с ролями

**Реализация:**
- **JWT токены**:
  - Создание: `app/auth.py:create_access_token`
  - Валидация: `app/auth.py:get_current_user`
  - Использует `python-jose` для работы с JWT
  - Настраиваемое время жизни (default: 30 минут)

- **Роли пользователей**:
  - User (обычный пользователь)
  - Admin (администратор)
  - Модель: `app/models.py:User`
  - Поле `is_admin` в таблице users

- **Endpoints с авторизацией**:
  - `POST /register` - регистрация пользователей
  - `POST /token` - получение JWT токена
  - `GET /history` - только для администраторов
  - `GET /stats` - только для администраторов

- **Хеширование паролей**:
  - Использует bcrypt
  - Файл: `app/auth.py:get_password_hash`

**Пример использования:**
```bash
# Регистрация администратора
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123", "is_admin": true}'

# Получение токена
curl -X POST http://localhost:8000/token \
  -d "username=admin&password=admin123"

# Использование токена
curl -X GET http://localhost:8000/history \
  -H "Authorization: Bearer TOKEN"
```

---

## Alembic миграции

**Требование:** Механизм миграций базы данных при помощи alembic

**Реализация:**
- **Конфигурация Alembic**:
  - Файл: `alembic.ini`
  - Environment: `alembic/env.py`
  - Template: `alembic/script.py.mako`

- **Миграции**:
  - Папка: `alembic/versions/`
  - Начальная миграция: `alembic/versions/001_initial_migration.py`
  - Создает таблицы: `users`, `request_history`
  - Создает индексы

- **Асинхронная поддержка**:
  - Использует `async_engine_from_config`
  - Поддержка SQLAlchemy 2.0

**Команды:**
```bash
# Применение миграций
alembic upgrade head

# Создание новой миграции
alembic revision --autogenerate -m "description"

# Откат миграции
alembic downgrade -1

# История миграций
alembic history
```

---

## ML модель

- **Isolation Forest** для детекции аномалий в логах
- **Preprocessing pipeline**:
  - Нормализация сообщений (IP, hex, paths, numbers, block IDs)
  - Токенизация с component и level
  - TF-IDF векторизация
- **Threshold-based классификация**
- **Файл модели**: `models/isolation_forest.joblib`

---

## Структура файлов

```
yearprojectmlapi/
├── app/
│   ├── __init__.py             # Инициализация пакета
│   ├── main.py                 # Основное приложение
│   ├── config.py               # Конфигурация
│   ├── database.py             # База данных
│   ├── models.py               # ORM модели
│   ├── schemas.py              # Pydantic схемы
│   ├── auth.py                 # JWT авторизация
│   └── ml_model.py             # ML модель (Isolation Forest)
├── models/
│   └── isolation_forest.joblib # Обученная модель
├── test_logs/                  # Тестовые файлы
│   ├── normal_logs.json
│   ├── normal_logs_2.json
│   ├── anomaly_logs.json
│   └── anomaly_logs_2.json
├── alembic/
│   ├── versions/
│   │   └── 001_initial_migration.py
│   ├── env.py
│   └── script.py.mako
├── alembic.ini
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── create_admin.py
├── README.md
├── QUICKSTART.md
├── EXAMPLES.md
├── ARCHITECTURE.md
├── TESTING.md
├── DEPLOYMENT.md
└── REQUIREMENTS_CHECKLIST.md
```

## Команды для проверки

```bash
# Установка
pip install -r requirements.txt

# Миграции
alembic upgrade head

# Локальный запуск
uvicorn app.main:app --reload

# Docker запуск
docker-compose up -d

# Тестирование
cat test_logs/normal_logs.json | curl -X POST http://localhost:8000/forward \
  -H "Content-Type: application/json" -d @-

# Документация
open http://localhost:8000/docs
```
