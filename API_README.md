# ML Service API

ML-сервис на базе FastAPI для детекции аномалий в логах с использованием модели Isolation Forest.

## Возможности

- POST `/forward` - детекция аномалий в последовательности логов (Isolation Forest)
- GET `/history` - просмотр истории запросов (требует JWT авторизацию администратора)
- DELETE `/history` - удаление истории запросов (требует admin token в заголовке)
- GET `/stats` - статистика запросов с квантилями и характеристиками (требует JWT авторизацию администратора)
- JWT авторизация с ролями (пользователь/администратор)
- Миграции базы данных через Alembic
- Асинхронная работа с БД (SQLAlchemy + aiosqlite)

## Структура проекта

```
yearprojectmlapi/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Основное приложение с endpoints
│   ├── config.py               # Конфигурация приложения
│   ├── database.py             # Настройка БД
│   ├── models.py               # SQLAlchemy модели
│   ├── schemas.py              # Pydantic схемы
│   ├── auth.py                 # JWT авторизация
│   └── ml_model.py             # ML модель (Isolation Forest)
├── models/
│   └── isolation_forest.joblib # Обученная модель
├── test_logs/                  # Тестовые файлы с логами
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
├── .env.example
├── .env
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Установка

### 1. Клонирование репозитория

```bash
cd yearprojectmlapi
```

### 2. Создание виртуального окружения

```bash
python -m venv venv
source venv/bin/activate  # На Linux/Mac
# или
venv\Scripts\activate  # На Windows
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Настройка переменных окружения

Скопируйте `.env.example` в `.env` и измените значения при необходимости:

```bash
cp .env.example .env
```

Содержимое `.env`:
```
DATABASE_URL=sqlite+aiosqlite:///./ml_service.db
SECRET_KEY=your-secret-key-here-change-in-production
ADMIN_TOKEN=your-admin-token-here-change-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=30
```

### 5. Применение миграций базы данных

```bash
alembic upgrade head
```

## Запуск сервера

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Сервер будет доступен по адресу: `http://localhost:8000`

Автоматическая документация API:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Использование API

### 1. Регистрация пользователя

```bash
curl -X POST "http://localhost:8000/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "admin123",
    "is_admin": true
  }'
```

### 2. Получение JWT токена

```bash
curl -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123"
```

Ответ:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

Сохраните `access_token` для последующих запросов.

### 3. POST /forward - Детекция аномалий в логах

```bash
curl -X POST "http://localhost:8000/forward" \
  -H "Content-Type: application/json" \
  -d '{
    "logs": [
      {
        "message": "Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106",
        "component": "DataNode$DataXceiver",
        "level": "INFO"
      },
      {
        "message": "Received block blk_-1608999687919862906 of size 67108864",
        "component": "DataNode$DataXceiver",
        "level": "INFO"
      }
    ]
  }'
```

Ответ (нормальные логи):
```json
{
  "score": -0.5814839173818528,
  "is_anomaly": false,
  "threshold": -0.5827027071289144,
  "num_events": 2
}
```

Ответ (аномальные логи):
```json
{
  "score": -0.6472622282626409,
  "is_anomaly": true,
  "threshold": -0.5827027071289144,
  "num_events": 8
}
```

### 4. GET /history - Просмотр истории запросов

Требует JWT авторизацию с правами администратора:

```bash
curl -X GET "http://localhost:8000/history" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

Ответ:
```json
{
  "total": 10,
  "items": [
    {
      "id": 1,
      "request_type": "log_anomaly_detection",
      "processing_time": 0.023,
      "input_data_size": 8,
      "status_code": 200,
      "result": "...",
      "error_message": null,
      "created_at": "2025-12-25T10:30:00"
    }
  ]
}
```

### 5. DELETE /history - Удаление истории запросов

Требует admin token в заголовке Authorization:

```bash
curl -X DELETE "http://localhost:8000/history" \
  -H "Authorization: Bearer admin_secret_token_12345"
```

### 6. GET /stats - Статистика запросов

Требует JWT авторизацию с правами администратора:

```bash
curl -X GET "http://localhost:8000/stats" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

Ответ:
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

## Формат входных данных

### Структура лога

```json
{
  "message": "Текст лог-сообщения",
  "component": "Компонент системы (опционально)",
  "level": "Уровень логирования (опционально)"
}
```

### Пример запроса

```json
{
  "logs": [
    {
      "message": "Receiving block blk_-1608999687919862906",
      "component": "DataNode$DataXceiver",
      "level": "INFO"
    },
    {
      "message": "Exception in receiveBlock java.io.IOException",
      "component": "DataNode$DataXceiver",
      "level": "ERROR"
    }
  ]
}
```

## Коды ответов

- `200` - Успешная обработка запроса
- `201` - Успешное создание ресурса
- `204` - Успешное удаление (без тела ответа)
- `400` - Неверный формат запроса
- `401` - Не авторизован
- `403` - Доступ запрещен
- `500` - Внутренняя ошибка сервера
- `503` - Модель не загружена

## Миграции базы данных

### Создание новой миграции

```bash
alembic revision --autogenerate -m "description of changes"
```

### Применение миграций

```bash
alembic upgrade head
```

### Откат последней миграции

```bash
alembic downgrade -1
```

### Просмотр истории миграций

```bash
alembic history
```

## Разработка

### Запуск в режиме разработки

```bash
uvicorn app.main:app --reload
```

### Тестирование endpoints

Используйте Swagger UI по адресу `http://localhost:8000/docs` для интерактивного тестирования всех endpoints.

## Технологии

- **FastAPI** - веб-фреймворк для построения API
- **SQLAlchemy** - ORM для работы с базой данных
- **Alembic** - инструмент миграций базы данных
- **scikit-learn** - Isolation Forest модель для детекции аномалий
- **JWT (python-jose)** - авторизация и аутентификация
- **Pydantic** - валидация данных
- **aiosqlite** - асинхронный драйвер для SQLite

## Безопасность

- Пароли хешируются с использованием bcrypt
- JWT токены с настраиваемым временем жизни
- Разделение прав доступа (пользователь/администратор)
- Валидация входных данных через Pydantic
- Защищенные эндпоинты для административных операций

## Лицензия

MIT
