# Быстрый старт

## 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

## 2. Применение миграций

```bash
alembic upgrade head
```

## 3. Создание администратора

```bash
python create_admin.py
```

Введите имя пользователя: `admin`
Введите пароль: `admin123`

## 4. Запуск сервера

```bash
uvicorn app.main:app --reload
```

## 5. Тестирование API

### Вариант 1: Swagger UI

Откройте в браузере: http://localhost:8000/docs

### Вариант 2: cURL команды

#### Получение токена

```bash
curl -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123"
```

#### Детекция аномалий в логах

```bash
curl -X POST "http://localhost:8000/forward" \
  -H "Content-Type: application/json" \
  -d '{
    "logs": [
      {
        "message": "Receiving block blk_-1608999687919862906",
        "component": "DataNode$DataXceiver",
        "level": "INFO"
      }
    ]
  }'
```

#### Использование тестовых файлов

```bash
# Нормальные логи
cat test_logs/normal_logs.json | curl -X POST "http://localhost:8000/forward" \
  -H "Content-Type: application/json" -d @-

# Аномальные логи
cat test_logs/anomaly_logs.json | curl -X POST "http://localhost:8000/forward" \
  -H "Content-Type: application/json" -d @-
```

#### Просмотр истории

```bash
curl -X GET "http://localhost:8000/history" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

#### Просмотр статистики

```bash
curl -X GET "http://localhost:8000/stats" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## Порты

- API: http://localhost:8000
- Документация: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Переменные окружения

Настройте в файле `.env`:

```
DATABASE_URL=sqlite+aiosqlite:///./ml_service.db
SECRET_KEY=your-secret-key-here
ADMIN_TOKEN=your-admin-token-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=30
```

## Docker Compose

```bash
# Запуск
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановка
docker-compose down
```

## Готово!

Сервис запущен и готов к использованию.

### Ожидаемые результаты

**Нормальные логи** (без аномалий):
```json
{
  "score": -0.58,
  "is_anomaly": false,
  "threshold": -0.58,
  "num_events": 8
}
```

**Аномальные логи** (с ошибками):
```json
{
  "score": -0.65,
  "is_anomaly": true,
  "threshold": -0.58,
  "num_events": 8
}
```
