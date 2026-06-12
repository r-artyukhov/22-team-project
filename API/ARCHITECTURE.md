# Архитектура проекта

## Обзор

ML Service API - это асинхронный веб-сервис для детекции аномалий в логах с использованием модели Isolation Forest. Сервис построен на FastAPI и поддерживает JWT авторизацию, хранение истории запросов и сбор статистики.

## Компоненты системы

### 1. Web Layer (FastAPI)

**app/main.py** - основной файл приложения

Endpoints:
- `POST /register` - регистрация пользователей
- `POST /token` - получение JWT токена
- `POST /forward` - детекция аномалий в логах
- `GET /history` - просмотр истории запросов (admin)
- `DELETE /history` - удаление истории (admin token)
- `GET /stats` - статистика запросов (admin)

### 2. Database Layer

**app/database.py** - конфигурация базы данных

- Асинхронный движок SQLAlchemy
- SQLite с драйвером aiosqlite
- Session management через dependency injection
- Автоматическая инициализация таблиц

**app/models.py** - ORM модели

```
RequestHistory:
  - id
  - request_type
  - processing_time
  - input_data_size
  - status_code
  - result
  - error_message
  - created_at

User:
  - id
  - username (unique)
  - hashed_password
  - is_admin
  - created_at
```

### 3. Validation Layer

**app/schemas.py** - Pydantic схемы

- `LogEntry` - структура одной лог-записи
- `LogSequenceRequest` - запрос с последовательностью логов
- `AnomalyResponse` - ответ с результатом детекции
- Валидация входных данных
- Сериализация/десериализация
- Автоматическая генерация документации

### 4. Authentication & Authorization

**app/auth.py** - система авторизации

Функции:
- Хеширование паролей (bcrypt)
- Создание JWT токенов
- Валидация токенов
- Role-based access control (RBAC)
- Проверка admin токена

Роли:
- **User** - обычный пользователь
- **Admin** - администратор (доступ к /history и /stats)

### 5. ML Layer

**app/ml_model.py** - ML модель

- Isolation Forest для детекции аномалий
- Загрузка модели из joblib файла
- Нормализация лог-сообщений:
  - IP адреса → `<ip>`
  - Hex значения → `<hex>`
  - Пути → `<path>`
  - Числа → `<num>`
  - Block ID → `<blk>`
- TF-IDF векторизация
- Threshold-based классификация

**models/isolation_forest.joblib** - обученная модель

Содержит:
- `model` - обученный Isolation Forest
- `vectorizer` - TF-IDF векторизатор
- `threshold` - порог для классификации аномалий

### 6. Configuration

**app/config.py** - управление конфигурацией

Использует `pydantic-settings` для загрузки переменных окружения:
- DATABASE_URL
- SECRET_KEY
- ADMIN_TOKEN
- JWT_ALGORITHM
- JWT_EXPIRATION_MINUTES

### 7. Database Migrations

**alembic/** - система миграций

- Версионирование схемы БД
- Автоматическая генерация миграций
- Rollback поддержка
- История изменений

## Поток данных

### Детекция аномалий

```
Client
  ↓ POST /forward (JSON с логами)
FastAPI Endpoint
  ↓ Pydantic validation
ML Model (Isolation Forest)
  ↓ Нормализация сообщений
  ↓ Токенизация
  ↓ TF-IDF векторизация
  ↓ Prediction (score_samples)
  ↓ Сравнение с threshold
Database
  ↓ Save request history
Response
  ↓ JSON с score, is_anomaly, threshold
Client
```

### Авторизованный запрос

```
Client
  ↓ Request with JWT token
JWT Middleware
  ↓ Token validation
  ↓ Extract user info
Database
  ↓ Verify user exists
  ↓ Check permissions
Endpoint Handler
  ↓ Business logic
Response
  ↓ JSON data
Client
```

## Безопасность

### 1. Аутентификация

- Пароли хешируются с bcrypt (work factor: 12)
- JWT токены с временем жизни (default: 30 минут)
- Secure token generation (python-jose)

### 2. Авторизация

- Role-based access control
- Admin endpoints защищены JWT
- DELETE /history требует отдельный admin token
- Dependency injection для проверки прав

### 3. Валидация данных

- Pydantic схемы для всех входных данных
- Type checking на уровне Python
- Автоматическая валидация JSON
- Проверка минимальной длины списка логов

### 4. Обработка ошибок

- Централизованная обработка исключений
- Правильные HTTP коды ошибок
- Логирование всех запросов в БД
- Graceful degradation

## Зависимости

### Runtime

```
FastAPI - веб-фреймворк
uvicorn - ASGI сервер
SQLAlchemy - ORM
alembic - миграции БД
pydantic - валидация данных
python-jose - JWT
bcrypt - хеширование паролей
scikit-learn - Isolation Forest модель
joblib - сериализация модели
numpy - численные операции
```

## Производительность

### Оптимизации

1. **Асинхронность**
   - Async/await для I/O операций
   - Асинхронная работа с БД
   - Non-blocking endpoints

2. **Database**
   - Индексы на часто запрашиваемые поля
   - Lazy loading для связей

3. **ML Model**
   - Модель загружается один раз при первом запросе
   - Ленивая инициализация (lazy loading)
   - Векторизация через sparse matrices

### Метрики производительности

Endpoint `/stats` предоставляет:
- Среднее время обработки
- Медиана (50-й перцентиль)
- 95-й перцентиль
- 99-й перцентиль
- Средний размер входных данных

## Расширяемость

### Добавление новых моделей

1. Создайте новый класс модели в `app/ml_model.py`
2. Реализуйте методы `predict()` и `predict_from_logs()`
3. Обновите схемы в `app/schemas.py`

### Добавление новых endpoints

1. Добавьте handler в `app/main.py`
2. Создайте Pydantic схемы в `app/schemas.py`
3. При необходимости обновите модели БД
4. Создайте миграцию через alembic

## Тестирование

### Unit тесты

- Тесты для auth функций
- Тесты для ML модели
- Тесты для схем валидации

### Integration тесты

- Тесты endpoints
- Тесты БД операций
- Тесты авторизации

### Тестовые данные

В папке `test_logs/` находятся примеры:
- `normal_logs.json` - нормальные логи
- `anomaly_logs.json` - аномальные логи

## Деплой

### Development

```bash
uvicorn app.main:app --reload
```

### Docker Compose

```bash
docker-compose up -d
```

## Best Practices

1. **Код**
   - Понятные имена переменных и функций
   - Type hints везде
   - Async/await для I/O

2. **БД**
   - Используйте миграции для всех изменений схемы
   - Индексы на часто используемых полях

3. **Безопасность**
   - Никогда не коммитьте .env файл
   - Регулярно обновляйте зависимости
   - Валидируйте все входные данные

4. **API**
   - Правильные HTTP коды
   - Понятные сообщения об ошибках
   - Документация (Swagger)
