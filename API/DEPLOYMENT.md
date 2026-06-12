# Руководство по деплою

## Локальный запуск

### Прямой запуск

```bash
# Установка зависимостей
pip install -r requirements.txt

# Применение миграций
alembic upgrade head

# Запуск сервера
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Docker Compose

### Запуск

```bash
# Запуск (миграции применяются автоматически при старте контейнера)
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановка
docker-compose down
```

### Пересборка после изменений

```bash
docker-compose down
docker-compose build
docker-compose up -d
```

**Примечание:** Docker контейнер автоматически запускает `alembic upgrade head` перед стартом приложения (см. Dockerfile CMD)

## База данных

### SQLite (по умолчанию)

SQLite используется по умолчанию и подходит для разработки и небольших нагрузок.

```
DATABASE_URL=sqlite+aiosqlite:///./ml_service.db
```

## Работа с миграциями Alembic

### Проверка текущей версии

```bash
# Локально
alembic current

# В Docker контейнере
docker exec yearprojectmlapi-ml-service-1 alembic current
```

### Создание новой миграции

```bash
# После изменения моделей в app/models.py
alembic revision --autogenerate -m "описание изменений"

# Пример: добавили поле phone в User
alembic revision --autogenerate -m "add phone field to users"
```

### Применение миграций

```bash
# Применить все непримененные миграции
alembic upgrade head

# Применить конкретную миграцию
alembic upgrade <revision_id>

# В Docker автоматически при запуске
```

### Откат миграций

```bash
# Откатить одну миграцию назад
alembic downgrade -1

# Откатить до конкретной версии
alembic downgrade <revision_id>

# Откатить все миграции
alembic downgrade base
```

### История миграций

```bash
# Просмотр всех миграций
alembic history

# Подробная информация
alembic history --verbose
```

## Резервное копирование

### SQLite

```bash
# Backup
sqlite3 ml_service.db ".backup ml_service_backup.db"

# Или через cron (ежедневно в 2:00)
0 2 * * * /usr/bin/sqlite3 /path/to/ml_service.db ".backup /backups/ml_service_$(date +\%Y\%m\%d).db"
```

## Checklist перед деплоем

- [ ] SECRET_KEY изменен на уникальный
- [ ] ADMIN_TOKEN изменен на уникальный
- [ ] .env файл не в git репозитории
- [ ] Миграции применены
- [ ] Модель isolation_forest.joblib присутствует в папке models/

## Генерация секретов

```bash
# Генерация нового SECRET_KEY
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Генерация нового ADMIN_TOKEN
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Полезные команды

### Docker

```bash
# Просмотр логов
docker-compose logs -f ml-service

# Перезапуск
docker-compose restart ml-service

# Остановка
docker-compose stop

# Удаление контейнеров и volumes
docker-compose down -v

# Проверка использования ресурсов
docker stats
```

### Локальный запуск

```bash
# Просмотр процессов
ps aux | grep uvicorn

# Проверка использования ресурсов
htop
df -h
free -h
```
