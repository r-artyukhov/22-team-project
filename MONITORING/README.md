# Мониторинг и деплой

## Порты

| Сервис     | URL |
|------------|-----|
| FastAPI    | http://localhost:8080/docs |
| MLflow     | http://localhost:5050 |
| Grafana    | http://localhost:3030 (admin / admin) |
| Prometheus | http://localhost:9090 |
| Alerts     | http://localhost:9090/alerts |

## Запуск сервиса
Мониторинг:
```bash
cd MONITORING
docker compose up --build -d
```
Симуляция работы модели:
```bash
cd MONITORING
python send_logs.py --interval 2 --rounds 40
```