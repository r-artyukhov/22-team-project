# Примеры использования API

## Python примеры

### 1. Регистрация и авторизация

```python
import requests

BASE_URL = "http://localhost:8000"

# Регистрация администратора
response = requests.post(
    f"{BASE_URL}/register",
    json={
        "username": "admin",
        "password": "admin123",
        "is_admin": True
    }
)
print(response.json())

# Получение токена
response = requests.post(
    f"{BASE_URL}/token",
    data={
        "username": "admin",
        "password": "admin123"
    }
)
token = response.json()["access_token"]
print(f"Access Token: {token}")
```

### 2. Детекция аномалий в логах

```python
import requests

# Нормальные логи
logs_normal = {
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
        },
        {
            "message": "Verification succeeded for blk_-1608999687919862906",
            "component": "DataBlockScanner",
            "level": "INFO"
        }
    ]
}

response = requests.post(
    f"{BASE_URL}/forward",
    json=logs_normal
)

result = response.json()
print(f"Score: {result['score']:.4f}")
print(f"Is Anomaly: {result['is_anomaly']}")
print(f"Threshold: {result['threshold']:.4f}")
print(f"Num Events: {result['num_events']}")
```

### 3. Детекция аномальных логов

```python
import requests

# Аномальные логи
logs_anomaly = {
    "logs": [
        {
            "message": "Exception in receiveBlock for block blk_-1608999687919862906 java.io.IOException",
            "component": "DataNode$DataXceiver",
            "level": "ERROR"
        },
        {
            "message": "writeBlock received exception java.net.SocketTimeoutException: 60000 millis timeout",
            "component": "DataNode$DataXceiver",
            "level": "ERROR"
        },
        {
            "message": "Lost connection to datanode 10.250.19.102:50010",
            "component": "DFSClient",
            "level": "ERROR"
        }
    ]
}

response = requests.post(
    f"{BASE_URL}/forward",
    json=logs_anomaly
)

result = response.json()
print(f"Score: {result['score']:.4f}")
print(f"Is Anomaly: {result['is_anomaly']}")  # True
```

### 4. Просмотр истории (требует авторизацию)

```python
import requests

headers = {"Authorization": f"Bearer {token}"}

response = requests.get(
    f"{BASE_URL}/history",
    headers=headers
)

history = response.json()
print(f"Total requests: {history['total']}")

for item in history['items'][:5]:
    print(f"- {item['request_type']}: {item['processing_time']:.3f}s, anomaly: {item['result']}")
```

### 5. Получение статистики (требует авторизацию)

```python
import requests

headers = {"Authorization": f"Bearer {token}"}

response = requests.get(
    f"{BASE_URL}/stats",
    headers=headers
)

stats = response.json()
print(f"Total requests: {stats['total_requests']}")
print(f"Mean processing time: {stats['mean_processing_time']:.3f}s")
print(f"95th percentile: {stats['percentile_95_processing_time']:.3f}s")
print(f"99th percentile: {stats['percentile_99_processing_time']:.3f}s")
```

### 6. Удаление истории (требует admin token)

```python
import requests

headers = {"Authorization": "Bearer admin_secret_token_12345"}

response = requests.delete(
    f"{BASE_URL}/history",
    headers=headers
)

if response.status_code == 204:
    print("History deleted successfully")
```

## cURL примеры

### Регистрация пользователя

```bash
curl -X POST "http://localhost:8000/register" \
  -H "Content-Type: application/json" \
  -d '{"username": "user1", "password": "pass123", "is_admin": false}'
```

### Получение JWT токена

```bash
curl -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123"
```

### Детекция аномалий в логах

```bash
curl -X POST "http://localhost:8000/forward" \
  -H "Content-Type: application/json" \
  -d '{
    "logs": [
      {
        "message": "Receiving block blk_-1608999687919862906",
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

### Использование тестовых файлов

```bash
# Нормальные логи
cat test_logs/normal_logs.json | curl -X POST "http://localhost:8000/forward" \
  -H "Content-Type: application/json" \
  -d @-

# Аномальные логи
cat test_logs/anomaly_logs.json | curl -X POST "http://localhost:8000/forward" \
  -H "Content-Type: application/json" \
  -d @-
```

### Просмотр истории

```bash
TOKEN="your_jwt_token_here"

curl -X GET "http://localhost:8000/history" \
  -H "Authorization: Bearer $TOKEN"
```

### Просмотр статистики

```bash
TOKEN="your_jwt_token_here"

curl -X GET "http://localhost:8000/stats" \
  -H "Authorization: Bearer $TOKEN"
```

### Удаление истории

```bash
curl -X DELETE "http://localhost:8000/history" \
  -H "Authorization: Bearer admin_secret_token_12345"
```