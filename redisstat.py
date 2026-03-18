import redis
import time

r = redis.Redis(host='localhost', port=6379)

try:
    while True:
        queue_len = r.llen("image_batch_queue")
        memory = r.info('memory')['used_memory_human']
        print(f"--- Мониторинг Redis ---")
        print(f"Кадров в очереди: {queue_len}")
        print(f"Память занята: {memory}")
        print(f"------------------------")
        time.sleep(1)
except KeyboardInterrupt:
    print("Стоп")