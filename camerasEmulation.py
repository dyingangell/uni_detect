import subprocess
import sys
import time

# Настройки теста
NUM_CAMERAS = 16  # Попробуй сначала 20, потом увеличивай до 100
VIDEO_PATH = "video/test7.mp4" # Путь к твоему тестовому файлу
PRODUCER_SCRIPT = "producer.py"
PYTHON_EXE = sys.executable
processes = []

print(f"Запуск {NUM_CAMERAS} эмуляторов камер...")

for i in range(1, NUM_CAMERAS + 1):
    # Запускаем каждый продюсер как отдельный фоновый процесс
    # Команда: python producer.py [ID] [PATH]
    time.sleep(0.1)
    p = subprocess.Popen([PYTHON_EXE, PRODUCER_SCRIPT, str(i), VIDEO_PATH])
    processes.append(p)



print(f"\nВсе {NUM_CAMERAS} камер запущены.")
print("Нажми Ctrl+C, чтобы остановить все тесты.")

try:
    # Держим скрипт активным, пока работают процессы
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nОстановка всех камер...")
    for p in processes:
        p.terminate()
    print("Тест завершен.")