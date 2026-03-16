HOW TO START TS

1) сначала запускаем докер с запущенным docker desktop - docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:lates
2)  python multithread.py "номер папки в которой будут видео от 1 до 8, пример test1.mp4,test2.mp4"
3) python worker.py

нужно установить cv2 с поддержкой gstream


