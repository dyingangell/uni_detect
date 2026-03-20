HOW TO START TS

1) сначала запускаем докер с запущенным docker desktop - docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:lates
2)  python multithread.py "номер папки в которой будут видео от 1 до 8, пример test1.mp4,test2.mp4"
4)  проект должен выгледить так
5)  <img width="2206" height="586" alt="image" src="https://github.com/user-attachments/assets/dcf8cbb6-eaea-4032-99d3-630fe0ede2a4" />
тут видно 8 папок, это для эмулирования 
6) python worker.py просто запускаешь, он работает в любом случаи, остальное сам дебаж чмо 

Desktop UI:
- python desktop_viewer.py  (окно с камерами + warnings)
- python warnings_window.py (отдельное окно со списком warnings, можно скроллить/фильтровать/экспортировать)



