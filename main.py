from multiprocessing import shared_memory
import time

SHM_NAME = "cv_frame_buffer"
SIZE = int(200 * 640 * 640 * 3)

def start_master():
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SIZE)
        print(f"✅ Память '{SHM_NAME}' создана и УДЕРЖИВАЕТСЯ.")
        print("НЕ ЗАКРЫВАЙТЕ это окно, пока работают воркеры!")

        # Бесконечный цикл, чтобы процесс не умер
        while True:
            time.sleep(10)

    except FileExistsError:
        print("Память уже создана кем-то другим.")
    except KeyboardInterrupt:
        print("\nУдаление памяти и выход...")
        shm.close()
        shm.unlink()

if __name__ == "__main__":
    start_master()