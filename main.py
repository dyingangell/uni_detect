from multiprocessing import shared_memory
import time

SHM_NAME = "cv_frame_buffer"
SIZE = int(200 * 640 * 640 * 3)

def start_master():
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SIZE)
        print(f"Shared Memory '{SHM_NAME}' created and held.")
        print("Don't close this terminal window until workers are done!")

        # Infinite loop to keep the process alive
        while True:
            time.sleep(10)

    except FileExistsError:
        print("Shared Memory already exists.")
    except KeyboardInterrupt:
        print("\nDeleting Shared Memory and exiting...")
        shm.close()
        shm.unlink()

if __name__ == "__main__":
    start_master()