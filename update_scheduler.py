import schedule
import threading

def run_scheduler():
    # Запуск обновления каждые 24 часа
    schedule.every().day.at("06:00").do(update_vector_db)
    while True:
        schedule.run_pending()
        time.sleep(60)

# Запуск в отдельном потоке
threading.Thread(target=run_scheduler, daemon=True).start()