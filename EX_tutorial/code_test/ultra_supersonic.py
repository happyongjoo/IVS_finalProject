from gpiozero import DistanceSensor, LED
from time import sleep
import threading

ECHO_PIN = 8
TRIG_PIN = 11
LED_PIN = 6

sensor = DistanceSensor(echo = ECHO_PIN, trigger = TRIG_PIN)
led = LED(LED_PIN)

stop_event = threading.Event()

def sensor_loop(stop_flag: threading.Event):
    while not stop_flag.is_set():
        dist = sensor.distance
        print(f"[{threading.current_thread().name}] distance = {dist:.2f} m")

def led_loop(stop_flag: threading.Event):
    while not stop_flag.is_set():
        led.on()
        sleep(0.5)
        led.off()
        sleep(0.5)

if __name__ == "__main__":
    print("Press Ctrl + C to exit.")

    sensor_worker = threading.Thread(
        target = sensor_loop,
        args = (stop_event, ),
        name = "SensorThread"
    )
    sensor_worker.start()

    led_worker = threading.Thread(
        target = led_loop,
        args = (stop_event, ),
        name = "LEDThread"
    )
    led_worker.start()

    try:
        while True:
            sleep(1.0)

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        stop_event.set()
        sensor_worker.join(timeout = 1.0)
        led_worker.join(timeout = 1.0)
        led.off()
        print("Stopped.")