from gpiozero import DigitalInputDevice
from time import sleep

sensor = DigitalInputDevice(26, pull_up = True)

print("System Ready")

while True:
    if sensor.value == 1:
        print("Black or nothing")

    else:
        print("white")

    sleep(0.1)