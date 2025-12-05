from gpiozero import DistanceSensor
from time import sleep

sensor = DistanceSensor(echo = 8, trigger = 11)
print("Press Ctrl + C to exit")

try:
	while True:
		dist = sensor.distance
		
		print(f"dist: {dist:.2f} m")
		
		sleep(0.5)
		
except KeyboardInterrupt:
	print("Measurement Stopped.")
