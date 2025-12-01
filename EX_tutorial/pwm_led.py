
from gpiozero import PWMLED
from time import sleep

led = PWMLED(17)
print("Press Ctrol + C to exit")

try:
	while True:
			led.value = 0
			sleep(0.2)
			led.value = 0.3
			sleep(0.2)
			led.value = 0.6
			sleep(0.2)
			led.value = 1
			sleep(0.1)
except KeyboardInterrupt:
		led.off()
		print("Done")
