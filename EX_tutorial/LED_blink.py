from gpiozero import LED
from time import sleep
led = LED(17) 
print("Press Ctrl+C to exit")
 
 
try:
	while True:
		led.on()    
		sleep(0.5)  
		led.off()   
		sleep(0.5)  
except KeyboardInterrupt:
	print("Done")
