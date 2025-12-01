from gpiozero import LED, Button
from signal import pause

led = LED(17)
button = Button(2)

print("Press button.")

button.when_pressed = led.toggle

pause()
