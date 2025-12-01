from gpiozero import AngularServo
from time import sleep

servo = AngularServo(18, min_angle=0, max_angle=180)
print("Servo Control... Press Ctrl+C to exit")

try:
    while True:
        print("Angle: 0")
        servo.angle = 0
        sleep(1)
        
        print("Angle: 90")
        servo.angle = 90
        sleep(1)
        
        print("Angle: 180")
        servo.angle = 180
        sleep(1)
        
        print("Angle: 90")
        servo.angle = 90
        sleep(1)
        
except KeyboardInterrupt:
    print("Stopping...")
    servo.detach()