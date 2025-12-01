from picamera2 import Picamera2
import time

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={ "size": (640, 360), "format": "RGB888" }
)
picam2.configure(config)
picam2.start()
time.sleep(1)  # 카메라 워밍업

print("Sensor full resolution:", picam2.sensor_resolution)

meta = picam2.capture_metadata()
print("Current ScalerCrop:", meta.get("ScalerCrop", None))
