from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()

sensor_w, sensor_h = picam2.sensor_resolution  # (4608, 2592)

config = picam2.create_video_configuration(
    main={          # 우리가 실제로 처리/보는 용
        "size": (640, 360),
        "format": "RGB888"
    },
    raw={           # 여기서 풀센서 모드 강제
        "size": (sensor_w, sensor_h)
    }
)

picam2.configure(config)
picam2.start()
time.sleep(1)

meta = picam2.capture_metadata()
print("Sensor resolution:", sensor_w, sensor_h)
print("ScalerCrop after full-sensor config:", meta.get("ScalerCrop", None))

while True:
    frame_rgb = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("RPi Cam (try full FOV)", frame_bgr)

    if cv2.waitKey(1) & 0xFF == 27:
        break

picam2.stop()
cv2.destroyAllWindows()
