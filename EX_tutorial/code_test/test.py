from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()

video_config = picam2.create_video_configuration(
    main={
        "size": (640, 480),   # 필요하면 640x480으로 더 줄여봐
        "format": "BGR888"     # OpenCV가 바로 쓰는 포맷
    },
    controls={
        "FrameRate": 30        # 30fps 목표 (카메라 모드에 따라 조절됨)
    }
)

picam2.configure(video_config)
picam2.start()

prev_time = time.time()
frame_count = 0

while True:
    # 이미 BGR이니까 색 변환 필요 없음
    frame = picam2.capture_array("main")

    frame_count += 1
    now = time.time()
    if now - prev_time >= 1.0:
        print("FPS:", frame_count)
        frame_count = 0
        prev_time = now

    cv2.imshow("RPi Cam (Video Mode)", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cv2.destroyAllWindows()
