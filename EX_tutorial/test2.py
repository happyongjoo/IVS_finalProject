from picamera2 import Picamera2
import cv2

picam2 = Picamera2()

# rpicam-hello와 비슷한 preview 설정 (ISP 거친 RGB888)
config = picam2.create_preview_configuration(
    main={
        "size": (640, 480),   # 필요에 따라 조절 가능
        "format": "RGB888"     # ISP에서 나온 순수 RGB
    }
)

picam2.configure(config)
picam2.start()

# rpicam-hello랑 비슷한 자동 화이트밸런스/노출 사용
picam2.set_controls({
    "AwbEnable": True,   # 자동 화이트밸런스 켜기
    "AeEnable": True     # 자동 노출
})

while True:
    # RGB888로 한 프레임 받기
    frame_rgb = picam2.capture_array()

    # OpenCV는 BGR을 쓰니까 RGB → BGR 변환
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    cv2.imshow("RPi Cam (RGB888 -> BGR)", frame_bgr)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cv2.destroyAllWindows()
