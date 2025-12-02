import time
import numpy as np
import cv2
from gpiozero import AngularServo, Motor
from picamera2 import Picamera2

servo = AngularServo(18, min_angle=0, max_angle=180)
motor = Motor(forward=14, backward=15, enable=23, pwm=True)

BASE_SPEED = 0.6  # 평소 직진 속도
BLIND_SPEED = 0.8  # 횡단보도 탈출 시 속도
STOP_TIME = 4.0  # 정지 시간 (초)
BLIND_RUN_TIME = 2.0  # 탈출 직진 시간 (초)

SERVO_CENTER = 90

CROSSWALK_THRESHOLD = 14000

ROI_Y_LOW = 200
ROI_Y_HIGH = 400
ROI_X_LOW = 0
ROI_X_HIGH = 640

servo.angle = SERVO_CENTER

def main():
    picam2 = Picamera2()
    sensor_w, sensor_h = picam2.sensor_resolution
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        raw={"size": (sensor_w, sensor_h)}
    )
    picam2.configure(config)
    picam2.set_controls({"ScalerCrop": (0, 0, sensor_w, sensor_h)})
    picam2.start()
    time.sleep(1)
    try:
        while True:
            frame = picam2.capture_array()
            # frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            height, width = frame.shape[:2]

            roi = frame[ROI_Y_LOW:ROI_Y_HIGH, ROI_X_LOW:ROI_X_HIGH]

            # BGR -> HSV
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # RED 범위 설정 1
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])

            # RED 범위 설정 2
            lower_red2 = np.array([170, 100, 100])
            upper_red2 = np.array([180, 255, 255])

            # 범위 병합
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.addWeighted(mask1, 1.0, mask2, 1.0, 0.0)

            # 빨간색 픽셀 개수 세기
            red_pixel_count = cv2.countNonZero(red_mask)

            # 시각화, 화면에 픽셀 수 표시
            cv2.putText(
                frame,
                f"Red Pixels: {red_pixel_count}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.rectangle(
                frame,
                (ROI_X_LOW, ROI_Y_LOW),
                (ROI_X_HIGH, ROI_Y_HIGH),
                (0, 0, 255),
                2,
            )

            # 판단 및 제어
            if red_pixel_count > CROSSWALK_THRESHOLD:
                print(f"횡단보도 ({red_pixel_count}) -> 정지")
                # 일단 정지
                motor.stop()
                time.sleep(STOP_TIME)

                # 탈출
                servo.angle = SERVO_CENTER
                motor.forward(speed=BLIND_SPEED)
                time.sleep(BLIND_RUN_TIME)

                continue

            else:
                print(f"직진 ({red_pixel_count})")
                servo.angle = SERVO_CENTER
                motor.forward(speed=BASE_SPEED)

            cv2.imshow("Main View", frame)
            #cv2.imshow("Red Mask View", red_mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        print("시스템 종료")
        motor.stop()
        motor.close()
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
