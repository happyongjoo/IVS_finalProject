from picamera2 import Picamera2
import cv2
import numpy as np
import time
import math

from gpiozero import AngularServo, Motor

# ============ parameter ============
SERVO_PIN = 18
SERVO_MIN_ANGLE = 0
SERVO_MAX_ANGLE = 180
SERVO_CENTER_ANGLE = 90

MOTOR_FORWARD_PIN = 14
MOTOR_BACKWARD_PIN = 15
MOTOR_ENABLE_PIN = 23
BASE_SPEED = 0.4

K_ANGLE = 2.0                    # 선형 매핑 게인

MAX_LANE_ANGLE = 60.0            # fitLine 클램핑 기준
HARD_TURN_THRESHOLD = 40.0       # ±45도 넘으면 최대 조향

ROI_TOP_RATIO = 0.5

CROSSWALK_THRESHOLD = 14000

ROI_Y_LOW = 200
ROI_Y_HIGH = 400
ROI_X_LOW = 0
ROI_X_HIGH = 640

src_points = np.float32([(80, 70), (0, 280), (640, 280), (560, 70)])
dst_points = np.float32([(0, 0), (0, 480), (640, 480), (640, 0)])
    
# ============ Init ============
def Init():
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

    servo = AngularServo(
        SERVO_PIN,
        min_angle=SERVO_MIN_ANGLE,
        max_angle=SERVO_MAX_ANGLE
    )
    servo.angle = SERVO_CENTER_ANGLE

    motor = Motor(
        forward=MOTOR_FORWARD_PIN,
        backward=MOTOR_BACKWARD_PIN,
        enable=MOTOR_ENABLE_PIN,
        pwm=True
    )
    motor.stop()
    
    return picam2, servo, motor

# ============ Lane Keeping ============
# 차선 기울기 계산 함수
def fit_lane_angle_deg(side_img):
    points = cv2.findNonZero(side_img)
    if points is None or len(points) < 50:
        return None

    line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line.flatten()

    if vy < 0:
        vx, vy = -vx, -vy

    angle_rad = math.atan2(vx, vy)
    angle_deg = math.degrees(angle_rad)

    angle_deg = max(-MAX_LANE_ANGLE, min(MAX_LANE_ANGLE, angle_deg))
    return angle_deg

# 좌/우 차선 기울기 판별
def get_lane_angle_split(mask):
    h, w = mask.shape

    roi_top = int(h * ROI_TOP_RATIO)
    roi = mask[roi_top:h, :]

    mid = w // 2
    left_roi = roi[:, :mid]
    right_roi = roi[:, mid:]

    angle_left = fit_lane_angle_deg(left_roi)
    angle_right = fit_lane_angle_deg(right_roi)

    # 🔴 차선 둘 다 없음 → None만 반환
    if angle_left is None and angle_right is None:
        return None

    # 한쪽만 있는 경우
    if angle_left is not None and angle_right is None:
        return angle_left, "left", roi_top

    if angle_left is None and angle_right is not None:
        return angle_right, "right", roi_top

    # 둘 다 있음 → 평균
    return (angle_left + angle_right) / 2.0, "both", roi_top

# ============ main ============
def main():
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