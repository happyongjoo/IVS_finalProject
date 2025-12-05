from picamera2 import Picamera2
import cv2
import numpy as np
import time
import math

from gpiozero import AngularServo, Motor

# =========================
# 서보 / 제어 파라미터
# =========================
SERVO_PIN = 18
SERVO_MIN_ANGLE = 0
SERVO_MAX_ANGLE = 180
SERVO_CENTER_ANGLE = 90

K_ANGLE = 3.0                    # 선형 매핑 게인

MAX_LANE_ANGLE = 60.0            # fitLine 클램핑 기준
HARD_TURN_THRESHOLD = 45.0       # ±45도 넘으면 최대 조향

ROI_TOP_RATIO = 0.25

# =========================
# 모터 파라미터
# =========================
MOTOR_FORWARD_PIN = 14
MOTOR_BACKWARD_PIN = 15
MOTOR_ENABLE_PIN = 23
BASE_SPEED = 0.5

# ==================================
# 차선 기울기 계산 함수
# ==================================
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


# ==================================
# 좌/우 차선 기울기 판별
# ==================================
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


# ==================================
# 메인 루프
# ==================================
def main():
    picam2 = Picamera2()
    sensor_w, sensor_h = picam2.sensor_resolution

    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        raw={"size": (sensor_w, sensor_h)}
    )

    picam2.configure(config)
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

    src_points = np.float32([(80, 30), (0, 280), (640, 280), (560, 30)])
    dst_points = np.float32([(0, 0), (0, 480), (640, 480), (640, 0)])

    print("Auto Lane Tracking (Full-steer logic) Start")

    try:
        while True:
            frame = cv2.cvtColor(picam2.capture_array(), cv2.COLOR_RGB2BGR)
            cv2.imshow("frame", frame)

            M = cv2.getPerspectiveTransform(src_points, dst_points)
            bird_view = cv2.warpPerspective(frame, M, (640, 480))


            hsv = cv2.cvtColor(bird_view, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0,0,0]), np.array([180,255,80]))

            result = get_lane_angle_split(mask)

            if result is not None:
                # 항상 float 반환됨
                lane_angle_deg, side, roi_top = result

                # ---- 조향 로직 ----
                if lane_angle_deg <= -HARD_TURN_THRESHOLD:
                    steering_angle = SERVO_MIN_ANGLE
                elif lane_angle_deg >= HARD_TURN_THRESHOLD:
                    steering_angle = SERVO_MAX_ANGLE
                else:
                    steering_angle = SERVO_CENTER_ANGLE - K_ANGLE * lane_angle_deg

                steering_angle = max(SERVO_MIN_ANGLE,
                                     min(SERVO_MAX_ANGLE, steering_angle))

                servo.angle = steering_angle
                motor.forward(BASE_SPEED)

            else:
                # 라인 없음
                side = "none"
                lane_angle_deg = 0
                steering_angle = SERVO_CENTER_ANGLE
                roi_top = int(480 * ROI_TOP_RATIO)

                servo.angle = SERVO_CENTER_ANGLE
                motor.stop()

            # ---- 디버그 출력 ----
            bev_debug = bird_view.copy()
            h, w = bev_debug.shape[:2]

            # ---------------------------------------------------------
            # [수정됨] ROI 영역 빨간색 선 표시 (두께 2)
            # (0, 0, 255) -> BGR 코드 (Red)
            # ---------------------------------------------------------
            # 1. ROI 상단 가로줄 (확실히 보이게)
            cv2.line(bev_debug, (0, roi_top), (w, roi_top), (0, 0, 255), 2)
            
            # 2. ROI 전체 박스 (선택 사항, 필요 없으면 주석 처리)
            cv2.rectangle(bev_debug, (0, roi_top), (w-1, h-1), (0, 0, 255), 2)
            # ---------------------------------------------------------

            cv2.putText(bev_debug, f"side: {side}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)
            cv2.putText(bev_debug, f"lane_angle: {lane_angle_deg:.1f}",
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)
            cv2.putText(bev_debug, f"steer: {steering_angle:.1f}",
                        (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)

            cv2.imshow("Mask", mask)
            cv2.imshow("BEV Debug", bev_debug)

            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    finally:
        picam2.stop()
        motor.stop()
        servo.angle = SERVO_CENTER_ANGLE
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()