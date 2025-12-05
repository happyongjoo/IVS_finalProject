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

# [수정] ROI 비율 상수 두 개로 분리
ROI_RATIO_NORMAL = 0.6  # 평상시 (절반만 봄)
ROI_RATIO_ROTARY = 0.25  # 로터리/이벤트 시 (3/4를 봄, 더 넓게)

# =========================
# 초록색 플래그 감지 파라미터 (HSV)
# =========================
# 초록색은 보통 Hue 40~80 사이
LOWER_GREEN = np.array([40, 50, 50])
UPPER_GREEN = np.array([90, 255, 255])
GREEN_PIXEL_THRESHOLD = 300  # 이 픽셀 수 이상 초록색이 보이면 로터리로 판단

# =========================
# 모터 파라미터
# =========================
MOTOR_FORWARD_PIN = 14
MOTOR_BACKWARD_PIN = 15
MOTOR_ENABLE_PIN = 23
BASE_SPEED = 0.6

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
# 좌/우 차선 기울기 판별 (수정됨)
# ==================================
# [수정] roi_ratio를 인자로 받도록 변경
def get_lane_angle_split(mask, roi_ratio):
    h, w = mask.shape

    # 동적으로 변하는 비율 적용
    roi_top = int(h * roi_ratio)
    roi = mask[roi_top:h, :]

    mid = w // 2
    left_roi = roi[:, :mid]
    right_roi = roi[:, mid:]

    angle_left = fit_lane_angle_deg(left_roi)
    angle_right = fit_lane_angle_deg(right_roi)

    if angle_left is None and angle_right is None:
        return None, roi_top  # roi_top도 반환하여 디버그에 활용

    if angle_left is not None and angle_right is None:
        return angle_left, "left", roi_top

    if angle_left is None and angle_right is not None:
        return angle_right, "right", roi_top

    return (angle_left + angle_right) / 2.0, "both", roi_top


# ==================================
# 메인 루프
# ==================================
def main():
    picam2 = Picamera2()
    sensor_w, sensor_h = picam2.sensor_resolution
    
    # [수정] 화각 문제 해결을 위해 preview configuration 사용
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

    # ROI 영역 (사다리꼴)
    src_points = np.float32([(40, 50), (0, 280), (640, 280), (600, 50)])
    dst_points = np.float32([(0, 0), (0, 480), (640, 480), (640, 0)])

    print("Auto Lane Tracking with Green Flag Detection Start")

    # 초기 ROI 비율 설정
    current_roi_ratio = ROI_RATIO_NORMAL
    mode_status = "NORMAL"

    try:
        while True:
            # 1. 이미지 캡처
            frame = cv2.cvtColor(picam2.capture_array(), cv2.COLOR_RGB2BGR)

            # 2. BEV 변환
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            bird_view = cv2.warpPerspective(frame, M, (640, 480))

            cv2.imshow("bird_view", frame)

            # 3. HSV 변환
            hsv = cv2.cvtColor(bird_view, cv2.COLOR_BGR2HSV)

            # ---------------------------------------------------------
            # [추가] 초록색 플래그 감지 로직
            # ---------------------------------------------------------
            green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
            green_count = cv2.countNonZero(green_mask)

            if green_count > GREEN_PIXEL_THRESHOLD:
                current_roi_ratio = ROI_RATIO_ROTARY  # 0.25 (넓게 봄)
                mode_status = "ROTARY (Green)"
            else:
                current_roi_ratio = ROI_RATIO_NORMAL  # 0.5 (좁게 봄)
                mode_status = "NORMAL"
            # ---------------------------------------------------------

            # 4. 차선 마스크 생성 (노란색/흰색 등 기존 로직 유지)
            # (주의: 사용자의 기존 코드 범위가 0~180, 255, 80 인데 이는 검은색에 가깝습니다.
            #  환경에 맞게 차선 색상 범위를 조정하세요. 여기선 그대로 둡니다.)
            lane_mask = cv2.inRange(hsv, np.array([0,0,0]), np.array([180,255,80]))

            # 5. 차선 인식 (현재 결정된 roi_ratio 전달)
            result = get_lane_angle_split(lane_mask, current_roi_ratio)

            # 결과 처리
            lane_angle_deg = 0
            side = "none"
            roi_top = int(480 * current_roi_ratio) # 기본값

            if result is not None and result[0] is not None:
                lane_angle_deg, side, roi_top = result

                # 조향 로직
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
                # 차선 없음
                side = "none"
                servo.angle = SERVO_CENTER_ANGLE
                motor.stop()


            # ---------------------------------------------------------
            # 디버그 출력
            # ---------------------------------------------------------
            bev_debug = bird_view.copy()
            h, w = bev_debug.shape[:2]

            # 1. 인식된 초록색 영역 표시 (초록색 테두리)
            # green_mask에서 윤곽선을 찾아 표시하면 어디가 초록색인지 알 수 있음
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(bev_debug, contours, -1, (0, 255, 0), 2)

            # 2. ROI 라인 (빨간 줄) - 모드에 따라 높이가 휙휙 바뀜
            cv2.line(bev_debug, (0, roi_top), (w, roi_top), (0, 0, 255), 3)

            # 3. 정보 텍스트
            cv2.putText(bev_debug, f"Mode: {mode_status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(bev_debug, f"GreenPx: {green_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(bev_debug, f"Angle: {lane_angle_deg:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Lane Mask", lane_mask)
            cv2.imshow("Green Mask", green_mask) # 초록색만 따로 보는 창
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