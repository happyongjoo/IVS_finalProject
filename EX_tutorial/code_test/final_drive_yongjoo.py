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

# =========================
# ROI 비율 (기본 / 로터리)
# =========================
ROI_RATIO_NORMAL = 0.6   # 평상시 (아래쪽만 봄)
ROI_RATIO_ROTARY = 0.25  # 로터리/이벤트 시 (더 넓게)

# =========================
# 초록색 플래그 감지 파라미터 (HSV)
# =========================
LOWER_GREEN = np.array([40, 50, 50])
UPPER_GREEN = np.array([90, 255, 255])
GREEN_PIXEL_THRESHOLD = 300  # 이 픽셀 수 이상 초록색이 보이면 로터리로 판단

# =========================
# 횡단보도(빨강) / 주차(파랑) 색 범위 & 임계값
# =========================
# 빨강은 HSV에서 Hue가 0 근처 + 180 근처 두 구간으로 나뉨
LOWER_RED1 = np.array([0, 80, 80])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 80, 80])
UPPER_RED2 = np.array([180, 255, 255])

# 파란색 (대략 Hue 100~140 근처)
LOWER_BLUE = np.array([100, 80, 80])
UPPER_BLUE = np.array([140, 255, 255])

CW_THRESHOLD_RED = 300    # 빨간 픽셀 기준 (임시 값, 나중에 튜닝)
CW_THRESHOLD_BLUE = 300   # 파란 픽셀 기준 (임시 값, 나중에 튜닝)

# =========================
# 모터 파라미터
# =========================
MOTOR_FORWARD_PIN = 14
MOTOR_BACKWARD_PIN = 15
MOTOR_ENABLE_PIN = 23
BASE_SPEED = 0.6

# =========================
# BEV용 사다리꼴 ROI 포인트
# =========================
SRC_POINTS = np.float32([(40, 50), (0, 280), (640, 280), (600, 50)])
DST_POINTS = np.float32([(0, 0), (0, 480), (640, 480), (640, 0)])


# ==================================
# 카메라 / 모터 초기화 함수
# ==================================
def init_camera():
    picam2 = Picamera2()
    sensor_w, sensor_h = picam2.sensor_resolution

    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        raw={"size": (sensor_w, sensor_h)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    print(f"Camera initialized: sensor={sensor_w}x{sensor_h}")
    return picam2


def init_actuators():
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

    print("Actuators initialized (servo centered, motor stopped)")
    return servo, motor


def create_perspective_matrix():
    M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)
    return M


# ==================================
# 이미지 캡처 & 전처리
# ==================================
def capture_and_preprocess(picam2, M):
    """
    1. 카메라 프레임 캡처
    2. BGR 변환
    3. BEV (bird-eye view) 변환
    4. HSV 변환
    """
    frame_rgb = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    bird_view = cv2.warpPerspective(frame_bgr, M, (640, 480))
    hsv = cv2.cvtColor(bird_view, cv2.COLOR_BGR2HSV)

    return frame_bgr, bird_view, hsv


# ==================================
# 차선 기울기 계산 함수
# ==================================
def fit_lane_angle_deg(side_img):
    points = cv2.findNonZero(side_img)
    if points is None or len(points) < 50:
        return None

    line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line.flatten()

    # 위쪽 방향이 y+, 아래쪽이 y-인 경우 뒤집기
    if vy < 0:
        vx, vy = -vx, -vy

    angle_rad = math.atan2(vx, vy)
    angle_deg = math.degrees(angle_rad)

    angle_deg = max(-MAX_LANE_ANGLE, min(MAX_LANE_ANGLE, angle_deg))
    return angle_deg


def get_lane_angle_split(mask, roi_ratio):
    """
    전체 lane_mask에서 하단 roi_ratio 영역만 잘라서
    좌/우 나누어 기울기 계산
    """
    h, w = mask.shape

    roi_top = int(h * roi_ratio)
    roi = mask[roi_top:h, :]

    mid = w // 2
    left_roi = roi[:, :mid]
    right_roi = roi[:, mid:]

    angle_left = fit_lane_angle_deg(left_roi)
    angle_right = fit_lane_angle_deg(right_roi)

    if angle_left is None and angle_right is None:
        return None, "none", roi_top

    if angle_left is not None and angle_right is None:
        return angle_left, "left", roi_top

    if angle_left is None and angle_right is not None:
        return angle_right, "right", roi_top

    return (angle_left + angle_right) / 2.0, "both", roi_top


# ==================================
# 모드 판별: 초록색 플래그 (로터리 등)
# ==================================
def detect_green_mode(hsv):
    green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    green_count = cv2.countNonZero(green_mask)

    if green_count > GREEN_PIXEL_THRESHOLD:
        roi_ratio = ROI_RATIO_ROTARY
        mode_status = "ROTARY (Green)"
    else:
        roi_ratio = ROI_RATIO_NORMAL
        mode_status = "NORMAL"

    return roi_ratio, mode_status, green_mask, green_count


# ==================================
# 횡단보도 / 주차 구역 판별
# ==================================
def IsCrosswalk(hsv):
    """
    빨간색 픽셀 수를 세어서 횡단보도 영역인지 판별용
    (현재는 단순히 전체 BEV에서 카운트, 필요하면 ROI 줄이면 됨)
    """
    mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_count = cv2.countNonZero(red_mask)
    return red_count, red_mask


def IsParking(hsv):
    """
    파란색 픽셀 수를 세어서 주차 영역인지 판별용
    """
    blue_mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    blue_count = cv2.countNonZero(blue_mask)
    return blue_count, blue_mask


# ==================================
# 주행 모드별 동작 (추후 상세 구현)
# ==================================
def CrosswalkDrive(servo, motor):
    """
    횡단보도 모드 동작:
    - 예시: 속도 줄이기 or 정지 등
    - 실제 로직은 나중에 네가 구현하면 됨
    """
    motor.forward(0.3)
    
    # servo.angle 그대로 두고 직진 유지
    print("[MODE] CROSSWALK: Slow driving")


def ParkingDrive(servo, motor):
    """
    주차 모드 동작:
    - 예시: 정지 / 특정 패턴으로 주차 등
    """
    motor.stop()
    print("[MODE] PARKING: Stop for parking")


def LaneDrive(hsv, servo, motor, roi_ratio):
    """
    기본 차선 추종 주행
    - lane_mask 생성
    - 기울기 계산
    - 서보/모터 제어
    """
    # 차선 마스크 (환경에 맞게 값 조절 필요)
    lane_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 0]),
        np.array([180, 255, 80])
    )

    lane_angle_deg = 0.0
    side = "none"

    result_angle, side, roi_top = get_lane_angle_split(lane_mask, roi_ratio)

    if result_angle is not None:
        lane_angle_deg = result_angle

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
        servo.angle = SERVO_CENTER_ANGLE
        motor.stop()

    return lane_angle_deg, lane_mask, roi_top, side


# ==================================
# 디버그 화면 그리기
# ==================================
def draw_debug_view(bird_view,
                    green_mask,
                    lane_mask,
                    red_mask,
                    blue_mask,
                    roi_top,
                    mode_status,
                    green_count,
                    lane_angle_deg):
    bev_debug = bird_view.copy()
    h, w = bev_debug.shape[:2]

    # 초록 영역 컨투어 표시
    if green_mask is not None:
        contours, _ = cv2.findContours(
            green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(bev_debug, contours, -1, (0, 255, 0), 2)

    # ROI 라인
    cv2.line(bev_debug, (0, roi_top), (w, roi_top), (0, 0, 255), 3)

    # 텍스트 정보
    cv2.putText(bev_debug, f"Mode: {mode_status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(bev_debug, f"GreenPx: {green_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(bev_debug, f"Angle: {lane_angle_deg:.1f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 디버그 창 출력
    cv2.imshow("BEV Debug", bev_debug)

    if lane_mask is not None:
        cv2.imshow("Lane Mask", lane_mask)
    if green_mask is not None:
        cv2.imshow("Green Mask", green_mask)
    if red_mask is not None:
        cv2.imshow("Red Mask", red_mask)
    if blue_mask is not None:
        cv2.imshow("Blue Mask", blue_mask)


# ==================================
# 메인 루프
# ==================================
def main():
    picam2 = init_camera()
    servo, motor = init_actuators()
    M = create_perspective_matrix()

    print("Auto Lane Tracking with Color-based Mode Switching Start")

    # 초기 ROI 비율
    current_roi_ratio = ROI_RATIO_NORMAL

    try:
        while True:
            # 1. 이미지 캡처 & 전처리
            frame, bird_view, hsv = capture_and_preprocess(picam2, M)
            cv2.imshow("Raw Camera", frame)

            # 2. 초록색 플래그로 ROI/모드 결정
            current_roi_ratio, mode_status, green_mask, green_count = detect_green_mode(hsv)

            # 3. 색 기반 모드 판별
            CW_pixel_count_Red, red_mask = IsCrosswalk(hsv)
            CW_pixel_count_Blue, blue_mask = IsParking(hsv)

            lane_angle_deg = 0.0
            lane_mask = None
            roi_top = int(480 * current_roi_ratio)

            # 우선순위: 횡단보도 > 주차 > 차선 주행
            if CW_pixel_count_Red > CW_THRESHOLD_RED:
                CrosswalkDrive(servo, motor)
                print(f"RedPx={CW_pixel_count_Red} > {CW_THRESHOLD_RED} → CROSSWALK")
            elif CW_pixel_count_Blue > CW_THRESHOLD_BLUE:
                ParkingDrive(servo, motor)
                print(f"BluePx={CW_pixel_count_Blue} > {CW_THRESHOLD_BLUE} → PARKING")
            else:
                # 기본 차선 추종 주행
                lane_angle_deg, lane_mask, roi_top, side = LaneDrive(
                    hsv, servo, motor, current_roi_ratio
                )

            # 4. 디버그 출력
            draw_debug_view(
                bird_view,
                green_mask,
                lane_mask,
                red_mask,
                blue_mask,
                roi_top,
                mode_status,
                green_count,
                lane_angle_deg
            )

            # 종료 키
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    finally:
        print("Stopping...")
        picam2.stop()
        motor.stop()
        servo.angle = SERVO_CENTER_ANGLE
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
