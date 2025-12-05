from picamera2 import Picamera2
import cv2
import numpy as np
import time
import math

from gpiozero import AngularServo, Motor, DistanceSensor

# ===========================
# 초음파 센서
# ===========================
ULTRA_ECHO = 11
ULTRA_TRIG = 8
sensor = DistanceSensor(echo=ULTRA_ECHO, trigger=ULTRA_TRIG, max_distance=2.0)

# ===========================
# 파라미터
# ===========================
# 서보
SERVO_PIN = 18
SERVO_MIN_ANGLE = 0
SERVO_MAX_ANGLE = 180
SERVO_CENTER_ANGLE = 90

# DC 모터
MOTOR_FORWARD_PIN = 14
MOTOR_BACKWARD_PIN = 15
MOTOR_ENABLE_PIN = 23
BASE_SPEED = 0.4

# 주행 관련
K_ANGLE = 2.0
MAX_LANE_ANGLE = 60.0
HARD_TURN_THRESHOLD = 40.0
ROI_TOP_RATIO = 0.5

# 횡단보도
STOP_TIME = 4.0
BLIND_RUN_TIME = 2.0
CW_THRESHOLD = 14000

# BEV 변환 좌표
src_points = np.float32([(80, 70), (0, 280), (640, 280), (560, 70)])
dst_points = np.float32([(0, 0), (0, 480), (640, 480), (640, 0)])

# ===========================
# Init 함수
# ===========================
def Init():
    picam2 = Picamera2()
    sensor_w, sensor_h = picam2.sensor_resolution

    M = cv2.getPerspectiveTransform(src_points, dst_points)

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

    return picam2, M, servo, motor


# ===========================
# Lane Detection
# ===========================
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


def get_lane_angle_split(mask):
    h, w = mask.shape

    roi_top = int(h * ROI_TOP_RATIO)
    roi = mask[roi_top:h, :]

    mid = w // 2
    left_roi = roi[:, :mid]
    right_roi = roi[:, mid:]

    angle_left = fit_lane_angle_deg(left_roi)
    angle_right = fit_lane_angle_deg(right_roi)

    if angle_left is None and angle_right is None:
        return None

    if angle_left is not None and angle_right is None:
        return angle_left, "left", roi_top

    if angle_left is None and angle_right is not None:
        return angle_right, "right", roi_top

    return (angle_left + angle_right) / 2.0, "both", roi_top


def Lane_Drive(result, servo, motor):
    if result is None:
        side = "none"
        lane_angle_deg = 0
        steering_angle = SERVO_CENTER_ANGLE
        roi_top = int(480 * ROI_TOP_RATIO)
        servo.angle = SERVO_CENTER_ANGLE
        motor.stop()
        return roi_top, side, lane_angle_deg, steering_angle

    lane_angle_deg, side, roi_top = result

    if lane_angle_deg <= -HARD_TURN_THRESHOLD:
        steering_angle = SERVO_MIN_ANGLE
    elif lane_angle_deg >= HARD_TURN_THRESHOLD:
        steering_angle = SERVO_MAX_ANGLE
    else:
        steering_angle = SERVO_CENTER_ANGLE - K_ANGLE * lane_angle_deg

    steering_angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, steering_angle))
    servo.angle = steering_angle
    motor.forward(BASE_SPEED)

    return roi_top, side, lane_angle_deg, steering_angle


# ===========================
# Debug 화면
# ===========================
def View_debug(bird_view, roi_top, side, lane_angle_deg, steering_angle, mask):
    bev_debug = bird_view.copy()
    h, w = bev_debug.shape[:2]

    cv2.rectangle(bev_debug, (0, roi_top), (w-1, h-1), (0,255,255), 1)
    cv2.putText(bev_debug, f"side: {side}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)
    cv2.putText(bev_debug, f"lane_angle: {lane_angle_deg:.1f}",
                (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)
    cv2.putText(bev_debug, f"steer: {steering_angle:.1f}",
                (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)

    cv2.imshow("Mask", mask)
    cv2.imshow("BEV Debug", bev_debug)


# ===========================
# 횡단보도 RED Detection
# ===========================
def IsCrosswalk(hsv, servo, motor):
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    CW_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    CW_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    CW_mask = cv2.addWeighted(CW_mask1, 1.0, CW_mask2, 1.0, 0.0)

    CW_pixel_count = cv2.countNonZero(CW_mask)

    if CW_pixel_count > CW_THRESHOLD:
        print(f"[Crosswalk] {CW_pixel_count} → STOP")

        motor.stop()
        time.sleep(STOP_TIME)

        servo.angle = SERVO_CENTER_ANGLE
        motor.forward(BASE_SPEED)
        time.sleep(BLIND_RUN_TIME)

        return True, CW_pixel_count

    return False, CW_pixel_count


# ===========================
# BLUE Parking Zone Detection
# ===========================
def IsParkingZone(hsv, servo, motor):
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([140, 255, 255])

    P_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    P_pixel_count = cv2.countNonZero(P_mask)
    print(P_pixel_count)
    if P_pixel_count > CW_THRESHOLD:
        print(f"[Parking Zone] {P_pixel_count} → PARKING")
        motor.stop()
        time.sleep(0.5)
        parking(servo, motor)
        return True, P_pixel_count

    return False, P_pixel_count


# ===========================
# Parking Function
# ===========================
# ===========================
# Parking Function (수정됨)
# ===========================
def parking(servo, motor, picam2, M):
    print("=== Parking mode start ===")
    
    PARK_SPEED = BASE_SPEED * 1.5  # 속도 조정

    # 1. 초기 진입 (기존 코드 유지: 좌측으로 살짝 진입)
    servo.angle = SERVO_MIN_ANGLE
    print("Steering: Left MAX (Entry)")
    motor.forward(PARK_SPEED)
    time.sleep(1.0) 

    print("=== Searching for Parking Spot (Green Line Following) ===")
    
    # 초록색 라인 HSV 범위 설정 (환경에 따라 튜닝 필요)
    # 대략적인 초록색 범위
    lower_green = np.array([40, 80, 80])
    upper_green = np.array([90, 255, 255])

    hold_time = 0.0
    CHECK_INTERVAL = 0.05
    TARGET_DIST = 20.0  # 20cm 이상이면 빈 공간으로 인식

    try:
        while True:
            # --- 1. 영상 처리 (초록색 라인 추적) ---
            frame = picam2.capture_array()
            bird_view = cv2.warpPerspective(frame, M, (640, 480))
            hsv = cv2.cvtColor(bird_view, cv2.COLOR_BGR2HSV)

            # 초록색 마스크 추출
            G_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # 라인 각도 계산 (기존 get_lane_angle_split 로직 활용 또는 단순화)
            # 여기서는 오른쪽 초록색 라인을 따라간다고 가정하고 오른쪽 ROI만 봅니다.
            h, w = G_mask.shape
            roi = G_mask[int(h*0.5):, w//2:] # 화면 오른쪽 아래 절반
            
            angle_deg = fit_lane_angle_deg(roi) # 기존에 정의된 함수 재사용
            
            # 조향 제어 (라인이 안 보이면 직진 유지)
            target_angle = SERVO_CENTER_ANGLE
            if angle_deg is not None:
                # 라인 각도에 비례해 조향 (P제어)
                # 라인이 기울어진 만큼 반대로 꺾어서 평행 유지
                target_angle = SERVO_CENTER_ANGLE - (K_ANGLE * angle_deg)
                target_angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, target_angle))
                print(f"[Line] Angle: {angle_deg:.1f} | Servo: {target_angle:.1f}")
            else:
                print("[Line] Not found -> Go Straight")
            
            servo.angle = target_angle

            # --- 2. 초음파 거리 체크 ---
            dist_m = sensor.distance
            dist_cm = dist_m * 100
            
            # 노이즈 방지를 위해 약간의 필터링이나 로그 출력
            # print(f"Dist: {dist_cm:.1f} cm")

            # --- 3. 빈 공간 판단 로직 ---
            if dist_cm >= TARGET_DIST:
                hold_time += CHECK_INTERVAL
                print(f"!!! SPACE FOUND ({dist_cm:.1f}cm) - Holding {hold_time:.2f}s")

                # 일정 시간 이상 공간이 감지되면 주차 시작
                if hold_time >= 0.5: # 0.5초 이상 빈 공간 지속 시
                    print(">>> Parking Spot Confirmed! Starting Reverse Parking <<<")
                    motor.stop()
                    time.sleep(1.0)
                    break # 루프 탈출 후 후진 로직으로 이동
            else:
                hold_time = 0.0 # 다시 장애물이 감지되면 타이머 리셋
            
            # --- 4. 주행 ---
            motor.forward(PARK_SPEED)
            
            # 디버깅용 화면 출력 (필요시 주석 해제)
            # cv2.imshow("Parking Green", roi)
            # cv2.waitKey(1)

    except Exception as e:
        print(f"Error in parking loop: {e}")

    # ===========================
    # 후진 주차 로직 (Reverse)
    # ===========================
    
    # 1. 정렬을 위해 조금 더 앞으로 가서 멈춤
    print("Alignment Forward...")
    servo.angle = SERVO_CENTER_ANGLE
    motor.forward(PARK_SPEED)
    time.sleep(0.5) 
    motor.stop()
    time.sleep(1.0)

    # 2. 후진 진입 (오른쪽으로 꺾으며 후진)
    print("Reverse Maneuver Start")
    servo.angle = SERVO_MAX_ANGLE # 핸들 오른쪽 끝까지 (후진하여 엉덩이를 왼쪽으로 넣음)
    time.sleep(0.5)

    motor.backward(PARK_SPEED)
    time.sleep(3.0) # **중요: 주차 공간 깊이에 따라 시간 조절 필요**

    # 3. 정지 및 완료
    motor.stop()
    servo.angle = SERVO_CENTER_ANGLE
    print("Parking Completed.")
    print("=== Parking mode end ===")
# ===========================
# main
# ===========================
def main():
    picam2, M, servo, motor = Init()

    try:
        while True:
            frame = picam2.capture_array()
            bird_view = cv2.warpPerspective(frame, M, (640, 480))

            hsv = cv2.cvtColor(bird_view, cv2.COLOR_BGR2HSV)

            # 횡단보도 체크
            is_cross, _ = IsCrosswalk(hsv, servo, motor)
            if is_cross:
                continue

            # 파란색 주차구역 체크
            is_parking, _ = IsParkingZone(hsv, servo, motor)
            if is_parking:
                return

            # 차선 유지
            L_mask = cv2.inRange(hsv, np.array([0,0,0]), np.array([180,255,80]))
            result = get_lane_angle_split(L_mask)

            roi_top, side, lane_angle_deg, steering_angle = Lane_Drive(result, servo, motor)

            View_debug(bird_view, roi_top, side, lane_angle_deg, steering_angle, L_mask)

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