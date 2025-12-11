from picamera2 import Picamera2
import cv2
import numpy as np
import time
import math

from gpiozero import AngularServo, Motor, DistanceSensor

# ============ parameter ============
# 서보모터
SERVO_PIN = 18
SERVO_MIN_ANGLE = 0
SERVO_MAX_ANGLE = 180
SERVO_CENTER_ANGLE = 90

# DC모터
MOTOR_FORWARD_PIN = 14
MOTOR_BACKWARD_PIN = 15
MOTOR_ENABLE_PIN = 23
BASE_SPEED = 0.6

# 주행
K_ANGLE = 3.0               # 선형 매핑 게인

MAX_LANE_ANGLE = 60.0            # fitLine 클램핑 기준
HARD_TURN_THRESHOLD = 45.0       # ±45도 넘으면 최대 조향

ROI_RATIO_NORMAL = 0.5
ROI_RATIO_ROTARY = 0.25

src_points = np.float32([(40, 50), (0, 280), (640, 280), (600, 50)])
dst_points = np.float32([(0, 0), (0, 480), (640, 480), (640, 0)])

# 로터리 
GREEN_PIXEL_THRESHOLD = 300
RED_PIXEL_THRESHOLD = 24000
BLUE_PIXEL_THRESHOLD = 3000

# 횡단보도
STOP_TIME = 1.5      # 정지 시간 (초)
BLIND_RUN_TIME = 1.0 # 탈출 직진 시간 (초)

# ============ Init ============
def Init():
    # CAM1 포트 카메라 사용 (CAM0 쓰고 싶으면 camera_num=0)
    picam2 = Picamera2(camera_num=1)

    sensor_w, sensor_h = picam2.sensor_resolution
    M = cv2.getPerspectiveTransform(src_points, dst_points)
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
    time.sleep(2)
    
    return picam2, M, servo, motor


# ============ Lane Keeping ============
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

def get_lane_angle_split(mask, roi_ratio):
    h, w = mask.shape

    roi_top = int(h * roi_ratio)
    roi = mask[roi_top:h, :]

    mid = w // 2
    left_roi = roi[:, :mid]
    right_roi = roi[:, mid:]

    angle_left = fit_lane_angle_deg(left_roi)
    angle_right = fit_lane_angle_deg(right_roi)

    if angle_left is None and angle_right is None:
        return None, None, roi_top

    if angle_left is not None and angle_right is None:
        return angle_left, "left", roi_top

    if angle_left is None and angle_right is not None:
        return angle_right, "right", roi_top

    return (angle_left + angle_right) / 2.0, "both", roi_top

def Lane_Drive(result, servo, motor, current_roi_ratio):
    if result is not None and result[0] is not None:
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
    else:
        side = "none"
        roi_top = int(480 * current_roi_ratio)
        lane_angle_deg = 0
        servo.angle = SERVO_CENTER_ANGLE
        motor.stop()
    return roi_top, lane_angle_deg


# ============ Color Detect ============
def Color_Define(hsv):
    LOWER_GREEN = np.array([40, 50, 50])
    UPPER_GREEN = np.array([90, 255, 255])
    GREEN_MASK = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    LOWER_RED1 = np.array([0, 100, 100])
    UPPER_RED1 = np.array([10, 255, 255])
    LOWER_RED2 = np.array([170, 100, 100])
    UPPER_RED2 = np.array([180, 255, 255])
    RED_MASK1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    RED_MASK2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    RED_MASK = cv2.addWeighted(RED_MASK1, 1.0, RED_MASK2, 1.0, 0.0)
    
    LOWER_BLUE = np.array([100, 150, 0])
    UPPER_BLUE = np.array([140, 255, 255])
    BLUE_MASK = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    GREEN_PIXEL_COUNT = cv2.countNonZero(GREEN_MASK)
    RED_PIXEL_COUNT = cv2.countNonZero(RED_MASK)
    BLUE_PIXEL_COUNT = cv2.countNonZero(BLUE_MASK)

    # 디버그용으로 픽셀 수 보고 싶으면 아래 주석 해제
    # print("G:", GREEN_PIXEL_COUNT, "R:", RED_PIXEL_COUNT, "B:", BLUE_PIXEL_COUNT)

    if GREEN_PIXEL_COUNT > GREEN_PIXEL_THRESHOLD:
        return "GREEN"
    elif RED_PIXEL_COUNT > RED_PIXEL_THRESHOLD:
        return "RED"
    elif BLUE_PIXEL_COUNT > BLUE_PIXEL_THRESHOLD:
        return "BLUE"
    else:
        return "NONE"   


# ============ Rotary ============
def IsRotary():
    return ROI_RATIO_ROTARY  # 0.25 (넓게 봄)


# ============ Crosswalk ============
def IsCrosswalk(servo, motor):   
    motor.stop()
    time.sleep(STOP_TIME)
        
    motor.forward(speed=BASE_SPEED)
    servo.angle = SERVO_CENTER_ANGLE
    time.sleep(BLIND_RUN_TIME)


# ============ Parking ============
def parking(picam2, servo, motor):
    pass

# ============ Debug ============
def View_debug(bird_view, mask, roi_top, lane_angle_deg, color_status):
    bev_debug = bird_view.copy()
    h, w = bev_debug.shape[:2]

    cv2.line(bev_debug, (0, roi_top), (w, roi_top), (0, 0, 255), 3)
    
    cv2.putText(bev_debug, f"Mode: {color_status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(bev_debug, f"Angle: {lane_angle_deg:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Mask", mask)
    cv2.imshow("BEV Debug", bev_debug)


# ============ main ============
def main():
    picam2, M, servo, motor = Init()
    
    current_roi_ratio = ROI_RATIO_NORMAL
    CW_Flag = 0
    Parking_Flag = 0
    
    sensor = DistanceSensor(echo = 8, trigger = 11)
    turning_time = 4.0
    forward_speed = 0.6

    target_distance = 0.20  
    distance_count = 0
    target_count = 50
    
    try:
        while True:
            frame = picam2.capture_array()
            bird_view = cv2.warpPerspective(frame, M, (640, 480))
            hsv = cv2.cvtColor(bird_view, cv2.COLOR_BGR2HSV)
            
            # ============ Color Detection ============
            color_status = Color_Define(hsv)
            if color_status == "GREEN":
                current_roi_ratio = IsRotary()
                
            elif color_status == "RED" and CW_Flag == 0:
                IsCrosswalk(servo, motor)
                CW_Flag = 1
                continue

            elif color_status == "BLUE":
                # 파란색 → 주차 모드 진입
                motor.stop()
                time.sleep(1)
                servo.angle = SERVO_MIN_ANGLE
                time.sleep(1)
                motor.forward(forward_speed)
                time.sleep(turning_time)
                servo.angle = SERVO_CENTER_ANGLE
                motor.stop()
                time.sleep(1)
                #BASE_SPEED = 0.4
                
                Parking_Flag = 1

            elif Parking_Flag == 1:
                dist = sensor.distance
                print(f"dist: {dist:.2f} m")
                
            else:
                current_roi_ratio = ROI_RATIO_NORMAL

            # 기본 차선 인식 (검정 계열 기준)
            L_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
            result = get_lane_angle_split(L_mask, current_roi_ratio)
            roi_top, lane_angle_deg = Lane_Drive(result, servo, motor, current_roi_ratio)

            # ============ 시각화 ============
            View_debug(bird_view, L_mask, roi_top, lane_angle_deg, color_status)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        motor.stop()
        motor.close()
        picam2.stop()
        servo.angle = SERVO_CENTER_ANGLE
        servo.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
