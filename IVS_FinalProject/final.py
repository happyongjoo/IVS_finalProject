from picamera2 import Picamera2
import cv2
import numpy as np
import time
import math
from gpiozero import AngularServo, Motor, DistanceSensor

# ============ parameter ============
# 서보모터 (조향 장치)
SERVO_PIN = 18
SERVO_MIN_ANGLE = 0
SERVO_MAX_ANGLE = 180
SERVO_CENTER_ANGLE = 90

# DC모터 (구동 장치)
MOTOR_FORWARD_PIN = 14
MOTOR_BACKWARD_PIN = 15
MOTOR_ENABLE_PIN = 23
BASE_SPEED = 0.6
PARKING_SPEED = 0.4  # 주차 공간 탐색 시 전진 속도

# 초음파 센서 (요청하신 핀 번호 반영)
TRIG_PIN = 11
ECHO_PIN = 8

# 주행 제어 상수
K_ANGLE = 3.0              
MAX_LANE_ANGLE = 60.0      
HARD_TURN_THRESHOLD = 45.0 

ROI_RATIO_NORMAL = 0.5
ROI_RATIO_ROTARY = 0.25

# 원근 변환 좌표
src_points = np.float32([(40, 50), (0, 280), (640, 280), (600, 50)])
dst_points = np.float32([(0, 0), (0, 480), (640, 480), (640, 0)])

# 색상 임계값 (로터리, 횡단보도, 주차선)
GREEN_PIXEL_THRESHOLD = 3000
RED_PIXEL_THRESHOLD = 24000
BLUE_PIXEL_THRESHOLD = 300

# 횡단보도 미션 상수
STOP_TIME = 1.5
BLIND_RUN_TIME = 0.7

# ============ Init ============
def Init():
    """시스템 초기화: 카메라, 모터, 센서 설정"""
    picam2 = Picamera2()
    sensor_w, sensor_h = picam2.sensor_resolution
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        raw={"size": (sensor_w, sensor_h)}
    )

    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    servo = AngularServo(SERVO_PIN, min_angle=SERVO_MIN_ANGLE, max_angle=SERVO_MAX_ANGLE)
    servo.angle = SERVO_CENTER_ANGLE

    motor = Motor(forward=MOTOR_FORWARD_PIN, backward=MOTOR_BACKWARD_PIN, enable=MOTOR_ENABLE_PIN, pwm=True)
    motor.stop()
    
    # 초음파 센서 초기화 (GPIOZero 라이브러리 사용)
    # DistanceSensor는 내부적으로 BCM 핀 번호를 사용합니다.
    ultrasonic = DistanceSensor(echo=ECHO_PIN, trigger=TRIG_PIN)
    
    return picam2, M, servo, motor, ultrasonic

# ============ Lane Keeping Helpers ============
def fit_lane_angle_deg(side_img):
    points = cv2.findNonZero(side_img)
    if points is None or len(points) < 50:
        return None
    line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line.flatten()
    if vy < 0: vx, vy = -vx, -vy
    angle_rad = math.atan2(vx, vy)
    angle_deg = math.degrees(angle_rad)
    return max(-MAX_LANE_ANGLE, min(MAX_LANE_ANGLE, angle_deg))

def get_lane_angle_split(mask, roi_ratio):
    h, w = mask.shape
    roi_top = int(h * roi_ratio)
    roi = mask[roi_top:h, :]
    mid = w // 2
    
    angle_left = fit_lane_angle_deg(roi[:, :mid])
    angle_right = fit_lane_angle_deg(roi[:, mid:])

    if angle_left is None and angle_right is None: return None, None, roi_top
    if angle_left is not None and angle_right is None: return angle_left, "left", roi_top
    if angle_left is None and angle_right is not None: return angle_right, "right", roi_top
    return (angle_left + angle_right) / 2.0, "both", roi_top

def Lane_Drive(result, servo, motor, current_roi_ratio):
    if result is not None and result[0] is not None:
        lane_angle_deg, side, roi_top = result
        if lane_angle_deg <= -HARD_TURN_THRESHOLD: steering_angle = SERVO_MIN_ANGLE
        elif lane_angle_deg >= HARD_TURN_THRESHOLD: steering_angle = SERVO_MAX_ANGLE
        else: steering_angle = SERVO_CENTER_ANGLE - K_ANGLE * lane_angle_deg
        
        servo.angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, steering_angle))
        motor.forward(BASE_SPEED)
    else:
        roi_top = int(480 * current_roi_ratio)
        lane_angle_deg = 0
        servo.angle = SERVO_CENTER_ANGLE
        motor.stop()
    return roi_top, lane_angle_deg

def Color_Define(hsv):
    LOWER_GREEN = np.array([40, 50, 50])
    UPPER_GREEN = np.array([90, 255, 255])
    GREEN_MASK = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    LOWER_RED1 = np.array([0, 100, 100])
    UPPER_RED1 = np.array([10, 255, 255])
    LOWER_RED2 = np.array([170, 100, 100])
    UPPER_RED2 = np.array([180, 255, 255])
    RED_MASK = cv2.addWeighted(cv2.inRange(hsv, LOWER_RED1, UPPER_RED1), 1.0, cv2.inRange(hsv, LOWER_RED2, UPPER_RED2), 1.0, 0.0)
    
    LOWER_BLUE = np.array([100, 150, 0])
    UPPER_BLUE = np.array([140, 255, 255])
    BLUE_MASK = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    cv2.imshow("GREEN_MASK", GREEN_MASK)
    # cv2.imshow("RED_MASK", RED_MASK)
    cv2.imshow("BLUE_MASK", BLUE_MASK)

    if cv2.countNonZero(GREEN_MASK) > GREEN_PIXEL_THRESHOLD: return "GREEN"
    elif cv2.countNonZero(RED_MASK) > RED_PIXEL_THRESHOLD: return "RED"
    elif cv2.countNonZero(BLUE_MASK) > BLUE_PIXEL_THRESHOLD: return "BLUE"
    else: return "NONE" 

def IsRotary():
    return ROI_RATIO_ROTARY 
        
def IsCrosswalk(servo, motor): 
    motor.stop()
    time.sleep(STOP_TIME)
    motor.forward(speed=BASE_SPEED)
    servo.angle = SERVO_CENTER_ANGLE
    time.sleep(BLIND_RUN_TIME)

# ============ Parking Logic (Integrated) ============
def parking_sequence(servo, motor, ultrasonic, picam2):
    """
    파란색 인식 후 호출되는 함수.
    1. 전진하며 공간 탐색 (45cm 이상)
    2. 하드코딩된 후진 주차 수행
    """
    print("=== [Parking Mode Started] ===")
    
    # 카메라 자원 해제 (주차 동작에 집중하기 위함)
    # picam2.stop()
    # picam2.close()
    cv2.destroyAllWindows()

    # --- 1단계: 공간 탐색 (전진) ---
    print(">>> Searching for space (>= 45cm)...")
   
    motor.forward(0.5)
    servo.angle = SERVO_MIN_ANGLE  # 왼쪽 최대
    time.sleep(0.3)  


    servo.angle = SERVO_CENTER_ANGLE
    
    while True:
        # 거리 측정 (cm 변환)
        dist_cm = ultrasonic.distance
        print(f"dist: {dist_cm:.2f} m")
        
        if dist_cm >= 0.30 and dist_cm < 0.60:
            print(f"Space Found! ({dist_cm:.2f} m) Stopping...")
            motor.stop()            
            time.sleep(1.0) # 잠시 대기
            break
        
        time.sleep(0.1)

    # --- 2단계: 후방 주차 시퀀스 (하드코딩) ---
    # [파라미터 조정 영역] 아래 time.sleep 값을 수정하여 튜닝하세요.
    
    motor.forward(PARKING_SPEED)
    time.sleep(1.0)

    # 1. 왼쪽으로 핸들 최대한 꺾고 전진 (차체 비틀기)
    print("1. Left Turn & Forward")
    servo.angle = SERVO_MIN_ANGLE  # 왼쪽 최대
    time.sleep(0.5) # 서보 반응 대기
    motor.forward(PARKING_SPEED)
    time.sleep(1.5) # [시간 조절] 전진 시간 (차체 각도 만들기)


    # 2. 핸들 오른쪽 최대 꺾기 (후진 준비)
    print("2. Prepare Reverse (Right Max)")
    motor.stop()
    time.sleep(0.5)
    servo.angle = SERVO_MAX_ANGLE # 오른쪽 최대
    time.sleep(0.5) # 서보 반응 대기

    # 3. 후진 진입 (주차칸으로 엉덩이 넣기)
    print("3. Reverse into spot")
    motor.backward(PARKING_SPEED)
    time.sleep(1.9)

    # 4. 정지 및 핸들 중앙 정렬
    print("4. Align Center")
    motor.stop()
    servo.angle = SERVO_CENTER_ANGLE
    time.sleep(0.7)

    # 5. 최종 주차 (직진 후진으로 깊숙이)
    print("5. Final Parking")
    motor.backward(PARKING_SPEED)
    time.sleep(1.2)

    # 6. 종료
    print("=== Parking Completed ===")
    motor.stop()
    # 프로그램 종료

# ============ Debug ============
def View_debug(bird_view, mask, roi_top, lane_angle_deg, color_status):
    bev_debug = bird_view.copy()
    h, w = bev_debug.shape[:2]
    cv2.line(bev_debug, (0, roi_top), (w, roi_top), (0, 0, 255), 3)
    cv2.putText(bev_debug, f"Mode: {color_status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Mask", mask)
    cv2.imshow("BEV Debug", bev_debug)

# ============ main ============
def main():
    picam2, M, servo, motor, ultrasonic = Init()
    current_roi_ratio = ROI_RATIO_NORMAL
    CW_Flag = 0
    Rotary_Flag = 0

    try:
        while True:
            frame = picam2.capture_array()
            bird_view = cv2.warpPerspective(frame, M, (640, 480))
            hsv = cv2.cvtColor(bird_view, cv2.COLOR_BGR2HSV)
            
            # ============ Color Detection ============
            color_status = Color_Define(hsv)
            
            if color_status == "GREEN" and Rotary_Flag == 0:
                current_roi_ratio = IsRotary()
                Rotary_Flag = 1
                continue 
                
            elif color_status == "RED" and CW_Flag == 0:
                IsCrosswalk(servo, motor)
                CW_Flag = 1
                continue

            elif color_status == "BLUE":
                print("Blue Detected! Switching to Parking Mode...")
                # 파란색 감지 즉시 주차 함수 실행 (메인 루프 탈출)
                parking_sequence(servo, motor, ultrasonic, picam2)
                break 

            else:
                current_roi_ratio = ROI_RATIO_NORMAL

            # ============ Lane Keeping ============
            L_mask = cv2.inRange(hsv, np.array([0,0,0]), np.array([180,255,80]))
            result = get_lane_angle_split(L_mask, current_roi_ratio)
            roi_top, lane_angle_deg = Lane_Drive(result, servo, motor, current_roi_ratio)

            # ============ 시각화 ============
            View_debug(bird_view, L_mask, roi_top, lane_angle_deg, color_status)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        motor.stop()
        motor.close()
        try:
            picam2.stop()
            picam2.close()
        except: pass
        servo.angle = SERVO_CENTER_ANGLE
        servo.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()