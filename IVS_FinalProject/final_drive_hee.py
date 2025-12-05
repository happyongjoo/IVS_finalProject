from picamera2 import Picamera2
import cv2
import numpy as np
import time
import math

from gpiozero import AngularServo, Motor

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

ROI_RATIO_NORMAL = 0.6
ROI_RATIO_ROTARY = 0.25

src_points = np.float32([(40, 50), (0, 280), (640, 280), (600, 50)])
dst_points = np.float32([(0, 0), (0, 480), (640, 480), (640, 0)])

# 로터리 
LOWER_GREEN = np.array([40, 50, 50])
UPPER_GREEN = np.array([90, 255, 255])
GREEN_PIXEL_THRESHOLD = 300

# 횡단보도
STOP_TIME = 2.5  # 정지 시간 (초)
BLIND_RUN_TIME = 1.0 # 탈출 직진 시간 (초)
CW_THRESHOLD = 24000
CW_Flag = 0
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])


    
# ============ Init ============
def Init():
    picam2 = Picamera2()
    sensor_w, sensor_h = picam2.sensor_resolution
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        raw={"size": (sensor_w, sensor_h)}
    )

    picam2.configure(config)
    #picam2.set_controls({"ScalerCrop": (0, 0, sensor_w, sensor_h)})
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
        return None, None, roi_top

    if angle_left is not None and angle_right is None:
        return angle_left, "left", roi_top

    if angle_left is None and angle_right is not None:
        return angle_right, "right", roi_top

    return (angle_left + angle_right) / 2.0, "both", roi_top

def Lane_Drive(result, servo, motor, current_roi_ratio):
    if result is not None and result[0] is not None:
    # 항상 float 반환됨
        lane_angle_deg, side, roi_top = result
        # ---- 조향 로직 ----
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
        #steering_angle = SERVO_CENTER_ANGLE
        lane_angle_deg = 0
        servo.angle = SERVO_CENTER_ANGLE
        motor.stop()
    return roi_top, lane_angle_deg

# ============ Rotary ============
def IsRotary(hsv):
    green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    RT_pixel_count = cv2.countNonZero(green_mask)
    
    if RT_pixel_count > GREEN_PIXEL_THRESHOLD:
        current_roi_ratio = ROI_RATIO_ROTARY  # 0.25 (넓게 봄)
        mode_status = "ROTARY (Green)"
        print(f"Rotary ({RT_pixel_count})")
    else:
        current_roi_ratio = ROI_RATIO_NORMAL  # 0.5 (좁게 봄)
        mode_status = "NORMAL"
    return current_roi_ratio, green_mask, RT_pixel_count, mode_status
        
# ============ Crosswalk ============
def IsCrosswalk(hsv, servo, motor):   
    global CW_Flag

    CW_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    CW_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    CW_mask = cv2.addWeighted(CW_mask1, 1.0, CW_mask2, 1.0, 0.0)

    # 빨간색 픽셀 개수 세기
    CW_pixel_count = cv2.countNonZero(CW_mask)

    # 판단 및 제어
    if CW_pixel_count > CW_THRESHOLD and CW_Flag == 0:
        CW_Flag = 1
        print(f"Crosswalk ({CW_pixel_count}) -> Stop")
        # 일단 정지
        motor.stop()
        time.sleep(STOP_TIME)
        
        # 탈출
        motor.forward(speed=BASE_SPEED)
        servo.angle = SERVO_CENTER_ANGLE
        time.sleep(BLIND_RUN_TIME)
        
    return CW_pixel_count
                
# ============ Debug ============
def View_debug(bird_view, mask, green_mask, roi_top, lane_angle_deg, mode_status, RT_pixel_count, CW_pixel_count):
    bev_debug = bird_view.copy()
    h, w = bev_debug.shape[:2]

    # 초록색 영역
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bev_debug, contours, -1, (0, 255, 0), 2)

    # ROI
    cv2.line(bev_debug, (0, roi_top), (w, roi_top), (0, 0, 255), 3)
    
    cv2.putText(bev_debug, f"Mode: {mode_status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(bev_debug, f"Angle: {lane_angle_deg:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(bev_debug, f"Green Pixels: {RT_pixel_count}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(bev_debug, f"Red Pixels: {CW_pixel_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Mask", mask)
    cv2.imshow("BEV Debug", bev_debug)

# ============ main ============
def main():
    picam2, M, servo, motor = Init()

    current_roi_ratio = ROI_RATIO_NORMAL
    mode_status = "NORMAL"

    try:
        while True:
            frame = picam2.capture_array()
            # frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            bird_view = cv2.warpPerspective(frame, M, (640, 480))
            hsv = cv2.cvtColor(bird_view, cv2.COLOR_BGR2HSV)
            
            # ============ 회전 교차로 ============
            current_roi_ratio, green_mask, RT_pixel_count, mode_status  = IsRotary(hsv)
            
            # ============ 횡단보도 ============
            CW_pixel_count = IsCrosswalk(hsv, servo, motor)
            #if CW_pixel_count > CW_THRESHOLD:
            #    continue
            
            # ============ 차선 유지 ============
            L_mask = cv2.inRange(hsv, np.array([0,0,0]), np.array([180,255,80]))
            
            result = get_lane_angle_split(L_mask, current_roi_ratio)
            roi_top, lane_angle_deg = Lane_Drive(result, servo, motor, current_roi_ratio)

            # ============ 시각화 ============
            View_debug(bird_view, L_mask, green_mask, roi_top, lane_angle_deg, mode_status, RT_pixel_count, CW_pixel_count)
            
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
