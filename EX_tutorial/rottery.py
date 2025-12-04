from picamera2 import Picamera2
import cv2
import numpy as np
import time
from sklearn.linear_model import RANSACRegressor
from gpiozero import AngularServo, Motor

# ==========================================
# [기본 주행 파라미터]
# ==========================================
STEER_SENSITIVITY = 0.8
LOOK_AHEAD_RATIO = 0.6
BASE_SPEED = 0.4
CORNER_SLOW_DOWN = 0.003

# ==========================================
# [파란색 트리거(로터리) 파라미터]
# ==========================================
BLUE_THRESHOLD = 500      # 파란색 픽셀이 이 값보다 많으면 로터리로 인식 (크기 조절 필요)
RIGHT_TURN_ANGLE = 160    # 파란색 감지 시 꺾을 각도 (180이 최대 우회전)
RIGHT_TURN_SPEED = 0.4    # 우회전 시 속도 (천천히)
# ==========================================

def main():
    servo = AngularServo(18, min_angle=0, max_angle=180)
    motor = Motor(forward=14, backward=15, enable=23, pwm=True)

    picam2 = Picamera2()
    sensor_w, sensor_h = picam2.sensor_resolution

    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        raw={"size": (sensor_w, sensor_h)}
    )
    picam2.configure(config)
    picam2.set_controls({"ScalerCrop": (0, 0, sensor_w, sensor_h)})
    picam2.start()
    time.sleep(2) # 초기화 시간 충분히

    src_points = np.float32([(80, 70), (0, 280), (640, 280), (560, 70)])
    dst_points = np.float32([(0, 0), (0, 480), (640, 480), (640, 0)])

    last_left_peak = None
    last_right_peak = None
    
    current_servo_angle = 90
    current_speed = 0.0

    print("Driving Started. Blue line triggers right turn.")

    try:
        while True:
            frame = picam2.capture_array()
            height, width = frame.shape[:2]
            debug_frame = frame.copy()
            
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            bird_view = cv2.warpPerspective(frame, M, (width, height))
            hsv = cv2.cvtColor(bird_view, cv2.COLOR_BGR2HSV)
            
            # -----------------------------------------------------
            # 1. 색상 마스크 생성 (검은색 차선 & 파란색 트리거)
            # -----------------------------------------------------
            # 검은색 (차선)
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 80]) 
            lane_mask = cv2.inRange(hsv, lower_black, upper_black)

            # [NEW] 파란색 (로터리 트리거) - HSV 값 튜닝 필요할 수 있음
            # 일반적인 파란색 범위: H(100~140), S(100~255), V(50~255)
            lower_blue = np.array([100, 100, 50])
            upper_blue = np.array([140, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # 파란색 픽셀 개수 세기 (화면 하단부 절반에서만 검사하면 더 정확함)
            blue_count = cv2.countNonZero(blue_mask)

            # -----------------------------------------------------
            # 2. 주행 로직 분기 (파란색 감지 vs 일반 주행)
            # -----------------------------------------------------
            is_rotary_mode = False

            if blue_count > BLUE_THRESHOLD:
                # [CASE A] 파란색 라인 감지됨 -> 강제 우회전
                is_rotary_mode = True
                current_servo_angle = RIGHT_TURN_ANGLE
                current_speed = RIGHT_TURN_SPEED
                
                # 하드웨어 즉시 제어
                servo.angle = current_servo_angle
                motor.forward(current_speed)
                
                # 시각화 (화면에 파란색 경고)
                cv2.putText(debug_frame, "!!! BLUE LINE DETECTED !!!", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                
            else:
                # [CASE B] 일반 차선 주행 (Sliding Window & RANSAC)
                step_size = height // 10
                lane_points = []
                window_debug = bird_view.copy()

                for i in range(height, 0, -step_size):
                    y_start = max(0, i - step_size)
                    y_end = i
                    cropped_region = lane_mask[y_start:y_end, :]
                    histogram = np.sum(cropped_region, axis=0)

                    left_region = histogram[50:300]
                    right_region = histogram[340:590]

                    if np.sum(left_region) > 0:
                        left_peak = np.argmax(left_region) + 50
                        last_left_peak = left_peak
                    else:
                        left_peak = last_left_peak

                    if np.sum(right_region) > 0:
                        right_peak = np.argmax(right_region) + 340
                        last_right_peak = right_peak
                    else:
                        right_peak = last_right_peak

                    if left_peak is not None and right_peak is not None:
                        center_point = (left_peak + right_peak) // 2
                        lane_points.append((center_point, i))
                        cv2.circle(window_debug, (center_point, i), 5, (0, 255, 0), -1)

                if len(lane_points) > 3:
                    try:
                        lane_points_np = np.array(lane_points)
                        X = lane_points_np[:, 0]
                        Y = lane_points_np[:, 1]
                        
                        ransac = RANSACRegressor()
                        ransac.fit(Y.reshape(-1, 1), X)
                        target_y = height * LOOK_AHEAD_RATIO 
                        target_x = ransac.predict(np.array([[target_y]]))[0]

                        error = target_x - (width / 2)
                        calculated_angle = 90 + (error * STEER_SENSITIVITY)
                        current_servo_angle = np.clip(calculated_angle, 0, 180)

                        turn_error = abs(current_servo_angle - 90)
                        current_speed = np.clip(BASE_SPEED - (turn_error * CORNER_SLOW_DOWN), 0.0, 1.0)

                        servo.angle = current_servo_angle
                        motor.forward(current_speed)

                        cv2.circle(window_debug, (int(target_x), int(target_y)), 10, (0, 0, 255), -1)
                    except:
                        pass
                else:
                    motor.stop()

            # 상태 정보 출력
            mode_text = "MODE: ROTARY (RIGHT)" if is_rotary_mode else "MODE: LANE KEEPING"
            cv2.putText(debug_frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) if is_rotary_mode else (0,255,0), 2)
            cv2.putText(debug_frame, f"Ang: {current_servo_angle:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(debug_frame, f"BluePx: {blue_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            pts = src_points.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(debug_frame, [pts], True, (0, 0, 255), 2)
            
            # 파란색 마스크 확인용 창 추가 (디버깅 편하게)
            cv2.imshow("Driver View", debug_frame)
            # cv2.imshow("Lane Mask", lane_mask)
            cv2.imshow("Blue Trigger Mask", blue_mask) 

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
            
    finally:
        motor.stop()
        servo.angle = 90
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()