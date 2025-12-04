from picamera2 import Picamera2
import cv2
import numpy as np
import time
from sklearn.linear_model import RANSACRegressor
from gpiozero import AngularServo, Motor

# ==========================================
# [튜닝 파라미터] 이 값들로 조향감을 조절하세요
# ==========================================
STEER_SENSITIVITY = 0.8  # (기존 0.4) 높을수록 핸들을 확 꺾습니다. (너무 높으면 차가 흔들림)
LOOK_AHEAD_RATIO = 0.5   # (기존 0.7) 0.0(맨위) ~ 1.0(맨아래). 멀리 볼수록(0.4~0.5) 코너를 미리 꺾습니다.
BASE_SPEED = 0.6         # 기본 주행 속도 (0.0 ~ 1.0)
CORNER_SLOW_DOWN = 0.003 # 코너에서 속도 줄이는 비율 (높을수록 코너에서 천천히 감)
# ==========================================

def main():
    # 서보 핀 18번, 모터 핀 설정 확인
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
    time.sleep(1)

    # 투시 변환 좌표 (사용자 환경에 맞게 미세 조정 필요할 수 있음)
    src_points = np.float32([(80, 70), (0, 280), (640, 280), (560, 70)])
    dst_points = np.float32([(0, 0), (0, 480), (640, 480), (640, 0)])

    last_left_peak = None
    last_right_peak = None
    
    current_servo_angle = 90
    current_speed = 0.0

    print("Driving Started. Press 'q' to exit.")

    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            height, width = frame.shape[:2]

            debug_frame = frame.copy()
            
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            bird_view = cv2.warpPerspective(frame, M, (width, height))

            hsv = cv2.cvtColor(bird_view, cv2.COLOR_BGR2HSV)
            
            # 검은색 차선 인식 범위 (조명에 따라 80 조절)
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 80]) 
            mask = cv2.inRange(hsv, lower_black, upper_black)

            step_size = height // 10
            lane_points = []
            window_debug = bird_view.copy()

            for i in range(height, 0, -step_size):
                y_start = max(0, i - step_size)
                y_end = i
                cropped_region = mask[y_start:y_end, :]
                
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

            # ---------------------------------------------------------
            # 조향 및 속도 계산 로직
            # ---------------------------------------------------------
            if len(lane_points) > 3:
                lane_points_np = np.array(lane_points)
                X = lane_points_np[:, 0]
                Y = lane_points_np[:, 1]

                try:
                    ransac = RANSACRegressor()
                    ransac.fit(Y.reshape(-1, 1), X)

                    # [수정됨] 더 멀리 보기 (0.7 -> 0.5)
                    # 멀리 볼수록 코너에서 중심점 오차가 커져서 핸들을 더 많이 꺾게 됨
                    target_y = height * LOOK_AHEAD_RATIO 
                    target_x = ransac.predict(np.array([[target_y]]))[0]

                    error = target_x - (width / 2)
                    
                    # [수정됨] 감도 적용 (0.4 -> 0.7)
                    calculated_angle = 90 + (error * STEER_SENSITIVITY)
                    current_servo_angle = np.clip(calculated_angle, 0, 180)

                    # 가변 속도: 핸들을 많이 꺾으면 속도 줄임
                    turn_error = abs(current_servo_angle - 90)
                    speed_reduction = turn_error * CORNER_SLOW_DOWN
                    
                    current_speed = BASE_SPEED - speed_reduction
                    current_speed = np.clip(current_speed, 0.0, 1.0)

                    # 하드웨어 제어
                    servo.angle = current_servo_angle
                    motor.forward(current_speed)

                    # 디버그용 시각화
                    cv2.circle(window_debug, (int(target_x), int(target_y)), 10, (0, 0, 255), -1)
                    cv2.line(window_debug, (int(width/2), int(target_y)), (int(target_x), int(target_y)), (0,255,255), 2)
                    
                except Exception:
                    pass
            else:
                # 차선을 놓치면 정지 (안전을 위해)
                motor.stop()

            # 정보 출력
            cv2.putText(debug_frame, f"Ang: {current_servo_angle:.1f}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(debug_frame, f"Spd: {current_speed:.2f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # ROI 박스 표시
            pts = src_points.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(debug_frame, [pts], True, (0, 0, 255), 2)

            cv2.imshow("Driver View", debug_frame)
            cv2.imshow("Lane Detection", window_debug)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            
    finally:
        motor.stop()
        servo.angle = 90
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()