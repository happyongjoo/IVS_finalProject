from picamera2 import Picamera2
import cv2
import numpy as np
import time
from sklearn.linear_model import RANSACRegressor
from gpiozero import AngularServo, Motor

# ==========================================
# [튜닝 파라미터]
# ==========================================
STEER_SENSITIVITY = 0.8
LOOK_AHEAD_RATIO = 0.6
BASE_SPEED = 0.5
CORNER_SLOW_DOWN = 0.003

# [NEW] 오른쪽 차선 추종 전용 파라미터
TARGET_OFFSET = 300       # 오른쪽 차선에서 '왼쪽으로' 300픽셀 떨어진 곳을 중앙으로 잡음
STANDARD_LANE_WIDTH = 350 # 오른쪽 차선을 놓쳤을 때, 왼쪽 차선 + 350픽셀을 오른쪽으로 가정
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
    time.sleep(1)

    src_points = np.float32([(80, 70), (0, 280), (640, 280), (560, 70)])
    dst_points = np.float32([(0, 0), (0, 480), (640, 480), (640, 0)])

    last_left_peak = None
    last_right_peak = None
    
    current_servo_angle = 90
    current_speed = 0.0

    print("Right Lane Follower Started. Press 'q' to exit.")

    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            height, width = frame.shape[:2]
            debug_frame = frame.copy()
            
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            bird_view = cv2.warpPerspective(frame, M, (width, height))
            hsv = cv2.cvtColor(bird_view, cv2.COLOR_BGR2HSV)
            
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

                l_peak = None
                r_peak = None

                # 왼쪽 피크 찾기 (오른쪽 놓쳤을 때 대비용)
                if np.sum(left_region) > 0:
                    l_peak = np.argmax(left_region) + 50
                    last_left_peak = l_peak
                else:
                    l_peak = last_left_peak

                # 오른쪽 피크 찾기 (메인)
                if np.sum(right_region) > 0:
                    r_peak = np.argmax(right_region) + 340
                    last_right_peak = r_peak
                else:
                    r_peak = last_right_peak

                # =================================================
                # [핵심] 무조건 오른쪽 차선 기준 계산
                # =================================================
                center_point = None

                if r_peak is not None:
                    # 1순위: 오른쪽 차선이 보이면, 거기서 TARGET_OFFSET 만큼 왼쪽으로 이동
                    center_point = r_peak - TARGET_OFFSET
                    # 시각화 (오른쪽 차선에 빨간점)
                    cv2.circle(window_debug, (r_peak, i), 5, (0, 0, 255), -1)

                elif l_peak is not None:
                    # 2순위: 오른쪽이 안 보이고 왼쪽만 보이면?
                    # "왼쪽 + 도로폭"을 가상의 오른쪽 차선으로 생각하고 계산
                    virtual_right = l_peak + STANDARD_LANE_WIDTH
                    center_point = virtual_right - TARGET_OFFSET
                    # 시각화 (가상 오른쪽 위치에 노란점)
                    cv2.circle(window_debug, (virtual_right, i), 3, (0, 255, 255), -1)

                if center_point is not None:
                    lane_points.append((center_point, i))
                    # 우리가 가야 할 목표 지점 (초록점)
                    cv2.circle(window_debug, (int(center_point), i), 5, (0, 255, 0), -1)

            # ---------------------------------------------------------
            # 주행 제어 (RANSAC)
            # ---------------------------------------------------------
            if len(lane_points) > 3:
                lane_points_np = np.array(lane_points)
                X = lane_points_np[:, 0]
                Y = lane_points_np[:, 1]

                try:
                    ransac = RANSACRegressor()
                    ransac.fit(Y.reshape(-1, 1), X)
                    target_y = height * LOOK_AHEAD_RATIO 
                    target_x = ransac.predict(np.array([[target_y]]))[0]

                    error = target_x - (width / 2)
                    calculated_angle = 90 + (error * STEER_SENSITIVITY)
                    current_servo_angle = np.clip(calculated_angle, 0, 180)

                    # 속도 제어
                    turn_error = abs(current_servo_angle - 90)
                    current_speed = np.clip(BASE_SPEED - (turn_error * CORNER_SLOW_DOWN), 0.0, 1.0)

                    servo.angle = current_servo_angle
                    motor.forward(current_speed)

                    # 시각화
                    cv2.circle(window_debug, (int(target_x), int(target_y)), 10, (255, 0, 0), -1)
                    
                except Exception:
                    pass
            else:
                motor.stop()

            # 상태 텍스트
            cv2.putText(debug_frame, "Mode: RIGHT ONLY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(debug_frame, f"Ang: {current_servo_angle:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            pts = src_points.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(debug_frame, [pts], True, (0, 0, 255), 2)
            cv2.imshow("Lane Detection", window_debug)
            cv2.imshow("Driver View", debug_frame)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
            
    finally:
        motor.stop()
        servo.angle = 90
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()