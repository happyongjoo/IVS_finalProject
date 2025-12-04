from picamera2 import Picamera2
import cv2
import numpy as np
import time
import math

from gpiozero import AngularServo, Motor, DistanceSensor

# ===========================
# 초음파 센서
# ===========================
ULTRA_ECHO = 24
ULTRA_TRIG = 23
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
def parking(servo, motor):
    print("=== Parking mode start ===")

    servo.angle = SERVO_MIN_ANGLE
    print("Steering: Left MAX")
    time.sleep(2.0)

    print("Ultrasonic checking...")

    hold_time = 0.0
    CHECK_INTERVAL = 0.05
    TARGET_DIST = 20

    while True:
        dist_m = sensor.distance
        dist_cm = dist_m * 100

        print(f"[Parking] Distance: {dist_cm:.1f} cm")

        if dist_cm <= TARGET_DIST:
            hold_time += CHECK_INTERVAL
            print(f"  * Hold {hold_time:.2f} sec")

            if hold_time >= 1.0:
                print("▶ Gap reached → Forward 1 sec")
                motor.forward(0.25)
                time.sleep(1.0)

                motor.stop()
                time.sleep(0.3)

                # 후진 maneuver
                print("▶ Reverse maneuver...")

                servo.angle = SERVO_MIN_ANGLE
                time.sleep(0.3)

                motor.backward(0.25)
                time.sleep(3.0)

                motor.stop()
                servo.angle = SERVO_CENTER_ANGLE
                print("Parking completed.")
                break

        else:
            hold_time = 0.0

        motor.forward(0.25)
        time.sleep(CHECK_INTERVAL)

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
