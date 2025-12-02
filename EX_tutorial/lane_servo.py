from picamera2 import Picamera2
import cv2
import numpy as np
import time

from gpiozero import AngularServo

# =========================
# 서보 / 제어 파라미터
# =========================
SERVO_PIN = 18          # ★ 사용중인 GPIO핀으로 변경
SERVO_MIN_ANGLE = 0   # 서보 왼쪽 최대 각도
SERVO_MAX_ANGLE = 180    # 서보 오른쪽 최대 각도
SERVO_CENTER_ANGLE = 90  # 직진 기준 각도

# 조향 민감도 (P 게인) – 나중에 주행하면서 튜닝
Kp_steer = 90.0

# 라인 오프셋 계산에 사용할 ROI 비율 (아래쪽 40%)
ROI_TOP_RATIO = 0.6


def get_line_offset_split(mask):
    """
    mask: BEV에서 나온 이진 마스크 (차선 = 255, 배경 = 0)
    반환:
        norm_error: -1.0 ~ 1.0 범위의 정규화 에러 (None이면 라인 없음)
        side: "left", "right", "both", "none" 중 하나
    """
    h, w = mask.shape

    # 아래쪽 일부만 사용 (차량 바로 앞 영역)
    roi_top = int(h * ROI_TOP_RATIO)
    roi = mask[roi_top:h, :]

    mid = w // 2
    left_roi = roi[:, :mid]
    right_roi = roi[:, mid:]

    def get_center_x(side_img, x_offset):
        points = cv2.findNonZero(side_img)
        if points is None:
            return None
        # 너무 적으면 노이즈로 간주
        if len(points) < 50:
            return None

        mean_x = float(np.mean(points[:, 0, 0])) + x_offset
        return mean_x

    left_x = get_center_x(left_roi, 0)
    right_x = get_center_x(right_roi, mid)

    center_x = w / 2

    if left_x is None and right_x is None:
        return None, "none"

    if left_x is not None and right_x is None:
        error_px = left_x - center_x
        norm_error = error_px / (w / 2)
        return norm_error, "left"

    if left_x is None and right_x is not None:
        error_px = right_x - center_x
        norm_error = error_px / (w / 2)
        return norm_error, "right"

    # 둘 다 있는 경우 → 차선 중앙 기준
    lane_center = (left_x + right_x) / 2.0
    error_px = lane_center - center_x
    norm_error = error_px / (w / 2)
    return norm_error, "both"


def main():
    # =========================
    # Picamera2 설정
    # =========================
    picam2 = Picamera2()

    sensor_w, sensor_h = picam2.sensor_resolution  # 예: (4608, 2592)

    config = picam2.create_video_configuration(
        main={
            "size": (640, 480),
            "format": "RGB888"
        },
        raw={
            "size": (sensor_w, sensor_h)
        }
    )

    picam2.configure(config)
    picam2.set_controls({"ScalerCrop": (0, 0, sensor_w, sensor_h)})

    picam2.start()
    time.sleep(1)

    meta = picam2.capture_metadata()
    print("Sensor resolution:", sensor_w, sensor_h)
    print("ScalerCrop:", meta.get("ScalerCrop", None))

    # =========================
    # 서보 초기화
    # =========================
    servo = AngularServo(
        SERVO_PIN,
        min_angle=SERVO_MIN_ANGLE,
        max_angle=SERVO_MAX_ANGLE,
    )
    servo.angle = SERVO_CENTER_ANGLE

    # =========================
    # 투시 변환 좌표 (640x480 기준)
    # =========================
    src_points = np.float32([(80, 70), (0, 280), (640, 280), (560, 70)])
    dst_points = np.float32([(0, 0), (0, 480), (640, 480), (640, 0)])

    print("Vision + Steering Start. Press 'q' or ESC to exit.")

    try:
        while True:
            # ---- 카메라 프레임 획득 ----
            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            height, width = frame.shape[:2]

            # ---- 원본 + ROI 폴리곤 디버그 ----
            debug_frame = frame.copy()
            pts = src_points.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(debug_frame, [pts], True, (0, 0, 255), 2)

            # ---- 투시 변환 (BEV) ----
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            bird_view = cv2.warpPerspective(frame, M, (width, height))

            # ---- HSV에서 블랙 라인 추출 ----
            hsv = cv2.cvtColor(bird_view, cv2.COLOR_BGR2HSV)
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 80])
            mask = cv2.inRange(hsv, lower_black, upper_black)

            # ---- 라인 오프셋 계산 ----
            offset, side = get_line_offset_split(mask)

            # ---- 조향 각도 계산 & 서보 제어 ----
            if offset is not None:
                # offset: -1 ~ +1 (양수면 라인이 오른쪽에 있음)
                steering_angle = SERVO_CENTER_ANGLE - Kp_steer * offset
                steering_angle = max(SERVO_MIN_ANGLE,
                                     min(SERVO_MAX_ANGLE, steering_angle))
                servo.angle = steering_angle
            else:
                steering_angle = SERVO_CENTER_ANGLE
                servo.angle = SERVO_CENTER_ANGLE

            # =========================
            # 디버그용 BEV + ROI + 목표각 표시
            # =========================
            bev_debug = bird_view.copy()
            h, w = bev_debug.shape[:2]
            roi_top = int(h * ROI_TOP_RATIO)

            # ROI 박스
            cv2.rectangle(bev_debug,
                          (0, roi_top), (w - 1, h - 1),
                          (0, 255, 255), 1)

            # 중앙선(녹색)
            cv2.line(bev_debug,
                     (w // 2, roi_top), (w // 2, h - 1),
                     (0, 255, 0), 2)

            # 목표선(파랑) – offset이 있을 때만
            if offset is not None:
                target_x = int((offset * (w / 2)) + w / 2)
                cv2.line(bev_debug,
                         (target_x, roi_top), (target_x, h - 1),
                         (255, 0, 0), 2)

            # 텍스트 정보 표시
            info1 = f"side: {side}"
            info2 = f"offset: {0.00 if offset is None else offset:.2f}"
            info3 = f"angle: {steering_angle:.1f} deg"
            cv2.putText(bev_debug, info1, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(bev_debug, info2, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(bev_debug, info3, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # ---- 창 출력 ----
            #cv2.imshow("1. Original with ROI (Picamera2)", debug_frame)
            #cv2.imshow("2. Bird's Eye View", bird_view)
            cv2.imshow("3. Black Lane Mask", mask)
            cv2.imshow("4. BEV Tracking Debug", bev_debug)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        picam2.stop()
        servo.angle = SERVO_CENTER_ANGLE  # 종료 시 정면
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
