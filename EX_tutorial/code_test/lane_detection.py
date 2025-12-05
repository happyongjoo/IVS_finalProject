from picamera2 import Picamera2
import cv2
import numpy as np
import time

def main():
    picam2 = Picamera2()

    # 센서 풀 해상도 정보
    sensor_w, sensor_h = picam2.sensor_resolution  # 예: (4608, 2592)

    # ⬇ rpicam-hello 느낌에 더 가깝게: raw에 풀센서, main은 우리가 쓸 해상도(640x480)
    config = picam2.create_video_configuration(
        main={
            "size": (640, 480),   # 처리/표시용 해상도
            "format": "RGB888"    # ISP 거친 RGB
        },
        raw={
            "size": (sensor_w, sensor_h)  # 풀센서 모드 강제 시도
        }
    )

    picam2.configure(config)

    # 가능하면 풀센서 전체를 쓰도록 ScalerCrop 요청
    picam2.set_controls({"ScalerCrop": (0, 0, sensor_w, sensor_h)})

    picam2.start()
    time.sleep(1)  # 워밍업

    # 현재 ScalerCrop 확인용 (디버그)
    meta = picam2.capture_metadata()
    print("Sensor resolution:", sensor_w, sensor_h)
    print("ScalerCrop:", meta.get("ScalerCrop", None))

    # 투시 변환용 포인트 (기존 코드 그대로 사용)
    # 이 좌표는 640x480 기준으로 잡은 값이니까 main size와 맞춰져 있어야 함
    src_points = np.float32([(80, 70), (0, 280), (640, 280), (560, 70)])
    dst_points = np.float32([(0, 0), (0, 480), (640, 480), (640, 0)])

    print("Vision Test Started with Picamera2. Press 'q' or ESC to exit.")

    while True:
        # Picamera2에서 RGB 프레임 받아오기 (main 스트림)
        frame_rgb = picam2.capture_array()

        # OpenCV는 BGR 기준이라 변환
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        height, width = frame.shape[:2]

        # 혹시라도 해상도가 640x480이 아닌 경우 대비 (좌표 깨지는 것 방지)
        # width, height가 다르면 비율 맞춰서 src/dst를 다시 잡거나 resize 해야 함
        # 지금은 640x480 기준이니까 그대로 사용
        # 디버그용 원본 + ROI 폴리곤 그리기
        debug_frame = frame.copy()
        pts = src_points.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(debug_frame, [pts], True, (0, 0, 255), 2)

        # 투시 변환 (Bird's Eye View)
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        bird_view = cv2.warpPerspective(frame, M, (width, height))

        # HSV 변환 후 검은색 라인 추출
        hsv = cv2.cvtColor(bird_view, cv2.COLOR_BGR2HSV)

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 80])
        mask = cv2.inRange(hsv, lower_black, upper_black)

        # 창 3개 출력
        cv2.imshow("1. Original with ROI (Picamera2)", debug_frame)
        cv2.imshow("2. Bird's Eye View", bird_view)
        cv2.imshow("3. Black Lane Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' 또는 ESC
            break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
