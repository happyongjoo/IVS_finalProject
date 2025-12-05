import time
import numpy as np
import cv2
from picamera2 import Picamera2

# ==========================================
# [설정값]
# ==========================================
CROSSWALK_THRESHOLD = 17000  # 빨간 점 개수 기준값

# ROI (관심 영역) 좌표
ROI_Y_LOW = 200
ROI_Y_HIGH = 400
ROI_X_LOW = 0
ROI_X_HIGH = 640

def main():
    # ------------------------------------------------
    # 1. 카메라 초기화 (Picamera2)
    # ------------------------------------------------
    picam2 = Picamera2()
    sensor_w, sensor_h = picam2.sensor_resolution
    
    # 광각 모드 설정
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        raw={"size": (sensor_w, sensor_h)}
    )
    picam2.configure(config)
    picam2.set_controls({"ScalerCrop": (0, 0, sensor_w, sensor_h)})
    picam2.start()
    
    time.sleep(1) # 워밍업
    print("🎥 카메라 테스트 시작! (빨간색 물체를 비춰보세요)")

    try:
        while True:
            # ------------------------------------------------
            # 2. 이미지 캡처 & 전처리
            # ------------------------------------------------
            frame = picam2.capture_array()

            # ROI 자르기
            roi = frame[ROI_Y_LOW:ROI_Y_HIGH, ROI_X_LOW:ROI_X_HIGH]

            # HSV 변환
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # ------------------------------------------------
            # 3. 빨간색 마스크 생성 (0~10도, 170~180도)
            # ------------------------------------------------
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            
            lower_red2 = np.array([170, 100, 100])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.addWeighted(mask1, 1.0, mask2, 1.0, 0.0)

            # ------------------------------------------------
            # 4. 픽셀 수 카운트 & 시각화
            # ------------------------------------------------
            red_pixel_count = cv2.countNonZero(red_mask)
            

            # 화면에 픽셀 수 글씨 쓰기
            cv2.putText(
                frame,
                f"Red Pixels: {red_pixel_count}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            
            # ROI 박스 그리기
            cv2.rectangle(
                frame,
                (ROI_X_LOW, ROI_Y_LOW),
                (ROI_X_HIGH, ROI_Y_HIGH),
                (0, 0, 255),
                2,
            )

            # ------------------------------------------------
            # 5. 판단 로직 (모터 대신 글씨로 상태 알려줌)
            # ------------------------------------------------
            if red_pixel_count > CROSSWALK_THRESHOLD:
                # 횡단보도 인식됨!
                status_text = "STOP! (Crosswalk)"
                text_color = (0, 0, 255) # 빨간 글씨
                print(f"🚨 횡단보도 감지됨! ({red_pixel_count})")
            else:
                # 횡단보도 아님 (주행 상태)
                status_text = "GO (Straight)"
                text_color = (0, 255, 0) # 초록 글씨
                print(f"⬆️ 직진 구간 ({red_pixel_count})")

            # 상태를 화면 중앙에 크게 띄우기
            cv2.putText(frame, status_text, (200, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 3)

            # ------------------------------------------------
            # 6. 화면 출력
            # ------------------------------------------------
            cv2.imshow("Main View", frame)
            #cv2.imshow("Red Mask View", red_mask) # 빨간색이 잘 잡히는지 확인용

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("테스트 종료")

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()