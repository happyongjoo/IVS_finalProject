from picamera2 import Picamera2
import cv2
import numpy as np
import time

def main():
    # ==========================================
    # 1. 카메라 0번 초기화
    # ==========================================
    print("Camera 0 초기화 중...")
    picam0 = Picamera2(camera_num=0)
    config0 = picam0.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam0.configure(config0)
    picam0.start()

    # ==========================================
    # 2. 카메라 1번 초기화
    # ==========================================
    print("Camera 1 초기화 중...")
    picam1 = Picamera2(camera_num=1)
    config1 = picam1.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam1.configure(config1)
    picam1.start()
    
    # 카메라 워밍업 대기
    time.sleep(1)

    try:
        while True:
            # --------------------------------------
            # 영상 캡처 (RGB -> BGR 변환 필요)
            # --------------------------------------
            frame0 = cv2.cvtColor(picam0.capture_array(), cv2.COLOR_RGB2BGR)
            frame1 = cv2.cvtColor(picam1.capture_array(), cv2.COLOR_RGB2BGR)

            # --------------------------------------
            # 화면 표시
            # --------------------------------------
            # 보기 좋게 두 화면을 가로로 붙이기 (Horizontal Stack)
            # 세로로 붙이려면 np.vstack([frame0, frame1]) 사용
            combined_view = np.hstack([frame0, frame1])
            
            # 텍스트 추가 (왼쪽: Cam 0, 오른쪽: Cam 1)
            cv2.putText(combined_view, "CAM 0", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined_view, "CAM 1", (670, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Dual Camera Viewer", combined_view)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"에러 발생: {e}")

    finally:
        # 프로그램 종료 시 자원 해제
        print("카메라 종료 중...")
        picam0.stop()
        picam1.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()