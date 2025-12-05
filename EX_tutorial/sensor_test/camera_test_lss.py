import cv2
import numpy as np
from picamera2 import Picamera2
import cv2
import numpy as np
import time
import math

from gpiozero import AngularServo, Motor, DistanceSensor

src_points = np.float32([(80, 70), (0, 280), (640, 280), (560, 70)])
dst_points = np.float32([(0, 0), (0, 480), (640, 480), (640, 0)])

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
    
    return picam2

class GreenLaneFollower:
    def __init__(self):
        self.window_width = 320
        self.window_height = 240
        
        # [설정] Perspective Transform 좌표 (환경에 맞게 수정 필수)
        # 바닥의 직사각형 영역을 설정
        self.src_points = src_points
        self.dst_points = dst_points
        
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

    def warp_image(self, img):
        h, w = img.shape[:2]
        return cv2.warpPerspective(img, self.M, (w, h))

    def color_filter(self, img):
        """ 초록색 추출 """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([18, 70, 40])
        upper_green = np.array([87, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        return mask

    def filter_vertical_mask(self, binary_mask):
        """ 
        [핵심 추가 기능] 
        이진화된 마스크에서 '수직(세로) 비율'이 높은 덩어리만 남기고 
        가로 선이나 잡음은 모두 검은색으로 지워버립니다.
        """
        # 윤곽선 검출
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 결과 마스크 (처음엔 다 검은색)
        vertical_mask = np.zeros_like(binary_mask)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50: continue # 너무 작은 노이즈 제거

            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0: continue
            
            aspect_ratio = float(h) / w  # 세로 / 가로 비율
            
            # 비율이 1.0보다 크면 (세로가 더 길면) 통과
            # 차선은 보통 세로로 아주 기니까 1.2~1.5 정도로 주면 가로선을 확실히 거릅니다.
            if aspect_ratio > 1.2:
                # 원본 마스크에서 해당 부분만 가져와서 vertical_mask에 그리기
                cv2.drawContours(vertical_mask, [cnt], -1, 255, -1)
                
        return vertical_mask

    def sliding_window_search(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        
        if np.sum(histogram) == 0:
            return np.dstack((binary_warped, binary_warped, binary_warped)) * 255, None

        base_x = np.argmax(histogram)
        nwindows = 9
        window_height = int(binary_warped.shape[0] / nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        current_x = base_x
        margin = 50
        minpix = 50
        
        lane_inds = []
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_x_low = current_x - margin
            win_x_high = current_x + margin
            
            cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)
            
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                         (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            
            lane_inds.append(good_inds)
            
            if len(good_inds) > minpix:
                current_x = int(np.mean(nonzerox[good_inds]))

        lane_inds = np.concatenate(lane_inds)
        
        fit = None
        if len(lane_inds) > 0:
            x_pixels = nonzerox[lane_inds]
            y_pixels = nonzeroy[lane_inds]
            fit = np.polyfit(y_pixels, x_pixels, 2)
            
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
            fit_x = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
            
            pts = np.array([np.transpose(np.vstack([fit_x, ploty]))])
            cv2.polylines(out_img, np.int_([pts]), False, (0, 255, 255), 3)

        return out_img, fit

    def process(self, frame):
        frame = cv2.resize(frame, (self.window_width, self.window_height))
        
        # 1. Bird's Eye View
        warped = self.warp_image(frame)
        
        # 2. Color Filter (초록색 전체)
        raw_mask = self.color_filter(warped)
        
        # 3. [NEW] Vertical Filter (수직 성분만 남김)
        vertical_mask = self.filter_vertical_mask(raw_mask)
        
        # 4. Sliding Window (수직 마스크 기반으로 탐색)
        sliding_result, fit = self.sliding_window_search(vertical_mask)
        
        steering_angle = 0
        if fit is not None:
            img_center = self.window_width / 2
            y_eval = self.window_height
            lane_center = fit[0]*y_eval**2 + fit[1]*y_eval + fit[2]
            steering_angle = lane_center - img_center

        return frame, warped, vertical_mask, sliding_result, steering_angle

# --- 실행부 ---
cap = cv2.VideoCapture(0)
follower = GreenLaneFollower()

picam2= Init()
while True:
    frame = picam2.capture_array()
    
    
    # process 함수가 이제 '수직 필터링된 마스크'를 리턴합니다.
    original, warped, vertical_mask, sliding, angle = follower.process(frame)
    
    # 시각화 합치기
    # vertical_mask는 흑백이므로 컬러로 변환
    mask_3ch = cv2.cvtColor(vertical_mask, cv2.COLOR_GRAY2BGR)
    
    # 화면 배치:
    # [원본] [탑뷰(Warped)]
    # [수직마스크] [결과창]
    top_row = np.hstack((original, warped))
    bottom_row = np.hstack((mask_3ch, sliding))
    combined_view = np.vstack((top_row, bottom_row))
    
    cv2.putText(combined_view, f"Steering Error: {angle:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Vertical Lane Detection", combined_view)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()