import cv2
import numpy as np
import math

# =========================
# 서보 / 제어 파라미터
# =========================
SERVO_PIN = 18
SERVO_MIN_ANGLE = 0
SERVO_MAX_ANGLE = 180
SERVO_CENTER_ANGLE = 90

K_ANGLE = 3.0                    # 선형 매핑 게인

MAX_LANE_ANGLE = 60.0            # fitLine 클램핑 기준
HARD_TURN_THRESHOLD = 45.0       # ±45도 넘으면 최대 조향


class GreenLaneFollower:
    def __init__(self):
        self.window_width = 320
        self.window_height = 240
        
        # [설정] Perspective Transform 좌표
        self.src_points = np.float32([(40, 50), (0, 280), (640, 280), (600, 50)])
        self.dst_points = np.float32([(0, 0), (0, 480), (640, 480), (640, 0)])
        
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

    def warp_image(self, img):
        h, w = img.shape[:2]
        return cv2.warpPerspective(img, self.M, (w, h))

    def color_filter(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 튜닝하신 HSV 값 적용
        lower_green = np.array([40, 200, 23])
        upper_green = np.array([97, 255, 71])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        return mask

    def filter_vertical_mask(self, binary_mask):
        """ 
        형태학적 연산(Opening)을 이용한 수직선 분리
        """
        # (width, height) -> (4, 40)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 40))
        vertical_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, vertical_kernel)
        return vertical_mask

    # [수정 1] self 인자 추가 (클래스 메서드 문법)
    def fit_lane_angle_deg(self, side_img):
        points = cv2.findNonZero(side_img)
        if points is None or len(points) < 50:
            return None

        line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = line.flatten()

        # 위쪽이 y=0이므로 벡터 방향 보정
        if vy > 0: # vy가 양수면 아래를 향함 -> 위를 향하도록 반전 (좌표계 주의)
             vx, vy = -vx, -vy
        # 일반적으로 이미지 좌표계에서 위로 가는 벡터를 원한다면 vy는 음수여야 함.
        # 기존 로직: if vy < 0: vx, vy = -vx, -vy (이 부분은 상황에 따라 다를 수 있으나 일반적으론 위쪽 방향 벡터를 구함)
        # 여기서는 단순히 각도 계산용이므로 atan2만 잘 쓰면 됨.
        
        angle_rad = math.atan2(vx, -vy) # 수직선(y축) 기준 각도
        angle_deg = math.degrees(angle_rad)

        angle_deg = max(-MAX_LANE_ANGLE, min(MAX_LANE_ANGLE, angle_deg))
        return angle_deg
    
    def get_lane_angle_split(self, mask, roi_ratio):
        h, w = mask.shape

        # 동적으로 변하는 비율 적용
        roi_top = int(h * roi_ratio)
        roi = mask[roi_top:h, :]

        mid = w // 2
        left_roi = roi[:, :mid]
        right_roi = roi[:, mid:]

        angle_left = self.fit_lane_angle_deg(left_roi)
        angle_right = self.fit_lane_angle_deg(right_roi)

        # [수정 2] 반환 값의 개수와 형식을 통일 (항상 3개 반환)
        if angle_left is None and angle_right is None:
            return None, "none", roi_top 

        if angle_left is not None and angle_right is None:
            return angle_left, "left", roi_top

        if angle_left is None and angle_right is not None:
            return angle_right, "right", roi_top

        return (angle_left + angle_right) / 2.0, "both", roi_top    

    def process(self, frame):
        frame = cv2.resize(frame, (self.window_width, self.window_height))
        warped = self.warp_image(frame)
        raw_mask = self.color_filter(warped)
        
        vertical_mask = self.filter_vertical_mask(raw_mask)
        
        # [수정 3] ROI Ratio 변수 정의 (코드에 누락되어 있었음)
        current_roi_ratio = 0.25
        sliding_result = self.get_lane_angle_split(vertical_mask, current_roi_ratio)
        
        lane_angle_deg, side, roi_top = sliding_result

        # 조향 로직 계산
        steering_angle = SERVO_CENTER_ANGLE # 기본값
        
        if lane_angle_deg is not None:
            if lane_angle_deg <= -HARD_TURN_THRESHOLD:
                steering_angle = SERVO_MIN_ANGLE
            elif lane_angle_deg >= HARD_TURN_THRESHOLD:
                steering_angle = SERVO_MAX_ANGLE
            else:
                steering_angle = SERVO_CENTER_ANGLE - (K_ANGLE * lane_angle_deg) # 부호 주의 (왼쪽 기울어짐 -> 오른쪽 조향 필요하면 -)
            
            steering_angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, steering_angle))



        # [수정 4] 디버그용 이미지 생성 (np.hstack 에러 방지)
        # sliding_result는 튜플이므로 이미지로 시각화해야 함
        debug_img = cv2.cvtColor(vertical_mask, cv2.COLOR_GRAY2BGR)
        
        # ROI 영역 표시 (녹색 박스)
        cv2.rectangle(debug_img, (0, roi_top), (self.window_width, self.window_height), (0, 255, 0), 2)
        
        # 감지된 각도 텍스트 표시
        info_text = f"Side: {side}"
        if lane_angle_deg is not None:
            info_text += f" Ang: {lane_angle_deg:.1f}"
        cv2.putText(debug_img, info_text, (10, roi_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return frame, warped, vertical_mask, debug_img, steering_angle

# --- 실행부 ---
cap = cv2.VideoCapture(0)
follower = GreenLaneFollower()

while True:
    ret, frame = cap.read()
    if not ret: 
        print("Camera read failed")
        break
    
    # process 함수에서 모든 처리를 완료하고 리턴받음
    original, warped, vertical_mask, debug_img, steering_angle = follower.process(frame)
    
    # [수정 5] 시각화 로직 수정
    # mask_3ch는 vertical_mask를 BGR로 변환한 것
    mask_3ch = cv2.cvtColor(vertical_mask, cv2.COLOR_GRAY2BGR)
    
    # 상단: 원본 + 워핑된 이미지
    top_row = np.hstack((original, warped))
    
    # 하단: 마스크 + 디버그 정보(ROI 박스 등)
    # 기존 코드에서 sliding(튜플)을 hstack 하려다 에러가 났던 부분 수정
    bottom_row = np.hstack((mask_3ch, debug_img))
    
    combined_view = np.vstack((top_row, bottom_row))
    
    cv2.putText(combined_view, f"Steering: {steering_angle:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Green Lane Follower", combined_view)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()