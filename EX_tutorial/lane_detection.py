import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    src_points = np.float32([(220, 230), (70, 450), (570, 450), (420, 230)])
    dst_points = np.float32([(0, 0), (0, 480), (640, 480), (640, 0)])

    print("Vision Test Started. Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]

        debug_frame = frame.copy()
        pts = src_points.astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(debug_frame, [pts], True, (0, 0, 255), 2)

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        bird_view = cv2.warpPerspective(frame, M, (width, height))

        hsv = cv2.cvtColor(bird_view, cv2.COLOR_BGR2HSV)

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 80])

        mask = cv2.inRange(hsv, lower_black, upper_black)

        cv2.imshow("1. Original", debug_frame)
        cv2.imshow("2. Bird's Eye View", bird_view)
        cv2.imshow("3. Black Lane Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
