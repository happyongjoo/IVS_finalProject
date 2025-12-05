from gpiozero import AngularServo, Motor
from time import sleep
import curses

# 서보 각도 설정 (필요하면 직접 조정해서 맞춰봐)
SERVO_MIN = 0      # 가장 왼쪽
SERVO_MAX = 180    # 가장 오른쪽
SERVO_CENTER = 90  # 정면
ANGLE_STEP = 5     # 방향키 눌렀을 때 한 번에 움직이는 각도 (5도씩)
MAX_SPEED = 0.5    # 0.0 ~ 1.0 (50% 속도)

def main(stdscr):
    # curses 초기 설정
    curses.cbreak()
    stdscr.nodelay(True)     # 키 입력 없을 때 getch가 기다리지 않도록
    stdscr.keypad(True)      # 방향키 인식

    # GPIO 초기화 (핀 번호는 너 하드웨어에 맞게 수정해)
    servo = AngularServo(18, min_angle=SERVO_MIN, max_angle=SERVO_MAX)
    motor = Motor(forward=14, backward=15, enable=23, pwm=True)

    # 초기 값
    current_angle = SERVO_CENTER
    servo.angle = current_angle
    motor.stop()

    stdscr.addstr(0, 0, "W: 전진, A: 좌회전 전진, D: 우회전 전진")
    stdscr.addstr(1, 0, "S: 정지, X: 후진")
    stdscr.addstr(2, 0, "위/아래 방향키: 조향 각도 미세 조정")
    stdscr.addstr(3, 0, "Q: 종료")
    stdscr.refresh()

    try:
        while True:
            key = stdscr.getch()

            if key in (ord('q'), ord('Q')):
                motor.stop()
                break

            # --------- 주행 제어 ---------
            if key in (ord('w'), ord('W')):
                current_angle = SERVO_CENTER
                servo.angle = current_angle
                motor.forward(MAX_SPEED)
                stdscr.addstr(5, 0, "전진               ")

            elif key in (ord('s'), ord('S')):
                # 정지
                motor.stop()
                stdscr.addstr(5, 0, "정지               ")

            elif key in (ord('x'), ord('X')):
                # 후진
                current_angle = SERVO_CENTER
                servo.angle = current_angle
                motor.backward(MAX_SPEED)
                stdscr.addstr(5, 0, "후진               ")

            elif key in (ord('a'), ord('A')):
                current_angle = SERVO_MIN
                servo.angle = current_angle
                #motor.forward(MAX_SPEED)
                stdscr.addstr(5, 0, "왼쪽 전진          ")

            elif key in (ord('d'), ord('D')):
                current_angle = SERVO_MAX
                servo.angle = current_angle
                #motor.forward(MAX_SPEED)
                stdscr.addstr(5, 0, "오른쪽 전진        ")

            # --------- 조향 각도 제어 ---------
            elif key == curses.KEY_UP:
                current_angle = min(current_angle + ANGLE_STEP, SERVO_MAX)
                servo.angle = current_angle
                stdscr.addstr(6, 0, f"각도: {current_angle:3d} (→쪽)    ")

            elif key == curses.KEY_DOWN:
                current_angle = max(current_angle - ANGLE_STEP, SERVO_MIN)
                servo.angle = current_angle
                stdscr.addstr(6, 0, f"각도: {current_angle:3d} (←쪽)    ")

            stdscr.refresh()
            sleep(0.05)

    finally:
        motor.stop()
        servo.angle = SERVO_CENTER

if __name__ == "__main__":
    curses.wrapper(main)
