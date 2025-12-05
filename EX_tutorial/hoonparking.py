import time
from gpiozero import AngularServo, Motor

# ===========================
# 파라미터 (기존 코드 값 유지)
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

# ===========================
# Init 함수 (기존 형태 유지, 카메라 제외)
# ===========================
def Init():
    # 카메라는 주차 동작 테스트에 필요 없으므로 제외했습니다.
    # 서보 설정
    servo = AngularServo(
        SERVO_PIN,
        min_angle=SERVO_MIN_ANGLE,
        max_angle=SERVO_MAX_ANGLE
    )
    servo.angle = SERVO_CENTER_ANGLE

    # 모터 설정
    motor = Motor(
        forward=MOTOR_FORWARD_PIN,
        backward=MOTOR_BACKWARD_PIN,
        enable=MOTOR_ENABLE_PIN,
        pwm=True
    )
    motor.stop()

    return servo, motor

# ===========================
# 바로 실행될 후진 주차 함수
# ===========================
def parking_sequence(servo, motor):
    print("=== 후진 주차 모드 시작 ===")
    
    # 1. 정지 및 준비
    motor.stop()
    servo.angle = SERVO_CENTER_ANGLE
    time.sleep(0.5)

    # 2. 핸들 왼쪽 최대 꺾기 (스티어링)
    print("Steering: Left MAX (0도)")
    servo.angle = SERVO_MAX_ANGLE
    time.sleep(0.5) # 서보 반응 대기

    # 3. 후진 (1초)
    print("Reverse: 1.0 sec")
    motor.backward(BASE_SPEED)
    time.sleep(1.0) # [시간 조절] 후진 시간

    # 4. 정지 및 핸들 중앙 정렬
    motor.stop()
    print("Align Center")
    servo.angle = SERVO_CENTER_ANGLE
    time.sleep(0.5)

    # 5. 직진 주차 (주차칸 진입)
    print("Forward: Parking...")
    motor.backward(BASE_SPEED)
    time.sleep(1.0) # [시간 조절] 직진 시간

    # 6. 종료
    print("Parking Completed")
    motor.stop()

# ===========================
# main
# ===========================
def main():
    # 1. 초기화 (모터, 서보만)
    servo, motor = Init()

    try:
        # 2. 시작하자마자 주차 함수 실행
        parking_sequence(servo, motor)

    except KeyboardInterrupt:
        print("중단됨")

    finally:
        print("시스템 종료")
        motor.stop()
        motor.close()
        servo.close()

if __name__ == "__main__":
    main()