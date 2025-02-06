import os
import cv2
import numpy as np
from liboCams import FindCamera, oCams

# 캘리브레이션 결과 행렬과 왜곡 계수 설정
CAMERA_MATRIX_LEFT = np.array([[923.2823355, 0, 632.2321406],
                          [0, 917.23688189, 529.50830473],
                          [0, 0, 1]])
DIST_COEFFS_LEFT = np.array([-0.48583604, 0.21310863, -0.00417402, 0.01102949, 0.09510026])

CAMERA_MATRIX_RIGHT = np.array([[932.38249915, 0, 715.81215355],[0, 927.59499542, 434.75853632],[0, 0, 1]])

DIST_COEFFS_RIGHT = np.array([-0.56310752, 0.55097282, 0.00144988, -0.01251764, -0.37058488])

# 카메라 노드 설정 (기본적으로 /dev/video0 사용)
CAMERA_NODE = "/dev/video0"

# 카메라 탐색 및 초기화
devpath = FindCamera('oCamS-1CGN-U')
if devpath is None:
    print("oCam 장치를 찾을 수 없습니다.")
    exit()

# 카메라 장치 경로를 video0으로 설정
if devpath != CAMERA_NODE:
    devpath = CAMERA_NODE

try:
    camera = oCams(devpath, verbose=1)
except Exception as e:
    print(f"카메라 초기화 중 오류 발생: {e}")
    exit()

# 카메라 포맷 설정 (적적한 포맷 선택)
try:
    format_list = camera.GetFormatList()
    print("지원 포맷 리스트:")
    for fmt in format_list:
        print(fmt)
    camera.Set(format_list[0])
    camera.Start()
except Exception as e:
    print(f"카메라 설정 중 오류 발생: {e}")
    camera.Close()
    exit()

print("카메라가 시작되었습니다. 'q'를 누르면 종료합니다.")

# 초승 경로 및 카메라 간 경로 
FOCAL_LENGTH = 4  # 4mm
BASELINE = 12  # 두 카메라 간 경로: 12cm

# 특정 색상 범위 (빨강색) 정의
LOWER_RED_1 = np.array([0, 120, 70])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 120, 70])
UPPER_RED_2 = np.array([180, 255, 255])

def detect_object(frame, lower_color, upper_color):
    """특정 색상의 물체를 감지"""
    # 색상 공간 변환 (BGR -> HSV)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_color[0], upper_color[0])
    mask2 = cv2.inRange(hsv, lower_color[1], upper_color[1])
    mask = mask1 + mask2

    # 노이즈 제거
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 윤곽선 탐지
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 가장 큰 윤곽선을 선택
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        center_x = x + w // 2
        center_y = y + h // 2
        return (center_x, center_y, w, h)
    return None

def calculate_distance(center_x_left, center_x_right):
    """위상 차이를 이용해 거리 계산"""
    # 시차 계산
    disparity = abs(center_x_left - center_x_right)
    if disparity <= 0:
        return None

    # 거리 계산 (Z축 거리)
    depth = (FOCAL_LENGTH * BASELINE) / disparity
    return depth

def calculate_angle(center_x, frame_width):
    """물체의 방향과 각도 계산"""
    return (center_x - frame_width // 2) * (60 / frame_width)  # 화면 중심 기준 각도

# 동영상 녹화 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_left = None
out_right = None

try:
    out_left = cv2.VideoWriter('left_camera_output.avi', fourcc, 20.0, (960, 540))
    out_right = cv2.VideoWriter('right_camera_output.avi', fourcc, 20.0, (960, 540))

    while True:
        # 실시간 프레임 가져오기
        frame = camera.GetFrame(mode=1)  # mode=1: 병합된 프레임 (스테레오)

        # Bayer 포맷을 RGB로 변환
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_GB2BGR)
        except cv2.error as e:
            print(f"프레임 변환 오류: {e}")
            continue

        # 프레임을 좌우로 분리
        height, width, _ = rgb_frame.shape
        mid = width // 2
        left_frame = rgb_frame[:, :mid, :]
        right_frame = rgb_frame[:, mid:, :]

        # 왜곡 보정 적용
        left_frame_undistorted = cv2.undistort(left_frame, CAMERA_MATRIX_LEFT, DIST_COEFFS_LEFT)
        right_frame_undistorted = cv2.undistort(right_frame, CAMERA_MATRIX_RIGHT, DIST_COEFFS_RIGHT)

        # 물체 탐지
        detected_left = detect_object(left_frame_undistorted, (LOWER_RED_1, LOWER_RED_2), (UPPER_RED_1, UPPER_RED_2))
        detected_right = detect_object(right_frame_undistorted, (LOWER_RED_1, LOWER_RED_2), (UPPER_RED_1, UPPER_RED_2))

        # 거리 및 각도 계산
        if detected_left and detected_right:
            center_x_left, _, _, _ = detected_left
            center_x_right, _, _, _ = detected_right
            distance = calculate_distance(center_x_left, center_x_right)
            angle = calculate_angle((center_x_left + center_x_right) / 2, left_frame.shape[1])
        else:
            distance, angle = None, None

        # 바운딩 박스 그리기 및 정보 표시 (왜곡 보정 적용된 프레임)
        if detected_left:
            x, y, w, h = detected_left
            cv2.rectangle(left_frame_undistorted, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(left_frame_undistorted, "Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if detected_right:
            x, y, w, h = detected_right
            cv2.rectangle(right_frame_undistorted, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(right_frame_undistorted, "Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if distance is not None and angle is not None:
            cv2.putText(left_frame_undistorted, f"Distance: {distance:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(left_frame_undistorted, f"Angle: {angle:.2f} degrees", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 화면 출력 (해상도 조정)
        resized_left = cv2.resize(left_frame_undistorted, (960, 540))
        resized_right = cv2.resize(right_frame_undistorted, (960, 540))

        # 녹화 저장
        if out_left is not None and out_right is not None:
            out_left.write(resized_left)
            out_right.write(resized_right)

        # 화면 출력
        cv2.imshow("Left Camera - Undistorted", resized_left)
        cv2.imshow("Right Camera - Undistorted", resized_right)

        # 'q' 키로 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("종료 요청됨.")
finally:
    # 녹화 파일 및 리소스 해제
    if out_left is not None:
        out_left.release()
    if out_right is not None:
        out_right.release()
    camera.Stop()
    camera.Close()
    cv2.destroyAllWindows()

