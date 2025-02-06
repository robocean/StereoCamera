import numpy as np
import cv2
import glob

# 체스보드 패턴의 가로 세로 코너 개수 정의
chessboard_size = (8, 6)  # 체스보드 코너 수 (내부 코너 수)
square_size = 0.024  # 체스보드 정사각형 한 변의 길이 (미터 단위)

# 체스보드 3D 포인트 준비
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp = objp * square_size  # 실제 크기 반영

# 캘리브레이션 데이터를 저장할 배열 초기화
objpoints = []  # 3D 체스보드 포인트
imgpoints_left = []  # 왼쪽 카메라의 2D 이미지 포인트
imgpoints_right = []  # 오른쪽 카메라의 2D 이미지 포인트

# 체스보드 이미지가 저장된 경로 지정
images_left = glob.glob('Left Camera_screenshot_03.12.2024.png')
images_right = glob.glob('Right Camera_screenshot_03.12.2024.png')

# 왼쪽, 오른쪽 이미지가 동일한 개수여야 함
assert len(images_left) == len(images_right), "Left and right image counts do not match."

# 각 이미지에서 체스보드 코너 찾기
for img_left_path, img_right_path in zip(images_left, images_right):
    img_left = cv2.imread(img_left_path)
    img_right = cv2.imread(img_right_path)
    
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # 체스보드 코너 찾기
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

    if ret_left and ret_right:
        # 정확도를 높이기 위해 코너를 세밀하게 조정
        corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), 
                                        (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))
        corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), 
                                         (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))
        
        # 3D 포인트 추가
        objpoints.append(objp)
        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)

# 카메라 매트릭스 및 왜곡 계수 초기화
ret_left, mtx_left, dist_left, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
ret_right, mtx_right, dist_right, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

# 스테레오 캘리브레이션 수행
ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtx_left, dist_left, mtx_right, dist_right,
    gray_left.shape[::-1], None, None, None, None,
    cv2.CALIB_FIX_INTRINSIC,
    (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 1e-6)
)

# 보정된 파라미터를 이용해 사각형 영역 얻기
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(mtx_left, dist_left, mtx_right, dist_right,
                                            gray_left.shape[::-1], R, T, alpha=0)

print("스테레오 캘리브레이션 완료!")
print("R:\n", R)
print("T:\n", T)

# 보정 후 리매핑을 위한 매핑 계산
left_map1, left_map2 = cv2.initUndistortRectifyMap(mtx_left, dist_left, R1, P1, gray_left.shape[::-1], cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(mtx_right, dist_right, R2, P2, gray_right.shape[::-1], cv2.CV_16SC2)

# 이후의 영상에서 리매핑 적용을 위해 매핑 결과 사용
# rectified_left = cv2.remap(original_left_frame, left_map1, left_map2, cv2.INTER_LINEAR)
# rectified_right = cv2.remap(original_right_frame, right_map1, right_map2, cv2.INTER_LINEAR)

