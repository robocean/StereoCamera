import os
import cv2
import numpy as np
from liboCams import FindCamera, oCams
from roboflow import Roboflow
import supervision as sv
from threading import Thread

# 카메라 노드 설정 (기본적으로 /dev/video0 사용)
CAMERA_NODE = "/dev/video0"

# Roboflow API 초기화 및 모델 설정
rf = Roboflow(api_key="1diGVjfv3ibvixCo3QZl")
project = rf.workspace().project("global-solution")
model = project.version(4).model

# 카메라 탐색 및 초기화
devpath = FindCamera('oCamS-1CGN-U')
if devpath is None:
    print("oCam 장치를 찾을 수 없습니다.")
    exit()

# 카메라 장치 경로를 video0으로 설정
if devpath != CAMERA_NODE:
    devpath = CAMERA_NODE

camera = None
try:
    camera = oCams(devpath, verbose=1)
except Exception as e:
    print(f"카메라 초기화 중 오류 발생: {e}")
    exit()

# 카메라 포맷 설정 (적절한 포맷 선택)
try:
    format_list = camera.GetFormatList()
    print("지원 포맷 리스트:")
    for fmt in format_list:
        print(fmt)
    if format_list:
        camera.Set(format_list[0])
    camera.Start()
except Exception as e:
    print(f"카메라 설정 중 오류 발생: {e}")
    if camera:
        camera.Close()
    exit()

print("카메라가 시작되었습니다. 'q'를 눌러 종료하세요.")

# Supervision의 Bounding Box 및 Label Annotator 초기화
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# 프레임 카운터 및 예측 간격 설정
frame_count = 0
prediction_interval = 5  # 5 프레임마다 예측

# 저장된 탐지 결과 (초기값 None)
previous_detections = None

try:
    while True:
        # 실시간 프레임 가져오기
        frame = camera.GetFrame(mode=1)  # mode=1: 병합된 프레임 (스테레오)

        # Bayer 포맷을 RGB로 변환
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_GB2BGR)
        except cv2.error as e:
            print(f"프레임 변환 오류: {e}")
            continue

        # 프레임 해상도 축소 (640x480)
        frame_resized = cv2.resize(rgb_frame, (1280, 960))

        # 프레임 카운터 증가
        frame_count += 1

        # 매 n번째 프레임에 대해 추론 실행
        if frame_count % prediction_interval == 0:
            # 추론을 위한 RGB 형식으로 변환
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # 추론 실행
            results = model.predict(frame_rgb, confidence=30, overlap=30).json()

            # Supervision Detections API를 이용하여 결과 처리
            detections = sv.Detections.from_inference(results)
            previous_detections = detections  # 최신 탐지 결과 저장

        # 최신 탐지 결과를 이용해 프레임을 주석 처리
        if previous_detections:
            annotated_frame = bounding_box_annotator.annotate(
                scene=frame_resized, detections=previous_detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=previous_detections
            )
        else:
            # 탐지 결과가 없는 경우 원본 프레임 사용
            annotated_frame = frame_resized

        # 주석 처리된 프레임 디스플레이
        cv2.imshow("Webcam Object Detection", annotated_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("종료 요청됨.")

finally:
    # 카메라 스트리밍 중지 및 정리
    if camera:
        camera.Stop()
        camera.Close()
    cv2.destroyAllWindows()

