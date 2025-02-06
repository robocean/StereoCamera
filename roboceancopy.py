from roboflow import Roboflow
import cv2
import supervision as sv
from threading import Thread

class VideoStream:
    def __init__(self, src="/dev/video0"):  # oCam Viewer의 출력 디바이스 지정
        self.cap = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.cap.read()

    def read(self):
        return self.grabbed, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# Initialize Roboflow with your API key
rf = Roboflow(api_key="1diGVjfv3ibvixCo3QZl")

# Load the specific project and model
project = rf.workspace().project("global-solution")
model = project.version(4).model

# Initialize camera stream
video_stream = VideoStream().start()

# Define bounding box and label annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Frame counter for controlling prediction interval
frame_count = 0
prediction_interval = 5  # 5 프레임마다 예측

# 저장된 탐지 결과 (초기값 None)
previous_detections = None

while True:
    # Capture frame-by-frame from oCam Viewer
    ret, frame = video_stream.read()
    if not ret:
        break  # oCam Viewer의 출력을 읽을 수 없는 경우 루프 종료

    # Adjust resolution (e.g., resize to 1280x960)
    frame = cv2.resize(frame, (1280, 960))

    # Increment frame counter
    frame_count += 1

    # Run inference only on every nth frame
    if frame_count % prediction_interval == 0:
        # Convert frame to RGB format for inference
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference on the frame
        results = model.predict(frame_rgb, confidence=30, overlap=30).json()

        # Process the results with supervision Detections API
        detections = sv.Detections.from_inference(results)
        previous_detections = detections  # 최신 탐지 결과 저장

    # 최신 탐지 결과를 이용해 프레임을 주석 처리
    if previous_detections:
        annotated_frame = bounding_box_annotator.annotate(
            scene=frame, detections=previous_detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=previous_detections
        )
    else:
        # 탐지 결과가 없는 경우 원본 프레임 사용
        annotated_frame = frame

    # Display the annotated frame
    cv2.imshow("oCam Object Detection", annotated_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
video_stream.stop()
cv2.destroyAllWindows()
