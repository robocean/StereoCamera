import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class StereoCameraNode(Node):
    def __init__(self):
        super().__init__('stereo_camera_node')
        self.publisher_left = self.create_publisher(Image, '/camera/left/image_raw', 10)
        self.publisher_right = self.create_publisher(Image, '/camera/right/image_raw', 10)
        self.bridge = CvBridge()

        # Open the stereo camera device
        self.camera = cv2.VideoCapture('/dev/video3')

        # Set camera properties if necessary
        # For example: self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)  # Width for both images
        # self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Height for both images

        # Timer to publish frames
        self.timer = self.create_timer(0.1, self.publish_frames)  # 10 Hz

    def publish_frames(self):
        ret, frame = self.camera.read()
        if not ret:
            self.get_logger().error('Failed to capture frame')
            return

        # Assume the frame contains both left and right images side by side
        height, width, _ = frame.shape
        left_image = frame[:, :width // 2, :]   # Left half of the frame
        right_image = frame[:, width // 2:, :] # Right half of the frame

        # Convert to ROS messages
        left_msg = self.bridge.cv2_to_imgmsg(left_image, encoding='bgr8')
        right_msg = self.bridge.cv2_to_imgmsg(right_image, encoding='bgr8')

        # Publish the images
        self.publisher_left.publish(left_msg)
        self.publisher_right.publish(right_msg)

    def destroy_node(self):
        self.camera.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = StereoCameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

