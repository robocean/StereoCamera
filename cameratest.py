import os
import fcntl
import v4l2
from liboCams import FindCamera, oCams

# EnumerateFormats 함수로 지원되는 포맷 확인
def check_supported_formats(camera):
    # 카메라가 지원하는 포맷 리스트 가져오기
    format_list = camera.EnumerateFormats()
    print("Supported formats by the camera:")
    for fmt in format_list:
        print(f" - {fmt}")

# 현재 포맷을 가져와서 지원되는지 확인
def verify_current_format(camera):
    try:
        # 현재 포맷 가져오기
        form, stp = camera.GetCurrentFormat()

        # EnumerateFormats로부터 지원되는 포맷 리스트 가져오기
        format_list = camera.EnumerateFormats()

        # 현재 포맷과 일치하는 포맷이 있는지 확인
        is_supported = False
        for fmt in format_list:
            (_, width, height, _) = fmt  # 포맷 튜플에서 해상도와 기타 정보 추출
            if form.fmt.pix.width == width and form.fmt.pix.height == height:
                is_supported = True
                break

        if is_supported:
            print(f"Current format ({form.fmt.pix.width}x{form.fmt.pix.height}) is supported.")
        else:
            print(f"Current format ({form.fmt.pix.width}x{form.fmt.pix.height}) is NOT supported. Please update to a supported format.")

    except Exception as e:
        print(f"Failed to verify current format: {e}")

# 테스트 시작
def main():
    devpath = FindCamera('oCamS-1CGN-U')
    if devpath is None:
        print("oCam 장치를 찾을 수 없습니다.")
        return

    camera = oCams(devpath, verbose=1)

    # 지원되는 포맷 확인
    check_supported_formats(camera)

    # 현재 포맷 확인 및 검증
    verify_current_format(camera)

    # 카메라 종료
    camera.Close()

if __name__ == "__main__":
    main()
