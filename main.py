import cv2
# 각 객체 감지를 위한 XML 파일의 경로
# 감지할 객체에 따라 적절한 파일을 선택해야 합니다.
haarcascades = ['C:/data/haarcascade_fullbody.xml',
                'C:/data/haarcascade_frontalface_default.xml']

# 각 객체 감지를 위한 CascadeClassifier 객체 생성
cascades = [cv2.CascadeClassifier(cascade) for cascade in haarcascades]


# 웹캠을 사용하여 영상을 캡처하기 위해 VideoCapture 객체를 생성합니다.
cap = cv2.VideoCapture(0)

while True:
    # 프레임 읽기
    ret, frame = cap.read()

    # 이미지를 흑백으로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 각 CascadeClassifier로 객체 감지 수행
    for cascade in cascades:
        objects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 감지된 객체 주위에 사각형 그리기
        for (x, y, w, h) in objects:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # 화면에 표시
    cv2.imshow('Object Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 사용한 자원 해제
cap.release()
cv2.destroyAllWindows()