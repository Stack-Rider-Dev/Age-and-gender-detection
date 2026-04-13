import cv2
from deepface import DeepFace
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        results = DeepFace.analyze(
            img_path = frame, 
            actions = ['age', 'gender'], 
            enforce_detection=False
        )
        if results and len(results) > 0:
            result = results[0] 

            region = result['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            age = result['age']
            gender = result['dominant_gender']

            text = f"{gender}, {age}"

            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    except Exception as e:
        
        pass

    cv2.imshow('Age and Gender Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()