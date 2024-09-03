import cv2
from ultralytics import YOLO


model = YOLO('yolov8n.pt')


img_path = 'istiklal-street.jpg'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


results = model(img_rgb)


for result in results:

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()
    names = result.names

    for box, score, class_id in zip(boxes, scores, class_ids):

        class_name = names[int(class_id)]
        print(f"Class: {class_name}, Score: {score:.2f}, Box Coordinates: {box}")

        x1, y1, x2, y2 = map(int, box)
        color = (255, 0, 0)
        thickness = 2

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        label = f"{class_name} {score:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)


cv2.imshow('YOLOv8 Detection', img)


cv2.waitKey(0)
cv2.destroyAllWindows()
