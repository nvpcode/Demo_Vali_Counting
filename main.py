import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("video_test.mp4")
assert cap.isOpened(), "Error reading video file"

# region_points = [(20, 400), (1080, 400)]   # line counting                                  
region_points = [[1515, 1014], [1478, 1813]]  

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("vali_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize object counter object
counter = solutions.ObjectCounter(
    show=False,  # display the output
    region=region_points,  # pass region points
    model="yolo11n.pt",  # model
    show_labels= False,
    show_conf= True,
    line_width= 5, 
    device=0,
    classes=[28],
    tracker="botsort.yaml"
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break

    # Nhận diện + đếm
    results = counter(im0)

    # Hiển thị kết quả tổng IN/OUT
    y_offset = 200
    font_scale = 1.2
    font_thickness = 2
    margin_right = 95

    line_y = y_offset + 10
    cv2.line(im0, (im0.shape[1] - 600, line_y),
             (im0.shape[1] - margin_right + 15, line_y), (0, 0, 0), 2)

    total_y = line_y + 50
    total_text_1 = f"TOTAL VALI IN: {results.in_count}"
    total_text_2 = f"TOTAL VALI OUT: {results.out_count}"

    (w1, h1), _ = cv2.getTextSize(
        total_text_1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    (w2, h2), _ = cv2.getTextSize(
        total_text_2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    total_width = max(w1, w2)

    x_total = im0.shape[1] - total_width - margin_right
    cv2.rectangle(im0, (x_total - 10, total_y - h1 - 15),
                  (x_total + total_width + 20, total_y + h2 + 15),
                  (255, 255, 255), -1)

    cv2.putText(im0, total_text_1, (x_total, total_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 128, 0),
                font_thickness, cv2.LINE_AA)
    cv2.putText(im0, total_text_2, (x_total, total_y + h2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255),
                font_thickness, cv2.LINE_AA)

    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()