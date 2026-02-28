import cv2
import numpy as np

cap = cv2.VideoCapture("tennis_video.mp4")

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=True
)

# HSV RANGE

lower_hsv = np.array([25, 80, 80])
upper_hsv = np.array([45, 255, 255])


# KALMAN FILTER (4,2)
# state: [x,y,vx,vy]
# measurement: [x,y]

kalman = cv2.KalmanFilter(4, 2)

kalman.measurementMatrix = np.array([
    [1,0,0,0],
    [0,1,0,0]
], np.float32)

kalman.transitionMatrix = np.array([
    [1,0,1,0],
    [0,1,0,1],
    [0,0,1,0],
    [0,0,0,1]
], np.float32)

kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

lk_params = dict(
    winSize=(15,15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
)
prev_gray = None
prev_pts = None

# TRAJECTORY MEMORY

trajectory = []
MAX_TRAIL = 100

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800,450))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # MOTION MASK

    motion_mask = bg_subtractor.apply(frame)
    motion_mask = cv2.medianBlur(motion_mask,5)
    _, motion_mask = cv2.threshold(motion_mask,200,255,cv2.THRESH_BINARY)

    # HSV MASK

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    combined_mask = cv2.bitwise_and(motion_mask, hsv_mask)
    kernel = np.ones((5,5),np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    measurement = None

    # CONTOUR DETECTION

    contours,_ = cv2.findContours(
        combined_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > 80:
            x,y,w,h = cv2.boundingRect(largest)
            cx = x + w//2
            cy = y + h//2

            measurement = np.array([[np.float32(cx)],
                                    [np.float32(cy)]])

            kalman.correct(measurement)

            prev_pts = np.array([[[cx,cy]]], dtype=np.float32)

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


    # KALMAN PREDICTION

    prediction = kalman.predict()
    px, py = int(prediction[0]), int(prediction[1])

    # OPTICAL FLOW (fallback tracking)

    if prev_gray is not None and prev_pts is not None:
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_pts, None, **lk_params
        )

        if status[0][0] == 1:
            flow_x, flow_y = next_pts[0][0]
            prev_pts = next_pts

            px, py = int(flow_x), int(flow_y)

    # STORE TRAJECTORY
    trajectory.append((px,py))
    trajectory = trajectory[-MAX_TRAIL:]

    # DRAW TRAJECTORY
    for i in range(1,len(trajectory)):
        cv2.line(frame, trajectory[i-1], trajectory[i], (255,0,0), 2)

    cv2.circle(frame,(px,py),5,(0,0,255),-1)


    cv2.imshow("Tracking", frame)
    cv2.imshow("Mask", combined_mask)

    prev_gray = gray.copy()

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()