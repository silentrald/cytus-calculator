import copy
import cv2
import numpy as np

cap = cv2.VideoCapture('videos/chrome_vox.mp4')

if (cap.isOpened() == False): 
    print('Error opening video stream or file')

_, frame = cap.read()
width = int(frame.shape[1] / 4)
height = int(frame.shape[0] / 4)

font = cv2.FONT_HERSHEY_SIMPLEX
print(width, height)
blank_image = np.zeros(shape=[width, height, 3], dtype=np.uint8)

MAX_DIST = 22
MAX_AGE = 20

# skip to
cap.set(cv2.CAP_PROP_POS_FRAMES, 4500)

tap = 0
hold = 0
lhold = 0
drag = 0
flick = 0

# prev_frame
points = []
# 1 : G TAP, 2: B TAP, 3: P HOLD, 4: O HOLD, 5: LONG HOLD
label = np.zeros(shape=[height, width], dtype=np.uint8)
count = np.zeros(shape=[height, width], dtype=np.uint16)
age = np.zeros(shape=[height, width], dtype=np.uint8) # max 255

while(True):
    ret, frame = cap.read()
    # print(cap.get(cv2.CAP_PROP_POS_FRAMES))
  
    if ret:
        resized = cv2.resize(
            frame,
            (width, height)
        )
        rotated = cv2.rotate(resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
        hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

        green = cv2.inRange(hsv, (52, 45, 80), (93, 255, 255))

        kernel = np.ones((5, 5), np.uint8)
        green = cv2.erode(green, kernel, iterations = 1)
        green = cv2.dilate(green, kernel, iterations = 3)
        green = cv2.erode(green, kernel, iterations = 2)

        green_contours, h = cv2.findContours(
            green,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for n, cnt in enumerate(green_contours):
            arc_len = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.035 * arc_len, True)
            green = cv2.fillPoly(green, pts=[approx], color=(255, 255, 255))

            # if arc_len > 120 and arc_len < 220:
            #     if len(approx) == 4:
            #         m = cv2.moments(approx)
            #         cx = int(m['m10'] / m['m00'])
            #         cy = int(m['m01'] / m['m00'])

            #         # Check if the point is on the points array
            #         insert = True
            #         for pt in points:
            #             # get the distance of the point to every point here
            #             if abs(pt[0] - cx) + abs(pt[1] - cy) < MAX_DIST:
            #                 age[pt[0], pt[1]] = 0
            #                 insert = False
            #                 break
                    
            #         if insert:
            #             flick += 1
            #             points.append([ cx, cy ])
            #             label[cx, cy] = 4
            #             count[cx, cy] = flick
        
        blue = cv2.inRange(hsv, (94, 45, 125), (110, 255, 255))

        blue = cv2.dilate(blue, kernel, iterations = 1)
        blue = cv2.erode(blue, kernel, iterations = 2)
        blue = cv2.dilate(blue, kernel, iterations = 1)
        # blue = cv2.erode(blue, kernel, iterations = 1)

        blue_contours, h = cv2.findContours(
            blue,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for n, cnt in enumerate(blue_contours):
            arc_len = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.035 * arc_len, True)
            blue = cv2.fillPoly(blue, pts=[approx], color=(255, 255, 255))

        orange = cv2.inRange(hsv, (177, 47, 125), (183, 255, 255))

        # orange = cv2.erode(orange, kernel, iterations = 1)
        # orange = cv2.dilate(orange, kernel, iterations = 2)
        # orange = cv2.erode(orange, kernel, iterations = 1)

        orange_contours, h = cv2.findContours(
            orange,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for n, cnt in enumerate(orange_contours):
            arc_len = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * arc_len, True)
            orange = cv2.fillPoly(orange, pts=[approx], color=(255, 255, 255))
        
        pink = cv2.inRange(hsv, (148, 50, 155), (163, 255, 255))

        pink_contours, h = cv2.findContours(
            pink,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for n, cnt in enumerate(pink_contours):
            arc_len = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * arc_len, True)
            pink = cv2.fillPoly(pink, pts=[approx], color=(255, 255, 255))
        
        yellow = cv2.inRange(hsv, (10, 32, 147), (36, 255, 255))

        yellow_contours, h = cv2.findContours(
            yellow,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for n, cnt in enumerate(yellow_contours):
            arc_len = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * arc_len, True)
            yellow = cv2.fillPoly(yellow, pts=[approx], color=(255, 255, 255))

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.25, 22, minRadius=8, maxRadius=40)
        if circles is not None:
            circles = np.round(circles[0, :]).astype('int')
            
            for x, y, r in circles:
                rotated = cv2.circle(rotated, (x, y), r, (0, 0, 0), 4)

                insert = True
                for pt in points:
                    # get the distance of the point to every point here
                    if abs(pt[0] - x) + abs(pt[1] - y) < MAX_DIST:
                        age[pt[0], pt[1]] = 0
                        insert = False
                        break
                
                if insert:
                    # check color here
                    if green[ y, x ] == 255:
                        tap += 1
                        points.append([ x, y ])
                        label[ x, y ] = 1
                        count[ x, y ] = tap
                    elif blue[ y, x ] == 255:
                        tap += 1
                        points.append([ x, y ])
                        label[ x, y ] = 2
                        count[ x, y ] = tap
                    elif pink[ y, x ] == 255:
                        hold += 1
                        points.append([ x, y ])
                        label[ x, y ] = 3
                        count[ x, y ] = hold
                    elif orange[ y, x ] == 255:
                        hold += 1
                        points.append([ x, y ])
                        label[ x, y ] = 4
                        count[ x, y ] = hold
                    elif yellow[ y, x ] == 255:
                        lhold += 1
                        points.append([ x, y ])
                        label[ x, y ] = 5
                        count[ x, y ] = lhold

        # update the age of the point in points
        for pt in points:
            x = pt[0]
            y = pt[1]

            age[x, y] += 1

            if age[x, y] > MAX_AGE:
                age[x, y] = 0
                points.remove([x, y])
            else:
                rotated = cv2.circle(rotated, (x, y), radius=4, color=(255, 255, 255), thickness=-1)
                temp = label[x][y]
                txt = ''
                color = None

                if temp == 1:
                    txt = 'TAP '
                    color = (0, 255, 0)
                elif temp == 2:
                    txt = 'TAP '
                    color = (255, 0, 0)
                elif temp == 3:
                    txt = 'HOLD '
                    color = (255, 204, 204)
                elif temp == 4:
                    txt = 'HOLD '
                    color = (0, 127, 255)
                elif temp == 5:
                    txt = 'LHOLD '
                    color = (0, 255, 255)
                
                txt += str(count[x, y])
                rotated = cv2.putText(rotated, txt, (x, y), font, 0.75, color, 2)

        cv2.imshow('Original', rotated)
        cv2.imshow('Green', green)
        cv2.imshow('Blue', blue)
        cv2.imshow('Pink', pink)
        cv2.imshow('Orange', orange)
        cv2.imshow('Yellow', yellow)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()