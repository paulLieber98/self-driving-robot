import cv2 as cv

import time

#https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector/python#live-stream_2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#for actual robot movement/control 
from gpiozero import Motor


#robot movement variables + functions + etc.
motor_left = Motor(forward=27, backward=17) #IN1 = FORWARD, IN2 = BACKWARD
motor_right = Motor(forward=22, backward=23) #IN3 = FORWARD, IN4 = BACKWARD

#functions for robot movement
def emergency_stop():
    print('EMERGENCY STOP: OBJECT RIGHT IN FRONT')
    #ADD ACTUAL STOP LOGIC FOR THE ROBOT HERE
    motor_left.stop()
    motor_right.stop()

def move_forward():
    motor_left.forward()
    motor_right.forward()

def move_backward():
    motor_left.backward()
    motor_right.backward()

def turn_left(): #ADD DEGREES OF ROTATION HERE LATER ON
    motor_left.backward()
    motor_right.forward()

def turn_right(): #ADD DEGREES OF ROTATION HERE LATER ON
    motor_left.forward()
    motor_right.backward()



model_path = 'efficientdet_lite0.tflite'

camera = cv.VideoCapture(1)

BaseOptions = mp.tasks.BaseOptions
DetectionResult = mp.tasks.components.containers.Detection
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode.LIVE_STREAM

latest_detections = None #global variable to store the latest detection results



#so the model detects things. what happens when it detects something? here is the function to do that
def print_result(result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
    global latest_detections
    latest_detections = result
    
    #DEBUG STATEMENTS + TO FEEL OUT RESULTS FROM THE MODEL
    # print(f'\n STARTS HERE: \n {result} \n ENDS HERE \n')
    
    # # Let's see what attributes this object actually has
    # print(f"Result type: {type(result)}")
    # print(f"Result attributes: {dir(result)}")
    
    # print(f'\n LATEST DETECTION LIST IN DETECTIONS OBJECT: \n {latest_detections.detections[0]} \n \n {latest_detections.detections[1]} \n \n {latest_detections.detections[2]} \n \n {latest_detections.detections[3]} \n \n {latest_detections.detections[4]}')

    # print(f'frame.shape: {frame.shape}')
    # print(f'\n starts here: \n {frame} \n ends here \n')

    # # Check if it has detections attribute
    # if hasattr(result, 'detections'):
    #     print(f"Number of detections: {len(result.detections)}")
    #     for i, detection in enumerate(result.detections):
    #         print(f"Detection {i}: {detection}")
    #         print(f"Detection attributes: {dir(detection)}")

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    max_results=5,
    result_callback=print_result)

detector = vision.ObjectDetector.create_from_options(options)

#MAIN LOOP HERE
with ObjectDetector.create_from_options(options) as detector:
    if not camera.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Our operations on the frame come here
        bgr_to_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_timestamp_ms = int(time.time() * 1000) # x1000 for timestamp in MILLISECONDS 
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=bgr_to_rgb) #.Image() is object function that the model expects to take in (next line, we use this object)
                                        #.imageFormat.SRGB = 'standard RGB'

        detected_result = detector.detect_async(mp_image, frame_timestamp_ms) #on docs. the results are sent to the callback function above

        # #invert image so looks normal
        # bgr_to_rgb_flipped = cv.flip(bgr_to_rgb, 1) #DROP FLIP FOR NOW
        #back to bgr for display so no blue-ish tint
        # final_frame = cv.cvtColor(bgr_to_rgb_flipped, cv.COLOR_RGB2BGR)
        final_frame = cv.cvtColor(bgr_to_rgb, cv.COLOR_RGB2BGR)


        ##VARIABLES HERE FOR MAIN LOOP FEATURES
        # for weighted risk calculation
        area_bounding_box = 0

        #REGIONS VARIABLES
        #define height, width, num_color_channels of frame
        #frame.shape is this: frame.shape: (1080, 1920, 3) from our print statement in callback function
        height, width, num_color_channels = final_frame.shape
        frame_in_thirds = width // 3 #flat division for int
        #calculate area of frame
        area_frame = width * height
        
        #for region CUTOFFS (x-coordinates)
        x1 = width // 3
        x2 = (width // 3) * 2
        #draw region cutoff lines (https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html)
        cv.line(final_frame, (x1, 0), (x1, height), (0, 0, 255), 2) #red line
        cv.line(final_frame, (x2, 0), (x2, height), (0, 0, 255), 2)

        #SPLITTING FRAME INTO 3 REGIONS: LEFT, CENTER, RIGHT.
        #These regions will be used to determine the 'risk' of each part of what the robot sees in order to decide which way to go.
        #will be measured in terms of 'how much stuff is in each region'. Then, robot will go to the region with the least risk or least amount of 'stuff'      
        #opencv slicing: region = frame[y_start:y_end, x_start:x_end]
        left_region = final_frame[0:height, 0:x1]
        center_region = final_frame[0:height, x1:x2]
        right_region = final_frame[0:height, x2:width]
        #displaying regions - SHOWING THE SINGLE WINDOW WITH BANDS SEPERATING REGIONS FOR NOW.
        # cv.imshow('left_region', left_region)
        # cv.imshow('center_region', center_region)
        # cv.imshow('right_region', right_region)

        #EMERGENCY STOP VARIABLES
        too_close_threshold = 0.5 #if bounding box of an object is more than 50% of the frame, it is too close
        conf_score_too_close = 0.70 #if confidence score is > 0.70, it is too close
        

        #IF DETECTIONS ARE FOUND, DO THE FOLLOWING:
        if latest_detections: 
            for detection in latest_detections.detections:
                #latest_detections.detections looks like this and is a LIST. this is like first index in list [0]:
                # Detection(bounding_box=BoundingBox(origin_x=346, origin_y=509, width=1534, height=566), categories=[Category(index=None, score=0.71875, display_name=None, category_name='person')], keypoints=[])
                #if detection has bounding box (hasattr = has attribute). look at the list above for reference what is going on.
                if hasattr(detection, 'bounding_box'): 
                    bounding_box = detection.bounding_box

                    #EMERGENCY STOP LOGIC
                    #logic: if object is more than 50% of the frame AND the confidence score is > 0.70, it is too close
                    if bounding_box.width / width > too_close_threshold and detection.categories[0].score > conf_score_too_close:
                        emergency_stop()

                    #DRAWING BOUNDING BOXES 
                    start_point_box = bounding_box.origin_x, bounding_box.origin_y
                    end_point_box = bounding_box.origin_x + bounding_box.width, bounding_box.origin_y + bounding_box.height
                    # Use the orange color for high visibility. #FROM GOOGLE GITHUB
                    cv.rectangle(final_frame, start_point_box, end_point_box, (0, 165, 255), 3)
                    #NOW ADDING LABELS TO THE BOUNDING BOXES
                    #from google github: FOR LABELS AROUND BOXES + CONFIDENCE SCORES
                    MARGIN = 10  # pixels
                    ROW_SIZE = 30  # pixels
                    FONT_SIZE = 1
                    FONT_THICKNESS = 1
                    TEXT_COLOR = (0, 0, 0)  # black
                    #FROM GOOGLE GITHUB: FOR LABELS AROUND BOXES + CONFIDENCE SCORES
                    category = detection.categories[0]
                    category_name = category.category_name
                    probability = round(category.score, 2)
                    result_text = f'{category_name} ( {str(probability)} )'
                    #FROM GOOGLE GITHUB: FOR LABELS AROUND BOXES + CONFIDENCE SCORES
                    text_location = (MARGIN + bounding_box.origin_x,
                                    MARGIN + ROW_SIZE + bounding_box.origin_y)
                    cv.putText(final_frame, result_text, text_location, cv.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

                else:
                    print('no detections found')
        

        #GETTING AREA OF FIRST DETECTION BOUNDING BOX
        area_bounding_box = 0
        if latest_detections:
            for detection in latest_detections.detections:
                if hasattr(detection, 'bounding_box'):
                    bounding_box = detection.bounding_box
                    area_bounding_box = bounding_box.width * bounding_box.height
                    break

        #GETTING AREA PROPORTION OF FIRST DETECTION BOUNDING BOX TO FULL FRAME
        area_proportion = 0
        if area_bounding_box:
            area_proportion = area_bounding_box / area_frame      

        #GETTING WEIGHTED VALUE USING THE LAST DETECTION'S SCORE
        weighted_risk_value = 0
        if latest_detections:
            for detection in latest_detections.detections:
                weighted_risk_value = round(area_proportion * detection.categories[0].score, 3)

        #ADD WEIGHTED RISK VALUE TO EACH BIN FOR EACH REGION
        risk_bins = [0, 0, 0] #LEFT, CENTER, RIGHT. every detection in whatever region will be added (+1) to the risk value for that region. at the end, taking path with min. risk
        if latest_detections:
            for detection in latest_detections.detections:
                if hasattr(detection, 'bounding_box'):
                    bounding_box = detection.bounding_box
                    bounding_box_center_x = bounding_box.origin_x + (bounding_box.width // 2)
                    if bounding_box_center_x < x1:  #left region
                        risk_bins[0] += weighted_risk_value
                    elif x1 <= bounding_box_center_x < x2:  #center region
                        risk_bins[1] += weighted_risk_value
                    elif bounding_box_center_x >= x2:  #right region
                        risk_bins[2] += weighted_risk_value
                    else:
                        print('detection not in any region')
                    
        #displaying risk bins
        print(f'risk bins: {risk_bins}')

        # Display the resulting frame
        cv.imshow('frame', final_frame)

        if cv.waitKey(1) == ord('q'):
            break
 
# When everything done, release the capture
camera.release()
cv.destroyAllWindows()




    