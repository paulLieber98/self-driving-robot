import cv2 as cv

import time

#https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector/python#live-stream_2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


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

    # bounding_box = result.bounding_box
    # obj_label = result.categories[0].category_name
    # conf_score = result.categories[0].score

    # for detection in result.detections:
        # bounding_box = result.bounding_box
        # start_point_box = bounding_box.origin_x, bounding_box.origin_y
        # end_point_box = bounding_box.origin_x + bounding_box.width, bounding_box.origin_y + bounding_box.height
        # # Use the orange color for high visibility.
        # cv.rectangle(output_image, start_point_box, end_point_box, (0, 165, 255), 3)

    global latest_detections
    latest_detections = result
    
    print(f'\n STARTS HERE: \n {result} \n ENDS HERE \n')
    
    # Let's see what attributes this object actually has
    print(f"Result type: {type(result)}")
    print(f"Result attributes: {dir(result)}")
    
    # Check if it has detections attribute
    if hasattr(result, 'detections'):
        print(f"Number of detections: {len(result.detections)}")
        for i, detection in enumerate(result.detections):
            print(f"Detection {i}: {detection}")
            print(f"Detection attributes: {dir(detection)}")

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
        #do whatever you need to do here with the RBG format image
        frame_timestamp_ms = int(time.time() * 1000) # x1000 for timestamp in MILLISECONDS 
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=bgr_to_rgb) #RESEARCH WHAT THIS DOES
        detected_result = detector.detect_async(mp_image, frame_timestamp_ms) #on docs. the results are sent to the callback function above


        #invert image so looks normal
        bgr_to_rgb_flipped = cv.flip(bgr_to_rgb, 1)
        #back to bgr for display so no blue-ish tint
        final_frame = cv.cvtColor(bgr_to_rgb_flipped, cv.COLOR_RGB2BGR)

        if latest_detections: #if detections are found
            for detection in latest_detections.detections:

                #if detection has bounding box (hasattr = has attribute)
                if hasattr(detection, 'bounding_box'): 
                    bounding_box = detection.bounding_box
                    start_point_box = bounding_box.origin_x, bounding_box.origin_y
                    end_point_box = bounding_box.origin_x + bounding_box.width, bounding_box.origin_y + bounding_box.height
                    # Use the orange color for high visibility. #FROM GOOGLE GITHUB
                    cv.rectangle(final_frame, start_point_box, end_point_box, (0, 165, 255), 3)

                else:
                    print('no detections found')

        if detection.categories: #adding labels to the bounding boxes
            pass


        # Display the resulting frame
        cv.imshow('frame', final_frame)

        if cv.waitKey(1) == ord('q'):
            break
 
# When everything done, release the capture
camera.release()
cv.destroyAllWindows()




    