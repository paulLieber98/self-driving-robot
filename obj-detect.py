import cv2 as cv

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

def print_result(result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
    print('detection result: {}'.format(result))

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    max_results=5,
    result_callback=print_result)

with ObjectDetector.create_from_options(options) as detector:
    #MAIN CAMERA LOOP
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
        #PASS

        #invert image so looks normal
        bgr_to_rgb_flipped = cv.flip(bgr_to_rgb, 1)
        #back to bgr for display so no blue-ish tint
        final_frame = cv.cvtColor(bgr_to_rgb_flipped, cv.COLOR_RGB2BGR)
        # Display the resulting frame
        cv.imshow('frame', final_frame)

        if cv.waitKey(1) == ord('q'):
            break
 
# When everything done, release the capture
camera.release()
cv.destroyAllWindows()




    