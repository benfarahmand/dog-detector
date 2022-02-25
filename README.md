# dog-detector
A dog has pooped in my front yard. We need to find out who did the deed.


Set up a webcam, analyze the video stream with pytorch and opencv with the Yolov5 model, then save a picture of the culprit.

Using a CUDA graphics card for the image analysis. You could possibly get away with using a CPU, but it'll be slow and the dog may get away before you snap a picture.

Feel free to use this code.

Credits:
- The python script was heavily inspired by https://github.com/akash-agni/Real-Time-Object-Detection
  - Note: Instead of using the torch version listed in the requirements.txt provided by akash-agni, I installed a CUDA version of pytorch: https://pytorch.org/get-started/locally/
- The model is from https://github.com/ultralytics/yolov5
