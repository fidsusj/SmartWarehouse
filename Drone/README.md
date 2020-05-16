# Smart-Warehousing---Drone
Repo containing the code base for the drone used for smart warehousing

This is just the first version. In this version a very basic webpage is created to start the drone and the drone follows the path assigned in the code and captures an image and returns back to its place. The captured image is then processed to count the number of packages in it and this result along with the output from the Yolo model are displayed on the webpage.

Software req:
1) Flask
2) Keras
3) Tensorflow
4) Open cv
5) FFMPEG
6) Boost
7) numpy
8) tqdm

Steps to start the code:

1) Turn on the drone and wait till it starts to blink in multiple colors
2) Connect to Tello Wifi and disable firewall if it is enabled
3) Connect to the drone by running "python tello-start.py"
4) Start the flask server by running "python main.py"
5) Open the webpage "start-index.html" from templates folder and click on "fly" button

