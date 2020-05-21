def gen():
	global counter, detected_objects
    while True:
        frame = tello.read()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fps = FPS().start()
            frame, new_detected_objects, counter = 
				YOLO.infere(frame, detected_objects, counter)
            if len(new_detected_objects) != 0:
                detected_objects = new_detectedobjects
            fps.update()
            fps.stop()
            , encodedImage = cv2.imencode('.jpg', frame)
            print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
            yield (b'--frame\r\n'
				    b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

 
@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')