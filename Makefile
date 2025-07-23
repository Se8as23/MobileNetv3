all:
	g++ ObjectDetector.cpp ObjectDetection.cpp --std=c++17 \
	-I/usr/local/include/opencv4/ \
	-L/usr/local/lib/ \
	-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_video -lopencv_videoio -lopencv_dnn \
	-o vision.bin -Wl,-rpath,/usr/local/lib


saludo:
	echo "Hola C++"

run:
	./vision.bin