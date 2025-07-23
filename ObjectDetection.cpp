#include <iostream>
#include <fstream>
#include "ObjectDetector.hpp"

int main()
{
    cv::dnn::Net neuralNetwork;
    std::vector<std::string> classes;
    std::string inputDirectory = "input";
    std::string outputDirectory = "output";

    try {
        neuralNetwork = cv::dnn::readNetFromTensorflow("/home/se8as23/Escritorio/Universidad/p66/VisionPorComputador/ClasificacionObjetos/MobileNet7/model/frozenInterfaceGraph.pb", 
            "/home/se8as23/Escritorio/Universidad/p66/VisionPorComputador/ClasificacionObjetos/MobileNet7/model/frozenInterfaceGraph.pbtxt");
        // Para GPU esar estas lineas:
        //neuralNetwork.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        //neuralNetwork.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    }
    catch (cv::Exception& error) {
        std::cout << error.msg << std::endl;
        return 1;
    }

    std::ifstream classesFile("model/classes.txt");
    if (!classesFile.is_open())
        return 1;

    std::string line;
    while (std::getline(classesFile, line))
        classes.push_back(line);

    ObjectDetector objectDetector(neuralNetwork, classes);
    objectDetector.setIODirectory(inputDirectory, outputDirectory);

    // Para imagen:
    // int result = objectDetector.detectObjects("test_image.jpg", ObjectDetector::SourceFileType::Image);

    // Para video:
    int result = objectDetector.detectObjects("test_video.mp4", ObjectDetector::SourceFileType::Video);

    return result;
}