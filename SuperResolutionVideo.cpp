#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Uso: " << argv[0] << " <video_entrada> <video_salida> <cpu|gpu>" << std::endl;
        return 1;
    }

    std::string inputVideo = argv[1];
    std::string outputVideo = argv[2];
    std::string device = argv[3];

    // Cargar modelo NSSR-DIL (ajusta la ruta y nombre del archivo ONNX)
    cv::dnn::Net srNet = cv::dnn::readNet("model/NSSR-DIL.onnx");

    if (device == "gpu") {
        srNet.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        srNet.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else {
        srNet.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        srNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    cv::VideoCapture cap(inputVideo);
    if (!cap.isOpened()) {
        std::cerr << "No se pudo abrir el video de entrada." << std::endl;
        return 1;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    // Ajusta el factor de escalado según el modelo (ejemplo: x4)
    int upscale = 4;
    cv::VideoWriter writer(outputVideo, cv::VideoWriter::fourcc('M','J','P','G'), fps, cv::Size(width * upscale, height * upscale));

    cv::Mat frame;
    while (cap.read(frame)) {
        // Preprocesamiento
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0);
        srNet.setInput(blob);

        // Inferencia
        cv::Mat output = srNet.forward();

        // Postprocesamiento
        cv::Mat result;
        // El output puede tener forma NCHW, convertir a imagen
        cv::dnn::imagesFromBlob(output, result);
        result.convertTo(result, CV_8U, 255.0);

        writer.write(result);
        cv::imshow("Super Resolution", result);
        if (cv::waitKey(1) == 27) break; // ESC para salir
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    return 0;
}