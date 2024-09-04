#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>

using namespace cv;

int main()
{
    // Open the video camera.
    std::string pipeline = "libcamerasrc"
                           " ! video/x-raw, width=800, height=600" // camera needs to capture at a higher resolution
                           " ! videoconvert"
                           " ! videoscale"
                           " ! video/x-raw, width=400, height=300" // can downsample the image after capturing
                           " ! videoflip method=rotate-180"        // remove this line if the image is upside-down
                           " ! appsink drop=true max_buffers=2";
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened())
    {
        printf("Could not open camera.\n");
        return 1;
    }

    // Create the OpenCV window
    namedWindow("Camera", cv::WINDOW_AUTOSIZE);
    // Create a control window
    namedWindow("Control", WINDOW_AUTOSIZE);
    // Create a ouput window
    namedWindow("Output", WINDOW_AUTOSIZE);
    Mat frame;

    // Measure the frame rate - initialise variables
    int frame_id = 0;
    timeval start, end;
    gettimeofday(&start, NULL);
    Mat hsv_frame;
    int iLowH = 0;
    int iHighH = 179;

    int iLowS = 0;
    int iHighS = 255;

    int iLowV = 0;
    int iHighV = 255;
    int morph = 2;

    // Create trackbars in "Control" window
    createTrackbar("LowH", "Control", &iLowH, 179); // Hue (0 - 179)
    createTrackbar("HighH", "Control", &iHighH, 179);

    createTrackbar("LowS", "Control", &iLowS, 255); // Saturation (0 - 255)
    createTrackbar("HighS", "Control", &iHighS, 255);

    createTrackbar("LowV", "Control", &iLowV, 255); // Value (0 - 255)
    createTrackbar("HighV", "Control", &iHighV, 255);

    createTrackbar("MorphSize", "Control", &morph, 10);
    
    // Moments moments(InputArray array, bool binaryImage);

    for (;;)
    {
        if (!cap.read(frame))
        {
            printf("Could not read a frame.\n");
            break;
        }

        // show frame
        imshow("Camera", frame);
        waitKey(1);

        cvtColor(frame, hsv_frame, COLOR_BGR2HSV);
        Mat thresh_img;
        inRange(hsv_frame, Scalar(iLowH, iLowS, iLowV),
                Scalar(iHighH, iHighS, iHighV), thresh_img);
        // Morphology
        morphologyEx(thresh_img,  // input
                     thresh_img,  // output
                     MORPH_CLOSE, // operation
                     getStructuringElement(MORPH_RECT, Size(morph, morph)));

        morphologyEx(thresh_img, // input
                     thresh_img, // output
                     MORPH_OPEN, // operation
                     getStructuringElement(MORPH_RECT, Size(morph, morph)));

        Moments m = moments(thresh_img, true);
        if (m.m00 > 0)
        {
            double x = m.m10 / m.m00;
            double y = m.m10 / m.m00;
            printf("Centre of mass is (%f,%f)\n", x, y);
        }
        imshow("Camera", frame);
        imshow("Output", thresh_img);
        waitKey(1);
        // Measure the frame rate
        frame_id++;
        if (frame_id >= 30)
        {
            gettimeofday(&end, NULL);
            double diff = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
            printf("30 frames in %f seconds = %f FPS\n", diff, 30 / diff);
            frame_id = 0;
            gettimeofday(&start, NULL);
        }
    }

    // Free the camera
    cap.release();
    return 0;
}
