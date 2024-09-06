#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

int main()
{
    // Open the video camera.
    std::string pipeline = "libcamerasrc"
                           " ! video/x-raw, width=800, height=600"
                           " ! videoconvert"
                           " ! videoscale"
                           " ! video/x-raw, width=400, height=300"
                           " ! videoflip method=rotate-180"
                           " ! appsink drop=true max_buffers=2";
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened())
    {
        printf("Could not open camera.\n");
        return 1;
    }

    // Create the OpenCV windows
    namedWindow("Camera", cv::WINDOW_AUTOSIZE);
    namedWindow("Output", WINDOW_AUTOSIZE);
    Mat frame, hsv_frame;

    // Define HSV thresholds for red, blue, and green LEDs
    int iLowH_R1 = 0, iHighH_R1 = 10, iLowH_R2 = 170, iHighH_R2 = 180;
    int iLowS_R = 150, iHighS_R = 255;
    int iLowV_R = 200, iHighV_R = 255;  // High brightness for red LEDs

    int iLowH_B = 100, iHighH_B = 140, iLowS_B = 150, iHighS_B = 255;
    int iLowV_B = 200, iHighV_B = 255;  // High brightness for blue LEDs

    int iLowH_G = 35, iHighH_G = 85, iLowS_G = 100, iHighS_G = 255;
    int iLowV_G = 200, iHighV_G = 255;  // High brightness for green LEDs

    // Structuring element for morphological operations
    int morph_size = 2;
    Mat struct_elem = getStructuringElement(MORPH_RECT, Size(morph_size, morph_size));

    Point red_center(-1, -1), blue_center(-1, -1), green_center(-1, -1);  // For LED centroids

    for (;;)
    {
        if (!cap.read(frame))
        {
            printf("Could not read a frame.\n");
            break;
        }

        // Convert the frame to HSV color space
        cvtColor(frame, hsv_frame, COLOR_BGR2HSV);

        // Threshold for red LED
        Mat thresh_img_red1, thresh_img_red2, thresh_img_red;
        inRange(hsv_frame, Scalar(iLowH_R1, iLowS_R, iLowV_R),
                Scalar(iHighH_R1, iHighS_R, iHighV_R), thresh_img_red1);
        inRange(hsv_frame, Scalar(iLowH_R2, iLowS_R, iLowV_R),
                Scalar(iHighH_R2, iHighS_R, iHighV_R), thresh_img_red2);
        thresh_img_red = thresh_img_red1 | thresh_img_red2;

        // Threshold for blue LED
        Mat thresh_img_blue;
        inRange(hsv_frame, Scalar(iLowH_B, iLowS_B, iLowV_B),
                Scalar(iHighH_B, iHighS_B, iHighV_B), thresh_img_blue);

        // Threshold for green LED
        Mat thresh_img_green;
        inRange(hsv_frame, Scalar(iLowH_G, iLowS_G, iLowV_G),
                Scalar(iHighH_G, iHighS_G, iHighV_G), thresh_img_green);

        // Morphological operations to clean up the image for each color
        morphologyEx(thresh_img_red, thresh_img_red, MORPH_CLOSE, struct_elem);
        morphologyEx(thresh_img_blue, thresh_img_blue, MORPH_CLOSE, struct_elem);
        morphologyEx(thresh_img_green, thresh_img_green, MORPH_CLOSE, struct_elem);

        // Find the centroids for each LED color
        Moments m_red = moments(thresh_img_red, true);
        if (m_red.m00 > 0)
        {
            red_center = Point((int)(m_red.m10 / m_red.m00), (int)(m_red.m01 / m_red.m00));
            circle(frame, red_center, 5, Scalar(0, 0, 255), -1);  // Draw red LED center
        }

        Moments m_blue = moments(thresh_img_blue, true);
        if (m_blue.m00 > 0)
        {
            blue_center = Point((int)(m_blue.m10 / m_blue.m00), (int)(m_blue.m01 / m_blue.m00));
            circle(frame, blue_center, 5, Scalar(255, 0, 0), -1);  // Draw blue LED center
        }

        Moments m_green = moments(thresh_img_green, true);
        if (m_green.m00 > 0)
        {
            green_center = Point((int)(m_green.m10 / m_green.m00), (int)(m_green.m01 / m_green.m00));
            circle(frame, green_center, 5, Scalar(0, 255, 0), -1);  // Draw green LED center
        }

        // Draw lines between detected LEDs
        if (red_center.x >= 0 && green_center.x >= 0)
        {
            line(frame, red_center, green_center, Scalar(0, 255, 255), 2);  // Yellow line
        }

        if (blue_center.x >= 0 && green_center.x >= 0)
        {
            line(frame, blue_center, green_center, Scalar(255, 255, 0), 2);  // Cyan line
        }

        // Calculate and display angle between green-red and green-blue
        if (green_center.x >= 0 && red_center.x >= 0 && blue_center.x >= 0)
        {
            // Vectors from green to red and green to blue
            Point vecGR = red_center - green_center;
            Point vecGB = blue_center - green_center;

            // Calculate the dot product and magnitudes
            double dotProduct = vecGR.x * vecGB.x + vecGR.y * vecGB.y;
            double magnitudeGR = sqrt(vecGR.x * vecGR.x + vecGR.y * vecGR.y);
            double magnitudeGB = sqrt(vecGB.x * vecGB.x + vecGB.y * vecGB.y);

            // Calculate the angle in radians and convert to degrees
            double angleRad = acos(dotProduct / (magnitudeGR * magnitudeGB));
            double angleDeg = angleRad * (180.0 / CV_PI);

            printf("Angle between Green-Red and Green-Blue: %f degrees\n", angleDeg);
        }

        // Display the frame and thresholded output
        imshow("Camera", frame);
        Mat combined_thresh = thresh_img_red | thresh_img_blue | thresh_img_green;
        imshow("Output", combined_thresh);

        // Slow down the frame rate
        if (waitKey(30) >= 0) break;  // Wait for 30 ms between frames
    }

    cap.release();
    return 0;
}
