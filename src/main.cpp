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

    // Create the OpenCV windows
    namedWindow("Camera", cv::WINDOW_AUTOSIZE);
    namedWindow("Control", WINDOW_AUTOSIZE);
    namedWindow("Output", WINDOW_AUTOSIZE);
    Mat frame;

    // Measure the frame rate - initialise variables
    int frame_id = 0;
    timeval start, end;
    gettimeofday(&start, NULL);
    Mat hsv_frame;
    
    // Green color range
    int iLowH_G = 35; 
    int iHighH_G = 85;
    int iLowS_G = 40;
    int iHighS_G = 255;
    int iLowV_G = 40;
    int iHighV_G = 255;
    
    // Blue color range
    int iLowH_B = 100;
    int iHighH_B = 140;
    int iLowS_B = 150;
    int iHighS_B = 255;
    int iLowV_B = 50;
    int iHighV_B = 255;

    // Red color range (note: red can wrap around the Hue range)
    int iLowH_R1 = 0;
    int iHighH_R1 = 10;
    int iLowH_R2 = 170;
    int iHighH_R2 = 180;
    int iLowS_R = 150;
    int iHighS_R = 255;
    int iLowV_R = 50;
    int iHighV_R = 255;

    int morph = 2;

    // Create trackbars for green in "Control" window
    createTrackbar("LowH_G", "Control", &iLowH_G, 179);
    createTrackbar("HighH_G", "Control", &iHighH_G, 179);
    createTrackbar("LowS_G", "Control", &iLowS_G, 255);
    createTrackbar("HighS_G", "Control", &iHighS_G, 255);
    createTrackbar("LowV_G", "Control", &iLowV_G, 255);
    createTrackbar("HighV_G", "Control", &iHighV_G, 255);

    // Create trackbars for blue in "Control" window
    createTrackbar("LowH_B", "Control", &iLowH_B, 179);
    createTrackbar("HighH_B", "Control", &iHighH_B, 179);
    createTrackbar("LowS_B", "Control", &iLowS_B, 255);
    createTrackbar("HighS_B", "Control", &iHighS_B, 255);
    createTrackbar("LowV_B", "Control", &iLowV_B, 255);
    createTrackbar("HighV_B", "Control", &iHighV_B, 255);

    // Create trackbars for red in "Control" window
    createTrackbar("LowH_R1", "Control", &iLowH_R1, 179);
    createTrackbar("HighH_R1", "Control", &iHighH_R1, 179);
    createTrackbar("LowH_R2", "Control", &iLowH_R2, 179);
    createTrackbar("HighH_R2", "Control", &iHighH_R2, 179);
    createTrackbar("LowS_R", "Control", &iLowS_R, 255);
    createTrackbar("HighS_R", "Control", &iHighS_R, 255);
    createTrackbar("LowV_R", "Control", &iLowV_R, 255);
    createTrackbar("HighV_R", "Control", &iHighV_R, 255);

    createTrackbar("MorphSize", "Control", &morph, 10);
    
    Point green_center(-1, -1), blue_center(-1, -1), red_center(-1, -1);

    for (;;)
    {
        if (!cap.read(frame))
        {
            printf("Could not read a frame.\n");
            break;
        }

        // Convert to HSV color space
        cvtColor(frame, hsv_frame, COLOR_BGR2HSV);

        // Threshold the image for green color
        Mat thresh_img_green, thresh_img_blue, thresh_img_red1, thresh_img_red2, thresh_img_red;
        inRange(hsv_frame, Scalar(iLowH_G, iLowS_G, iLowV_G),
                Scalar(iHighH_G, iHighS_G, iHighV_G), thresh_img_green);
        
        // Threshold the image for blue color
        inRange(hsv_frame, Scalar(iLowH_B, iLowS_B, iLowV_B),
                Scalar(iHighH_B, iHighS_B, iHighV_B), thresh_img_blue);

        // Threshold the image for red color (accounting for wrap-around)
        inRange(hsv_frame, Scalar(iLowH_R1, iLowS_R, iLowV_R),
                Scalar(iHighH_R1, iHighS_R, iHighV_R), thresh_img_red1);
        inRange(hsv_frame, Scalar(iLowH_R2, iLowS_R, iLowV_R),
                Scalar(iHighH_R2, iHighS_R, iHighV_R), thresh_img_red2);
        thresh_img_red = thresh_img_red1 | thresh_img_red2;

        // Morphological operations for green
        morphologyEx(thresh_img_green, thresh_img_green, MORPH_CLOSE,
                     getStructuringElement(MORPH_RECT, Size(morph, morph)));
        morphologyEx(thresh_img_green, thresh_img_green, MORPH_OPEN,
                     getStructuringElement(MORPH_RECT, Size(morph, morph)));

        // Morphological operations for blue
        morphologyEx(thresh_img_blue, thresh_img_blue, MORPH_CLOSE,
                     getStructuringElement(MORPH_RECT, Size(morph, morph)));
        morphologyEx(thresh_img_blue, thresh_img_blue, MORPH_OPEN,
                     getStructuringElement(MORPH_RECT, Size(morph, morph)));

        // Morphological operations for red
        morphologyEx(thresh_img_red, thresh_img_red, MORPH_CLOSE,
                     getStructuringElement(MORPH_RECT, Size(morph, morph)));
        morphologyEx(thresh_img_red, thresh_img_red, MORPH_OPEN,
                     getStructuringElement(MORPH_RECT, Size(morph, morph)));

        // Calculate Moments and centroid of the green blob
        Moments m_green = moments(thresh_img_green, true);
        if (m_green.m00 > 0)
        {
            double x = m_green.m10 / m_green.m00;
            double y = m_green.m01 / m_green.m00;
            green_center = Point((int)x, (int)y);
            printf("Green Centre of mass is (%f,%f)\n", x, y);
            circle(frame, green_center, 5, Scalar(0, 255, 0), -1);
        }

        // Calculate Moments and centroid of the blue blob
        Moments m_blue = moments(thresh_img_blue, true);
        if (m_blue.m00 > 0)
        {
            double x = m_blue.m10 / m_blue.m00;
            double y = m_blue.m01 / m_blue.m00;
            blue_center = Point((int)x, (int)y);
            printf("Blue Centre of mass is (%f,%f)\n", x, y);
            circle(frame, blue_center, 5, Scalar(255, 0, 0), -1);
        }

        // Calculate Moments and centroid of the red blob
        Moments m_red = moments(thresh_img_red, true);
        if (m_red.m00 > 0)
        {
            double x = m_red.m10 / m_red.m00;
            double y = m_red.m01 / m_red.m00;
            red_center = Point((int)x, (int)y);
            printf("Red Centre of mass is (%f,%f)\n", x, y);
            circle(frame, red_center, 5, Scalar(0, 0, 255), -1);
        }

        // Draw lines between the green and blue, and green and red centers
        if (green_center.x >= 0 && blue_center.x >= 0)
        {
            line(frame, green_center, blue_center, Scalar(255, 255, 0), 2); // Yellow line
        }

        if (green_center.x >= 0 && red_center.x >= 0)
        {
            line(frame, green_center, red_center, Scalar(255, 0, 255), 2); // Magenta line
        }

        // Calculate and print the angle between the two lines
        if (green_center.x >= 0 && blue_center.x >= 0 && red_center.x >= 0)
        {
            // Vectors from green to blue and green to red
            Point vecGB = blue_center - green_center;
            Point vecGR = red_center - green_center;

            // Calculate dot product and magnitudes
            double dotProduct = vecGB.x * vecGR.x + vecGB.y * vecGR.y;
            double magnitudeGB = sqrt(vecGB.x * vecGB.x + vecGB.y * vecGB.y);
            double magnitudeGR = sqrt(vecGR.x * vecGR.x + vecGR.y * vecGR.y);

            // Calculate the angle in radians and convert to degrees
            double angleRad = acos(dotProduct / (magnitudeGB * magnitudeGR));
            double angleDeg = angleRad * (180.0 / CV_PI);

            printf("Angle between Green-Blue and Green-Red: %f degrees\n", angleDeg);
        }

        // Display frames
        imshow("Camera", frame);
        imshow("Output", thresh_img_green | thresh_img_blue | thresh_img_red);

        // Slow down the frame rate
        waitKey(500); // 500 ms delay

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
