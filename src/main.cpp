#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <string.h>
#include <curl/curl.h>
#include <sstream>
#include <thread>
#include <chrono>
#include <errno.h>
#include <stdio.h>

using namespace cv;
using namespace std;

#define UART_DEVICE "/dev/serial0"
#define BAUD_RATE B115200

// Define possible vehicle states
enum VehicleState { STOPPED, MOVING_FORWARD, TURNING_LEFT, TURNING_RIGHT };

double angleDeg;
// Vehicle state to track current behavior
VehicleState current_state = STOPPED;

// UART setup
int uart_fd = -1;  // File descriptor for UART

int setup_uart() {
    uart_fd = open(UART_DEVICE, O_RDWR | O_NOCTTY);
    if (uart_fd == -1) {
        return -1;  // Return error if UART cannot be opened
    }

    struct termios options;
    tcgetattr(uart_fd, &options);
    options.c_cflag = BAUD_RATE | CS8 | CLOCAL | CREAD;
    options.c_iflag = IGNPAR;
    options.c_oflag = 0;
    options.c_lflag = 0;

    tcflush(uart_fd, TCIFLUSH);
    tcsetattr(uart_fd, TCSANOW, &options);

    return 0;
}

// Function to send UART command
void send_uart_command(const char* message) {
    if (uart_fd != -1) {
        write(uart_fd, message, strlen(message));
    }
}

// Vehicle control functions
void turnleft(){
    if (current_state != TURNING_LEFT) {
    send_uart_command("a");  // Send 'a' for left turn
    current_state = TURNING_LEFT;
    printf("Command: TURN_LEFT\n");
    }
} 

void turnright(){
    if (current_state != TURNING_RIGHT) {
    send_uart_command("d");  // Send 'd' for right turn
     current_state = TURNING_RIGHT;
    printf("Command: TURN_RIGHT\n");
    }
}

void moveVehicleForward(){
    if (current_state != MOVING_FORWARD) {
        send_uart_command("w");  // Send 'w' to move forward
        current_state = MOVING_FORWARD;  // Update state to moving forward
        printf("Command: MOVE_FORWARD\n");
    }
}

void stopVehicle(){
    if (current_state != STOPPED) {
        send_uart_command("s");  // Send 's' to stop the vehicle
        current_state = STOPPED;  // Update state to stopped
        printf("Command: STOP\n");
    }
}

// HTTP callback function
size_t http_callback(void *buffer, size_t sz, size_t nmemb, void *userp) {
    size_t size = sz * nmemb;
    if (size > 0) {
        fwrite(buffer, sz, nmemb, stdout);
        printf("\n");
    } else {
        printf("Received an empty response.\n");
    }
    return size;
}

// Function to send data to ThingSpeak in a separate thread
void sendDataToThingSpeak(double length_RG, double length_BG, double distance_ultra) {
    CURL *curl = curl_easy_init();
    if (!curl) {
        printf("Failed to initialise the curl library\n");
        return;
    }

    std::string apiKey = "Q3C2LDR4T1UKJFER"; // Your API key
    std::string baseUrl = "http://api.thingspeak.com/update?api_key=";

    // Build the full URL with vector data fields
    std::ostringstream url;
    url << baseUrl << apiKey
        << "&field1=" << length_RG
        << "&field2=" << length_BG
        << "&field3=" << distance_ultra;

    // Set the callback function to handle the response
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, http_callback);
    curl_easy_setopt(curl, CURLOPT_URL, url.str().c_str());

    // Perform the HTTP request in a separate thread to avoid blocking
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
    }

    // Clean up
    curl_easy_cleanup(curl);
}

int main(){
    if (setup_uart() != 0){
        printf("UART setup failed\n");
        return -1;  // Exit if UART setup fails
    }

    // Open the video camera.
    std::string pipeline = "libcamerasrc"
        " ! video/x-raw, width=800, height=600"
        " ! videoconvert"
        " ! videoscale"
        " ! video/x-raw, width=400, height=300"
        " ! videoflip method=rotate-180"
        " ! appsink drop=true max_buffers=2";
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    if (!cap.isOpened()){
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

    // Track time for ThingSpeak update
    auto lastUpdate = chrono::steady_clock::now();
    int updateInterval = 15000; // 15 seconds

    for (;;){
        if (!cap.read(frame)){
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
        if (m_red.m00 > 0){
            red_center = Point((int)(m_red.m10 / m_red.m00), (int)(m_red.m01 / m_red.m00));
            circle(frame, red_center, 5, Scalar(0, 0, 255), -1);  // Draw red LED center
        }

        Moments m_blue = moments(thresh_img_blue, true);
        if (m_blue.m00 > 0){
            blue_center = Point((int)(m_blue.m10 / m_blue.m00), (int)(m_blue.m01 / m_blue.m00));
            circle(frame, blue_center, 5, Scalar(255, 0, 0), -1);  // Draw blue LED center
        }

        Moments m_green = moments(thresh_img_green, true);
        if (m_green.m00 > 0){
            green_center = Point((int)(m_green.m10 / m_green.m00), (int)(m_green.m01 / m_green.m00));
            circle(frame, green_center, 5, Scalar(0, 255, 0), -1);  // Draw green LED center
        }

        // Draw lines between detected LEDs
        int32_t length_RG = -1, length_BG = -1;
        if (red_center.x >= 0 && green_center.x >= 0){
            line(frame, red_center, green_center, Scalar(0, 255, 255), 2);  // Yellow line

            // Calculate the distance (length) between red and green
            length_RG = sqrt(pow(red_center.x - green_center.x, 2) + pow(red_center.y - green_center.y, 2));
        }

        if (blue_center.x >= 0 && green_center.x >= 0){
            line(frame, blue_center, green_center, Scalar(255, 255, 0), 2);  // Cyan line

            // Calculate the distance (length) between blue and green
            length_BG = sqrt(pow(blue_center.x - green_center.x, 2) + pow(blue_center.y - green_center.y, 2));
        }

        // Vehicle control logic based on blue-green distance and angle
        if (green_center.x >= 0 && red_center.x >= 0 && blue_center.x >= 0){
            // Vectors from green to blue (for front) and green to red
            Point vecGB = blue_center - green_center;
            Point vecGR = red_center - green_center;

            // Calculate the dot product and magnitudes
            double dotProduct = vecGB.x * vecGR.x + vecGB.y * vecGR.y;
            double magnitudeGB = sqrt(vecGB.x * vecGB.x + vecGB.y * vecGB.y);
            double magnitudeGR = sqrt(vecGR.x * vecGR.x + vecGR.y * vecGR.y);

            // Calculate the angle in radians and convert to degrees
            double angleRad = acos(dotProduct / (magnitudeGB * magnitudeGR));
            double angleDeg = angleRad * (180.0 / CV_PI);

            // Use the cross product to determine the sign of the angle
            double crossProduct = vecGB.x * vecGR.y - vecGB.y * vecGR.x;
            if (crossProduct < 0){
                angleDeg = -angleDeg;  // Make the angle negative for right turns
            }
            
            if (length_BG <= 15){
                stopVehicle();  // Stop when blue-green distance is small
            }
            else {
                if (fabs(angleDeg) <= 3){
                    moveVehicleForward();  // Move forward if angle is within + or - 3 degrees.
                } 
                else if (angleDeg > 3){  // Turn left if angle greater than 3 degrees.
                    turnleft();
                } 
                else if (angleDeg < -3){  // Turn right if angle less than 3 degrees.
                    turnright();
                }
            }
        }

        // Display the frame and thresholded output
        imshow("Camera", frame);
        Mat combined_thresh = thresh_img_red | thresh_img_blue | thresh_img_green;
        imshow("Output", combined_thresh);

         // Take time at intervals in different thread
      auto currentTime = chrono::steady_clock::now();
        if (chrono::duration_cast<chrono::milliseconds>(currentTime - lastUpdate).count() >= updateInterval) {
            // Check for any RX bytes
            unsigned char rx_buffer[256];
            memset(rx_buffer, '\0', sizeof(rx_buffer));

            // Read up to 255 characters from the port if they are there
            int rx_length = read(uart_fd, (void*)rx_buffer, 255);
            if (rx_length < 0) {
                // Error occurred
            // printf("UART RX error: %s\n", strerror(errno));
            } else if (rx_length == 0) {
                // No data waiting
            } else {
                // Data received
                rx_buffer[rx_length] = '\0';  // Terminate the string
                printf("%i bytes read: %s\n", rx_length, rx_buffer);
            }
            int distance_ultra; // For easy view on ThingSpeak (no decimals)

            //Scan the UART for Distance: ...
            sscanf((const char*)rx_buffer, "%*[^D]Distance: %d", &distance_ultra);
            // Send values to ThingSpeak to plot
            thread sendThread(sendDataToThingSpeak, length_RG, length_BG, distance_ultra);
            sendThread.detach();  // Detach thread to avoid blocking
            lastUpdate = currentTime;
        }

        if (waitKey(30) >= 0) break;  // Exit loop on key press
    }

    cap.release();
    close(uart_fd);  // Close UART when done
    return 0;
}