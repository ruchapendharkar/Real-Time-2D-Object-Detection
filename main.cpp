/*
    main.cpp 
    Project 3

    Created by Rucha Pendharkar on 2/25/24

*/


#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "imageProcessing.cpp"
#include <cmath> 

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev;

    // open the video device

    //replace with Ip address of your phone
    capdev = new cv::VideoCapture("http://10.0.0.224:8080/video");

    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return (-1);
    }

    cv::namedWindow("Video", 1); // identifies a window
    cv::Mat frame, thresholded, imageWithBoundingBox;
    int count = -1;

    std::vector<RegionFeatures> trainingSet;
    std::ifstream file("training_set.csv");
    std::string line;

    // Skip the header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        RegionFeatures feature;
        std::string value;

        // Read centroid x and y
        std::getline(ss, value, ',');
        feature.center.x = std::stoi(value);
        std::getline(ss, value, ',');
        feature.center.y = std::stoi(value);
        //Read Hu Moments 
        for (int i = 0; i < 7; i++) {
            std::getline(ss, value, ',');
            feature.huMoments[i] = std::stod(value);
        }
        //Read percentFilled
        std::getline(ss, value, ',');
        feature.percentFilled = std::stod(value);

        std::getline(ss, value, ',');
        feature.hwRatio = std::stod(value);

        std::getline(ss, value, ',');
        feature.label = value;

        trainingSet.push_back(feature);
    }
    file.close();

    std::vector<double> stdDeviations = calculateStandardDeviations(trainingSet);

    for (int i = 0; i < 11; ++i) {
        std::cout<<"STD : " << stdDeviations[i] << std::endl;}

    for (;;) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }
        // Display Original and Thresholded frame
        cv::imshow("Video", frame);
        cv::imwrite("/home/rucha/CS5330/Project3/Result/Image.jpg", frame);
        thresholdImage(frame, thresholded, 90);
        cv::imshow("Thresholded Video", thresholded);
        cv::imwrite("/home/rucha/CS5330/Project3/Result/ThresholdedImage.jpg", thresholded);

        // Clean Binary Image
        cv::Mat cleanedImage = cleanBinaryImage(thresholded);
        cv::imshow("Cleaned Video", cleanedImage);
        cv::imwrite("/home/rucha/CS5330/Project3/Result/cleanedImage.jpg", cleanedImage);

        // Find regions
        std::vector<cv::Rect> regions = findRegions(cleanedImage, 100);

        // Show segmented image
        showSegmentedImage(cleanedImage, frame);

        //Compute features in regions
        vector<RegionFeatures> featuresVectors = computeFeaturesForRegions(cleanedImage, regions, frame);

        cv::Mat resultFrame = frame.clone();    
        
        // Classify each region
        
        for (RegionFeatures& feature : featuresVectors) {
            feature.label = nearestNeighbor(feature, trainingSet,stdDeviations);
            std::cout << "Predicted label: " << feature.label << std::endl;

            // Display the predicted label
            cv::putText(resultFrame, feature.label, feature.center + cv::Point2f(20, 0), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 255));
        }

        // Display the original frame with regions, features, and labels
        cv::imshow("Classified Object", resultFrame);
        
        //Classify using K Nearest Neighbor

        for (RegionFeatures& feature : featuresVectors) {
            feature.label = KnearestNeighbor(feature, trainingSet,stdDeviations,9);
            std::cout << "Predicted label using KNN: " << feature.label << std::endl;

            // Display the predicted label
            cv::putText(resultFrame, feature.label, feature.center + cv::Point2f(20, 0), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 255));
        }
        cv::imshow("Classified Object", resultFrame);

        // see if there is a waiting keystroke
        char key = cv::waitKey(10);


        // Enter training mode if n is pressed
        if (key == 'n') {
            std::string label;
            std::cout << "Enter the label for the current object - " << std::endl;
            std::cin >> label;

            // Find the region with the largest area
            RegionFeatures* largestRegion = &featuresVectors[0];
            for (RegionFeatures& feature : featuresVectors) {
                if (feature.orientedBoundingBox.size.area() > largestRegion->orientedBoundingBox.size.area()) {
                    largestRegion = &feature;
                }
            }

            // Add the largest region's features to the training set
            largestRegion->label = label;
            trainingSet.push_back(*largestRegion);

            // After the loop, save the training set to a CSV file
            std::ifstream infile("training_set.csv");
            infile.close();

            std::ofstream file("training_set.csv", std::ios::app); // Open in append mode

            RegionFeatures& feature = trainingSet.back();
            file << feature.center.x << "," << feature.center.y << ","
                << feature.huMoments[0] << "," << feature.huMoments[1] << "," << feature.huMoments[2] << "," << feature.huMoments[3] << ","
                << feature.huMoments[4] << "," << feature.huMoments[5] << "," << feature.huMoments[6] << ","
                << feature.percentFilled << "," << feature.hwRatio << ","
                << feature.label << "\n";
            file.close();
        }

        // exit if q is pressed
        else if (key == 'q') {
            break;
        }
    }

    delete capdev;
    return (0);
}
