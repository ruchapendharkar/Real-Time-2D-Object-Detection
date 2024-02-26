/*
    imageProcessing.cpp 
    Project 3

    Created by Rucha Pendharkar on 2/25/24

*/
/**/
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "kmeans.h"
#include "csv_util.h"
#include <unordered_map>

using namespace std;
using namespace cv;

#include <vector>

//Gaussian Blur filter
int blur(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        cout << "Input image is empty." << endl;
        return -1;
    }

    cv::Mat temp;
    temp.create(src.size(), src.type());
    dst.create(src.size(), src.type());
    cv::Vec3i resultRow = {0, 0, 0};
    cv::Vec3i resultCol = {0, 0, 0};

    // 5x1 filter (vertical filter)
    const float kernelVertical[5] = {0.1, 0.2, 0.4, 0.2, 0.1};

    // Loop through the image rows (excluding the outer two rows)
    for (int i = 2; i < src.rows - 2; i++) {
        // Get pointers to the current row in source and destination images
        const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b* tempRow = temp.ptr<cv::Vec3b>(i);

        // Loop through the image columns (excluding the outer two columns)
        for (int j = 2; j < src.cols - 2; j++) {
            // Initialize the result vector for each color channel
            cv::Vec3i result = {0, 0, 0};

            // Loop through the filter coefficients
            for (int k = -2; k <= 2; k++) {
                // Get pointers to the pixel values in the current filter window
                const cv::Vec3b* pixel = srcRow + j;
                
                // Update the result vector for each color channel
                for (int c = 0; c < 3; c++) {
                    result[c] += (*pixel)[c] * kernelVertical[k + 2];
                    pixel += src.cols;  // Move to the next row
                }
            }

            // Store the result vector in the destination image
            tempRow[j] = cv::Vec3b(result);
        }
    }

    // 1x5 filter (horizontal filter)
    const float kernelHorizontal[5] = {0.1, 0.2, 0.4, 0.2, 0.1};

    // Loop through the image rows (excluding the outer two rows)
    for (int i = 2; i < src.rows - 2; i++) {
        // Get pointers to the current row in the temporary and destination images
        const cv::Vec3b* tempRow = temp.ptr<cv::Vec3b>(i);
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(i);

        // Loop through the image columns (excluding the outer two columns)
        for (int j = 2; j < src.cols - 2; j++) {
            // Initialize the result vector for each color channel
            cv::Vec3i result = {0, 0, 0};

            // Loop through the filter coefficients
            for (int k = -2; k <= 2; k++) {
                // Update the result vector for each color channel
                for (int c = 0; c < 3; c++) {
                    result[c] += tempRow[j + k][c] * kernelHorizontal[k + 2];
                }
            }

            // Store the result vector in the destination image
            dstRow[j] = cv::Vec3b(result);
        }
    }

    return 0;
}

// Returns thresholded image 
void thresholdImage(const cv::Mat &src, cv::Mat &dst, int thresholdValue) {
    dst.create(src.size(), CV_8UC1); // Create a single-channel output image
    cv::Mat grayscale, blurred;
    //cv::GaussianBlur(src, blurred, Size(5,5), 0 );
    blur(src, blurred); //Blurs the image
    cv::cvtColor(blurred, grayscale, cv::COLOR_BGR2GRAY); // Convert to grayscale

    for (int i = 0; i < grayscale.rows; ++i) {
        for (int j = 0; j < grayscale.cols; ++j) {
            // Get the intensity value of the current pixel
            uchar intensity = grayscale.at<uchar>(i, j);

            // Apply thresholding - white foreground, black background 
            dst.at<uchar>(i, j) = (intensity > thresholdValue) ? 0 : 255;
        }
    }
}

//Clean up the binary image obtained after thresholding
cv::Mat cleanBinaryImage(const cv::Mat &binaryImage) {
    cv::Mat cleanedImage;

    // Define the structuring element
    int kernelSize = 5; 
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));

    // Apply erosion followed by dilation
    cv::morphologyEx(binaryImage, cleanedImage, cv::MORPH_OPEN, kernel, Point(-1, -1), 1);
    cv::morphologyEx(cleanedImage, cleanedImage, cv::MORPH_CLOSE, kernel, Point(-1, -1), 3);

    return cleanedImage;
}

std::vector<cv::Rect> findRegions(const cv::Mat &cleanedImage, int minArea){
    cv:: Mat labels, centroids, stats;
    std:: vector <cv::Rect> regions;

    //Apply connected components with stats 

    int numLabels = cv::connectedComponentsWithStats(cleanedImage, labels, stats, centroids);

    for (int i = 1; i<numLabels; i++){
        //Finds total area 
        int area = stats.at<int>(i, cv::CC_STAT_AREA); 
        
        //Gets information for the bounding box
        cv::Rect rect(stats.at<int>(i, cv::CC_STAT_LEFT), 
                    stats.at<int>(i, cv::CC_STAT_TOP),
                    stats.at<int>(i, cv::CC_STAT_WIDTH),
                    stats.at<int>(i, cv::CC_STAT_HEIGHT));
        
        //If area is not small, add to regions
        if (area > minArea){
            regions.push_back(rect);
        }
        
    }
    return regions;
}

void showSegmentedImage(const cv::Mat &cleanedImage, cv::Mat &frame) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(cleanedImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat frameWithFilledRegions = frame.clone();

    for (size_t i = 0; i < contours.size(); ++i) {
        // Generate a random color for each region
        cv::Scalar color(rand() & 255, rand() & 255, rand() & 255);

        // Fills the region with the specified color
        cv::fillPoly(frameWithFilledRegions, std::vector<std::vector<cv::Point>>{contours[i]}, color);
    }

    // Display the frame with filled regions
    cv::imshow("Frame with Filled Regions", frameWithFilledRegions);
}

struct RegionFeatures {
    double percentFilled;
    double hwRatio;
    double huMoments[7];
    Point2f center; 
    string label;
    RotatedRect orientedBoundingBox;
    Point2f axisParallelLinePoint1;
    Point2f axisParallelLinePoint2;
};

std::vector<RegionFeatures> computeFeaturesForRegions(const cv::Mat& cleanedImage, const std::vector<cv::Rect>& regions, cv::Mat &frame) {
    std::vector<RegionFeatures> featuresVector;

    for (const auto& region : regions) {
        // Extract the region of interest from the cleaned image
        Mat regionROI = cleanedImage(region);

        // Find contours
        vector<vector<Point>> contours;
        findContours(regionROI, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        double minContourArea = 100.0; 
        vector<vector<Point>> filteredContours;

        // Ignore smaller contours
        for (const auto& contour : contours) {
            double area = contourArea(contour);
            if (area > minContourArea) {
                filteredContours.push_back(contour);
            }
        }

        if (!filteredContours.empty()) {
            // Find the largest contour based on area
            auto largestContour = *max_element(filteredContours.begin(), filteredContours.end(),
                [](const vector<Point>& a, const vector<Point>& b) {
                    return contourArea(a) < contourArea(b);
                });

            // Calculate moments and centroid for the largest contour
            Moments m = moments(largestContour);

            Point2f center(m.m10 / m.m00, m.m01 / m.m00);

            // Centeral moment computation 
            double mu11 = m.mu11 / m.m00;
            double mu20 = m.mu20 / m.m00;
            double mu02 = m.mu02 / m.m00;

            //Angle of orientation
            double t = 0.5 * atan2(2 * mu11, mu20 - mu02);

            // Axis-parallel Line Computation Method
            double cos_t = (double)cos(t);
            double sin_t = (double)sin(t);
            double x1 = center.x; 
            double y1 = (cleanedImage.rows - 1) - center.y;
            int axisLength = 400;
            double x2 = center.x + (axisLength * cos_t );
            double y2 = (cleanedImage.rows - 1) - center.y + (axisLength * sin_t);
            //std::cout << "frame.cols: " <<  frame.cols << std::endl;

            // Compute the oriented bounding box
            RotatedRect orientedBoundingBox = minAreaRect(largestContour);
            orientedBoundingBox.center += Point2f(region.tl());

            // Compute Hu Moments
            double hu[7];
            HuMoments(m, hu);

            // Compute percentage filled
            double percentFilled = static_cast<double>(countNonZero(regionROI)) / (region.width * region.height);

            // Compute scale invariant height-to-width ratio
            double hwRatio = (std::min)(orientedBoundingBox.size.width, orientedBoundingBox.size.height) /
                             (std::max)(orientedBoundingBox.size.width, orientedBoundingBox.size.height);

            RegionFeatures features;
            features.percentFilled = percentFilled;
            features.hwRatio = hwRatio;
            features.center = center;
            memcpy(features.huMoments, hu, sizeof(hu));
            features.axisParallelLinePoint1 = Point2f(x1, y1);
            features.axisParallelLinePoint2 = Point2f(x2, y2);

            // Draw the bounding box on the original image
            Mat imageWithBoundingBox = frame.clone();
            rectangle(imageWithBoundingBox, region, Scalar(0, 255, 0), 2);
            line(imageWithBoundingBox, features.axisParallelLinePoint1, features.axisParallelLinePoint2, Scalar(0, 255, 0), 2);

            // Display the image with bounding boxes
            imshow("Image with Bounding Boxes", imageWithBoundingBox);

            // Add the features to the vector
            featuresVector.push_back(features);
        }
    }

    return featuresVector;
}


// Calculate the mean and standard deviation for each feature
std::vector<double> calculateStandardDeviations(const std::vector<RegionFeatures>& trainingSet) {
    const int numFeatures = 11;

    std::vector<double> stdDeviations(numFeatures, 0.0);

    for (int i = 0; i < numFeatures; ++i) {
        double sum = 0.0;
        double sumSquaredDiff = 0.0;

        // Calculate the mean
        for (const auto& feature : trainingSet) {
            switch (i) {
                case 0: // x-coordinate
                    sum += feature.center.x;
                    break;
                case 1: // y-coordinate
                    sum += feature.center.y;
                    break;
                case 2 ... 8: // huMoments
                    sum += feature.huMoments[i - 2];
                    break;
                case 9: // percentFilled
                    sum += feature.percentFilled;
                    break;
                case 10: // hwRatio
                    sum += feature.hwRatio;
                    break;
                default:
                    break;
            }
        }

        double mean = sum / trainingSet.size();

        // Calculate the sum of squared differences
        for (const auto& feature : trainingSet) {
            double diff = 0.0;
            switch (i) {
                case 0:
                    diff = feature.center.x - mean;
                    break;
                case 1:
                    diff = feature.center.y - mean;
                    break;
                case 2 ... 8:
                    diff = feature.huMoments[i - 2] - mean;
                    break;
                case 9:
                    diff = feature.percentFilled - mean;
                    break;
                case 10:
                    diff = feature.hwRatio - mean;
                    break;
                default:
                    break;
            }
            sumSquaredDiff += diff * diff;
        }

        // Calculate the standard deviation
        stdDeviations[i] = sqrt(sumSquaredDiff / trainingSet.size());
    }

    return stdDeviations;
}

double scaledEuclideanDistance(const RegionFeatures& a, const RegionFeatures& b, const std::vector<double>& stdDeviations) {
    double sum = 0.0;

    // Calculate the scaled difference for x-coordinate
    double scaledDiffX = (a.center.x -b.center.x) /stdDeviations[0]; ;
    sum += (scaledDiffX * scaledDiffX);

    // Calculate the scaled difference for y-coordinate
    double scaledDiffY = (a.center.y - b.center.y) /stdDeviations[1];
    sum += (scaledDiffY * scaledDiffY) ;

    // Calculate the scaled difference for huMoments
    for (int i = 0; i < 7; i++) {
        double scaledDiffHuMoment = (a.huMoments[i] - b.huMoments[i])/ stdDeviations[i + 2]; 
        sum += (scaledDiffHuMoment * scaledDiffHuMoment);
    }

    // Calculate the scaled difference for percentFilled
    double scaledDiffPercentFilled = (a.percentFilled - b.percentFilled) / stdDeviations[9];
    sum += (scaledDiffPercentFilled * scaledDiffPercentFilled);

    // Calculate the scaled difference for hwRatio
    double scaledDiffHwRatio = (a.hwRatio - b.hwRatio)/stdDeviations[10];
    sum += (scaledDiffHwRatio * scaledDiffHwRatio);

    return sum;
}

std::string nearestNeighbor(const RegionFeatures& instance, const std::vector<RegionFeatures>& trainingSet, const std::vector<double>& stdDeviations) {
    if (trainingSet.empty() || stdDeviations.empty()) {
        return "Load the Training Set";
    }

    double minDistance = scaledEuclideanDistance(instance, trainingSet[0], stdDeviations);
    std::string label = trainingSet[0].label;

    for (size_t i = 1; i < trainingSet.size(); ++i) {
        double distance = scaledEuclideanDistance(instance, trainingSet[i], stdDeviations);
        if (distance < minDistance) {
            minDistance = distance;
            label = trainingSet[i].label;
        }
    }

    return label;
}

std::string KnearestNeighbor(const RegionFeatures& instance, const std::vector<RegionFeatures>& trainingSet, const std::vector<double>& stdDeviations, size_t k) {
    if (trainingSet.empty() || stdDeviations.empty()) {
        return "Load the Training Set";
    }

    // Using an unordered_map to store the sum of distances for each class
    std::unordered_map<std::string, double> classDistances;

    for (const auto& trainingInstance : trainingSet) {
        double distance = scaledEuclideanDistance(instance, trainingInstance, stdDeviations);
        classDistances[trainingInstance.label] += distance;
    }

    // Using a vector of pairs to find the top k classes with the smallest sum of distances
    std::vector<std::pair<std::string, double>> sortedDistances(classDistances.begin(), classDistances.end());
    std::partial_sort(sortedDistances.begin(), sortedDistances.begin() + k, sortedDistances.end(),
                      [](const auto& left, const auto& right) { return left.second < right.second; });

    // The label with the smallest sum of distances is the result
    return sortedDistances[0].first;
}

std::vector<RegionFeatures> populateWithTrainingData() {
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

        for (int i = 0; i < 7; i++) {
            std::getline(ss, value, ',');
            feature.huMoments[i] = std::stod(value);
        }
        std::getline(ss, value, ',');
        feature.percentFilled = std::stod(value);

        std::getline(ss, value, ',');
        feature.hwRatio = std::stod(value);

        std::getline(ss, value, ',');
        feature.label = value;

        trainingSet.push_back(feature);
    }
    file.close();
    return trainingSet;
}

