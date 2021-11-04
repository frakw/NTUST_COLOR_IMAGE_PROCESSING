#include <iostream>
#include <random>
#include <time.h>
#include <opencv2/opencv.hpp>
#define CVUI_IMPLEMENTATION
#include "cvui.h"
#if _WIN32
#define DEFAULT_PATH "./"
#else
#define DEFAULT_PATH "/tmp"
#endif

#define WINDOW_NAME "B10815057_廖聖郝_彩色影像處理_作業1"
using namespace cv;

void get_gaussian_noise_table(double* rand_mapping_table, int gaussian_range, double gaussian_sigma) {
    double y_sum = 0.0f;
    double cumulative = 0.0f;
    for (int x = -gaussian_range, index = 0; x <= gaussian_range; x++, index++) {
        double y = exp(-(double)(x * x) / (2.0f * gaussian_sigma * gaussian_sigma));
        y_sum += y;
        rand_mapping_table[index] = y;
    }

    for (int x = -gaussian_range, index = 0; x <= gaussian_range; x++, index++) {
        rand_mapping_table[index] /= y_sum;
        cumulative += rand_mapping_table[index];
        rand_mapping_table[index] = cumulative;
    }
}

int gaussian_noise(double* rand_mapping_table, int gaussian_range, double random_val) {
    int index = 0;
    int x;
    for (x = -gaussian_range; x <= gaussian_range; x++, index++) {
        if (rand_mapping_table[index] > random_val) break;
    }
    if(x > gaussian_range)x = gaussian_range;
    return x;
}

constexpr int DOUBLE_MIN = 0;
constexpr int DOUBLE_MAX = 1;
std::random_device rd;
std::default_random_engine eng(rd());
std::uniform_real_distribution<float> distr(DOUBLE_MIN, DOUBLE_MAX);

double random01() {
    return distr(eng);
}

int main() {
    std::locale loc = std::locale::global(std::locale(""));
    cvui::init(WINDOW_NAME);
    cv::Mat frame = cv::Mat(cv::Size(1490, 520), CV_8UC1); //彩色用CV_8UC3
    std::string filename = "ntust_gray.jpg";
    Mat source_img = imread(filename, 0);
    Mat destination_image = imread(filename, 0);
    
    bool have_noise = false;
    double gaussian_sigma = 50.0f;
    double pre_gaussian_sigma = gaussian_sigma;

    int gaussian_range = 100;
    int pre_gaussian_range = gaussian_range;
    double* rand_mapping_table = new double[gaussian_range * 2 + 1];

    double salt_pepper_N_percent = 0.15f;
    get_gaussian_noise_table(rand_mapping_table, gaussian_range, gaussian_sigma);

    while (cv::getWindowProperty(WINDOW_NAME, 0) >= 0) {
        frame = cv::Scalar(49, 52, 49);
        

        


        cvui::printf(frame, 10, 10, "source image");
        if (!source_img.empty()) cvui::image(frame, 10, 30, source_img);
        if (!have_noise) {
            if (cvui::button(frame, 670, 100, 150, 30, "Gaussian noise")) {
                for (int i = 0; i < source_img.rows; i++) {
                    for (int j = 0; j < source_img.cols; j++) {
                        int noise = gaussian_noise(rand_mapping_table, gaussian_range, random01());
                        int base_color = source_img.at<uchar>(i, j);
                        int result = base_color + noise;
                        if (result > 255) result = 255;
                        else if (result < 0) result = 0;
                        destination_image.at<uchar>(i, j) = result;
                    }
                }
                have_noise = true;
            }
            cvui::counter(frame, 670, 20, &gaussian_range);
            if (gaussian_range > 255) { gaussian_range = 255; }
            else if (gaussian_range < 0) { gaussian_range = 0; }
            cvui::trackbar(frame, 670, 40, 150, &gaussian_sigma, (double)0.1f, (double)100.0f);
            cvui::trackbar(frame, 670, 140, 150, &salt_pepper_N_percent, (double)0.0f, (double)1.0f);
            if (cvui::button(frame, 670, 200, 150, 30, "Salt pepper noise")) {
                for (int i = 0; i < source_img.rows; i++) {
                    for (int j = 0; j < source_img.cols; j++) {
                        destination_image.at<uchar>(i, j) = random01() > salt_pepper_N_percent ? source_img.at<uchar>(i, j) : (random01() > 0.5f ? 255 : 0);
                    }
                }
                have_noise = true;
            }
        }
        else {
            if (cvui::button(frame, 670, 80, 150, 30, "Mean filter")) {
                
            }
            if (cvui::button(frame, 670, 200, 150, 30, "Median filter")) {
                
            }
        }
        if (cvui::button(frame, 670, 320, 150,30, "transform"));
        if (cvui::button(frame, 670, 440, 150, 30, "reset")) {
            source_img.copyTo(destination_image);
            have_noise = false;
        }
        cvui::printf(frame, 840, 10, "destination image");
        if (!destination_image.empty()) cvui::image(frame, 840, 30, destination_image);
        // Update cvui internal stuff

        if (gaussian_sigma != pre_gaussian_sigma) {
            get_gaussian_noise_table(rand_mapping_table, gaussian_range, gaussian_sigma);
        }
        pre_gaussian_sigma = gaussian_sigma;

        if (gaussian_range != pre_gaussian_range) {
            if (rand_mapping_table != nullptr) delete[] rand_mapping_table;
            rand_mapping_table = new double[gaussian_range * 2 + 1];
            get_gaussian_noise_table(rand_mapping_table, gaussian_range, gaussian_sigma);
        }
        pre_gaussian_range = gaussian_range;

        cvui::update();

        // Show window content
        cv::imshow(WINDOW_NAME, frame);


        if (cv::waitKey(20) == 27) {
            break;
        }
       
    }

    return 0;
}