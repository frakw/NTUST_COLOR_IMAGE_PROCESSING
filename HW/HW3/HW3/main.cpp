#include <iostream>
#include <cmath>
#include <vector>
#include <utility>
#include <limits>
#include <algorithm>
#include <opencv2/opencv.hpp>
#define CVPLOT_HEADER_ONLY
#include <CvPlot/cvplot.h>
#define CVUI_IMPLEMENTATION
#include "cvui.h"

#define WINDOW_NAME "rgb to gray"
using namespace std;
class HOG {
public:
    HOG() {
        data[0] = 0.0f;
        data[1] = 0.0f;
        data[2] = 0.0f;
        data[3] = 0.0f;
    }
    float distance(const HOG& input) {
        float err0 = (input.data[0] - data[0]);
        float err1 = (input.data[1] - data[1]);
        float err2 = (input.data[2] - data[2]);
        float err3 = (input.data[3] - data[3]);
        return err0 * err0 + err1 * err1 + err2 * err2 + err3 * err3;
    }
//private:
    float data[4] = {0,0,0,0};
};

float get_hogs_distance(vector<vector<HOG>>& hogsA, vector<vector<HOG>>& hogsB) {
    float result = 0.0f;
    if (hogsA.size() != hogsB.size()) return -1;
    for (int i = 0; i < hogsA.size(); i++) {
        for (int j = 0; j < hogsA[i].size(); j++) {
            result += hogsA[i][j].distance(hogsB[i][j]);
        }
    }
    return result;
}


vector<vector<HOG>> get_img_hogs(cv::Mat img) {
    cv::Mat img_x_sobel = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_32FC1);
    cv::Mat img_y_sobel = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_32FC1);
    cv::Mat img_magnitude;
    cv::Mat img_angle;
    cv::Mat x_sobel_kernel = (cv::Mat1s(3, 3) << 0, 0, 0, -1, 0, +1, 0, 0, 0);
    cv::Mat y_sobel_kernel = (cv::Mat1s(3, 3) << 0, -1, 0, 0, 0, 0, 0, +1, 0);
    cv::filter2D(img, img_x_sobel, CV_32FC1, x_sobel_kernel, cv::Point(-1, -1), 0.0f, cv::BORDER_REPLICATE);
    cv::filter2D(img, img_y_sobel, CV_32FC1, y_sobel_kernel, cv::Point(-1, -1), 0.0f, cv::BORDER_REPLICATE);
    try
    {
        cv::cartToPolar(img_x_sobel, img_y_sobel, img_magnitude, img_angle, 1);
        // ... Contents of your main
    }
    catch (cv::Exception& e)
    {
        cout << e.msg << endl;
    }
    vector<vector<HOG>> result;
    for (int i = 0, hog_i = 0; i < img_magnitude.rows; i += 4, hog_i++)
    {
        result.push_back(vector<HOG>());
        for (int j = 0, hog_j = 0; j < img_magnitude.cols; j += 4, hog_j++)
        {
            result[hog_i].push_back(HOG());
            for (int m = 0; m < 4; m++)
            {
                cv::Mat block = cv::Mat::zeros(cv::Size(4, 4), CV_32FC1);
                int row = i + m;
                for (int n = 0; n < 4; n++)
                {
                    int col = j + n;
                    int field_index = 0;
                    float angle = fmod(img_angle.at<float>(row, col), 180.0f);
                    float magnitude = img_magnitude.at<float>(row, col);
                    float splitA = 0.0f, splitB = 0.0f;

                    if (angle < 45.0f) {
                        field_index = 0;
                        splitB = ((angle - 0.0f) / 45.0f) * magnitude;
                    }
                    else if (angle >= 45.0f && angle < 90.0f) {
                        field_index = 1;
                        splitB = ((angle - 45.0f) / 45.0f) * magnitude;
                    }
                    else if (angle >= 90.0f && angle < 135.0f) {
                        field_index = 2;
                        splitB = ((angle - 90.0f) / 45.0f) * magnitude;
                    }
                    else if (angle >= 135.0f) {
                        field_index = 3;
                        splitB = ((angle - 135.0f) / 45.0f) * magnitude;
                    }
                    splitA = magnitude - splitB;
                    //cout << "index:" << field_index << '\n';
                    //cout << "angle:" << angle << '\n';
                    result[hog_i][hog_j].data[field_index] += splitA;
                    result[hog_i][hog_j].data[(field_index + 1) % 4] += splitB;
                    //cout << img_magnitude.at<float>(row, col) << '\n';
                }
            }
            //auto axes = CvPlot::plot(std::vector<double>{ img_HOG[i][j].data[0], img_HOG[i][j].data[1], img_HOG[i][j].data[2],img_HOG[i][j].data[3] }, "-o");




            //cout << "-------------------" << result[hog_i][hog_j].data[0] << endl;
            //cout << "-------------------" << result[hog_i][hog_j].data[1] << endl;
            //cout << "-------------------" << result[hog_i][hog_j].data[2] << endl;
            //cout << "-------------------" << result[hog_i][hog_j].data[3] << endl;
        }
    }
    return result;
}

int main() {
    cvui::init(WINDOW_NAME);
    cv::Mat frame = cv::Mat(cv::Size(550, 580), CV_32FC1);

    //cv::Mat origin = cv::imread("Lenna.jpg", CV_32F);
    cv::Mat img = cv::imread("ABC01.jpg", CV_32FC1);
    cv::Mat target = cv::imread("t3.jpg", CV_32FC1);
    cv::Mat result_img = cv::imread("t3.jpg", CV_8UC1);
    cv::Mat img_x_sobel = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_32FC1);
    cv::Mat img_y_sobel = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_32FC1);
    cv::Mat img_magnitude;
    cv::Mat img_angle;

    cv::resize(img, img, cv::Size(16, 16));
    cv::resize(target, target, cv::Size(128, 128));
    cv::resize(result_img, result_img, cv::Size(128,128));
    cv::cvtColor(result_img, result_img,cv::COLOR_GRAY2RGB);
    vector<vector<HOG>> alpha_hogs = get_img_hogs(img);
    vector<pair<float, cv::Point>> all_result;
    for (int i = 0; i < target.rows-16; i++) {
        for (int j = 0; j < target.cols-16; j++) {
            cv::Mat ROI = target(cv::Rect(j, i, 16, 16));
            vector<vector<HOG>> ROI_hogs = get_img_hogs(ROI);
            float result = get_hogs_distance(alpha_hogs,ROI_hogs);
            all_result.push_back(make_pair(result,cv::Point(i,j)));
        }
    }
    std::sort(all_result.begin(), all_result.end(), [](auto& left, auto& right) {
        return left.first < right.first;
        });
    for (int num = 0; num < 6; num++) {
        int i_base = all_result[num].second.x;
        int j_base = all_result[num].second.y;
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                result_img.at < cv::Vec3b >(i_base + i, j_base + j)[0] /= 2;
                result_img.at < cv::Vec3b >(i_base + i, j_base + j)[1] /= 2;
            }
        }
    }
    cv::imshow("result_img", result_img);

    /*
    while (cv::getWindowProperty(WINDOW_NAME, 0) >= 0) {
        frame = cv::Scalar(49, 52, 49);
        //cvui::text(frame, 10, 10, "Hello world!");
        //
        if (cvui::button(frame, 10, 10, 250, 30, "rgb")) {
            //img = imread("Lenna.jpg");
            origin.copyTo(img);
        }
        if (cvui::button(frame, 270, 10, 250, 30, "gray")) {
            for (int i = 0; i < img.rows; i++)
            {
                for (int j = 0; j < img.cols; j++)
                {
                    cv::Vec3b bgr = img.at<cv::Vec3b>(i, j);
                    uchar gray_value = 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2];
                    img.at<cv::Vec3b>(i, j)[0] = gray_value;
                    img.at<cv::Vec3b>(i, j)[1] = gray_value;
                    img.at<cv::Vec3b>(i, j)[2] = gray_value;
                }
            }
        }
        cvui::image(frame, 10, 50, img);
        // Update cvui internal stuff
        cvui::update();

        // Show window content
        cv::imshow(WINDOW_NAME, frame);


        if (cv::waitKey(20) == 27) {
            break;
        }
    }
    */
    while (true) {
        if (cv::waitKey(20) == 27) {
            break;
        }
    }
    return 0;
}



//cv::Mat hist;
//cv::Mat histImg(cv::Size(256 * 2, 256), CV_8UC1);
//int histHeight = 5000; //要繪製直方圖的最大高度
//float maxValue = max({ img_HOG[i][j].data[0], img_HOG[i][j].data[1], img_HOG[i][j].data[2], img_HOG[i][j].data[3] }); //直方圖中最大的bin的值
//for (size_t i = 0; i < 4; i++)//進行直方圖的繪製
//{
//    float bin_val = img_HOG[i][j].data[i];
//    int intensity = cvRound(bin_val * histHeight / maxValue);  //要繪製的高度 
//    for (size_t j = 0; j < 2; j++) //繪製直線 這裏用每scale條直線代表一個bin
//    {
//        cv::line(histImg, cv::Point(i * 2 + j, histHeight - intensity), cv::Point(i * 2 + j, histHeight - 1), 255);
//    }
//    //cv::rectangle(histImg, cv::Point(i*2, histHeight - intensity), cv::Point((i + 1)*2, histHeight - 1), 255); //利用矩形代表bin
//}
//imshow("histogram " + to_string(i) + " " + to_string(j), histImg);