#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>
#define CVUI_IMPLEMENTATION
#include "cvui.h"

#define WINDOW_NAME "pyrimid combine"
using namespace std;
using namespace cv;
int main() {
    Mat apple = imread("apple_900x900.png",CV_8UC4); resize(apple, apple, Size(512, 512));
    Mat material = imread("material_512x512.png", CV_8UC4);
    Mat a0 = cv::Mat::zeros(512, 512, CV_8UC4);
    Mat a1 = cv::Mat::zeros(512, 512, CV_8UC4);
    Mat d0 = cv::Mat::zeros(512, 512, CV_8UC4);
    Mat combine = cv::Mat::zeros(512, 512, CV_8UC3);
    vector<Mat> apple_layer;
    vector<Mat> material_layer;

    float rate[6] = { 1.0f,0.833,0.666f,0.5f,0.333f,0.0f };//加權比例

    //處理蘋果分解與重建
    apple.copyTo(a0);
    cv::imshow("origin", a0);
    for (int i = 0; i < 6; i++) {
        GaussianBlur(a0, a1, Size(5, 5), 0);//高斯模糊存到a1
        resize(a1, a1, Size(a0.rows / 2, a0.cols / 2));//a1縮小一倍
        resize(a1, d0, Size(a1.rows * 2, a1.cols * 2));//縮大一倍存到d0
        //d0與a0相減
        for (int row = 0; row < d0.rows; row++) {
            for (int col = 0; col < d0.cols; col++) {
                for (int color = 0; color < 3; color++) {
                    int result_color = (d0.at<Vec3b>(row, col)[color] - a0.at<Vec3b>(row, col)[color] + 127);
                    if (result_color > 255) result_color = 255;
                    else if (result_color < 0) result_color = 0;
                    d0.at<Vec3b>(row, col)[color] = result_color;
                }
            }
        }

        resize(d0, d0, Size(512, 512));//放大以展示
        cv::imshow("apple d0-" + std::to_string(i), d0);

        resize(a0, a0, Size(a0.rows / 2, a0.cols / 2));//分解圖，縮小a0來放上一層
        a1.copyTo(a0);//把上一層資料放入a0

        resize(a1, a1, Size(512, 512));//放大以展示
        cv::imshow("apple a1-" + std::to_string(5 - i), a1);//重建圖

        apple_layer.push_back(a1);
    }


    //處理材質圖分解與重建
    material.copyTo(a0);
    for (int i = 0; i < 6; i++) {
        GaussianBlur(a0, a1, Size(5, 5), 0);//高斯模糊存到a1
        resize(a1, a1, Size(a0.rows / 2, a0.cols / 2));//a1縮小一倍
        resize(a1, d0, Size(a1.rows * 2, a1.cols * 2));//縮大一倍存到d0
        //d0與a0相減
        for (int row = 0; row < d0.rows; row++) {
            for (int col = 0; col < d0.cols; col++) {
                for (int color = 0; color < 3; color++) {
                    int result_color = (d0.at<Vec3b>(row, col)[color] - a0.at<Vec3b>(row, col)[color] + 127);
                    if (result_color > 255) result_color = 255;
                    else if (result_color < 0) result_color = 0;
                    d0.at<Vec3b>(row, col)[color] = result_color;
                }
            }
        }

        resize(d0, d0, Size(512, 512));//放大以展示
        cv::imshow("material d0-" + std::to_string(i), d0);

        resize(a0, a0, Size(a0.rows / 2, a0.cols / 2));//分解圖，縮小a0來放上一層
        a1.copyTo(a0);//把上一層資料放入a0

        resize(a1, a1, Size(512, 512));//放大以展示
        cv::imshow("material a1-" + std::to_string(5 - i), a1);//重建圖

        material_layer.push_back(a1);
    }

    //融合圖片
    for (int i = 0; i < 6; i++) {
        for (int row = 0; row < 512; row++) {
            for (int col = 0; col < 512; col++) {
                for (int color = 0; color < 3; color++) {
                    combine.at<Vec3b>(row, col)[color] = (apple_layer[i].at<Vec3b>(row, col)[color] * (1.0f - rate[i])) + (material_layer[i].at<Vec3b>(row, col)[color] * rate[i]);
                }
            }
        }
        cv::imshow("combine" + std::to_string(5 - i), combine);//重建圖
    }

    waitKey();
    return 0;
}