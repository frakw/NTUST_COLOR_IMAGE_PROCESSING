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

    float rate[6] = { 1.0f,0.833,0.666f,0.5f,0.333f,0.0f };//�[�v���

    //�B�zī�G���ѻP����
    apple.copyTo(a0);
    cv::imshow("origin", a0);
    for (int i = 0; i < 6; i++) {
        GaussianBlur(a0, a1, Size(5, 5), 0);//�����ҽk�s��a1
        resize(a1, a1, Size(a0.rows / 2, a0.cols / 2));//a1�Y�p�@��
        resize(a1, d0, Size(a1.rows * 2, a1.cols * 2));//�Y�j�@���s��d0
        //d0�Pa0�۴�
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

        resize(d0, d0, Size(512, 512));//��j�H�i��
        cv::imshow("apple d0-" + std::to_string(i), d0);

        resize(a0, a0, Size(a0.rows / 2, a0.cols / 2));//���ѹϡA�Y�pa0�ө�W�@�h
        a1.copyTo(a0);//��W�@�h��Ʃ�Ja0

        resize(a1, a1, Size(512, 512));//��j�H�i��
        cv::imshow("apple a1-" + std::to_string(5 - i), a1);//���ع�

        apple_layer.push_back(a1);
    }


    //�B�z����Ϥ��ѻP����
    material.copyTo(a0);
    for (int i = 0; i < 6; i++) {
        GaussianBlur(a0, a1, Size(5, 5), 0);//�����ҽk�s��a1
        resize(a1, a1, Size(a0.rows / 2, a0.cols / 2));//a1�Y�p�@��
        resize(a1, d0, Size(a1.rows * 2, a1.cols * 2));//�Y�j�@���s��d0
        //d0�Pa0�۴�
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

        resize(d0, d0, Size(512, 512));//��j�H�i��
        cv::imshow("material d0-" + std::to_string(i), d0);

        resize(a0, a0, Size(a0.rows / 2, a0.cols / 2));//���ѹϡA�Y�pa0�ө�W�@�h
        a1.copyTo(a0);//��W�@�h��Ʃ�Ja0

        resize(a1, a1, Size(512, 512));//��j�H�i��
        cv::imshow("material a1-" + std::to_string(5 - i), a1);//���ع�

        material_layer.push_back(a1);
    }

    //�ĦX�Ϥ�
    for (int i = 0; i < 6; i++) {
        for (int row = 0; row < 512; row++) {
            for (int col = 0; col < 512; col++) {
                for (int color = 0; color < 3; color++) {
                    combine.at<Vec3b>(row, col)[color] = (apple_layer[i].at<Vec3b>(row, col)[color] * (1.0f - rate[i])) + (material_layer[i].at<Vec3b>(row, col)[color] * rate[i]);
                }
            }
        }
        cv::imshow("combine" + std::to_string(5 - i), combine);//���ع�
    }

    waitKey();
    return 0;
}