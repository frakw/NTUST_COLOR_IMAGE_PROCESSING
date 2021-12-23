#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <utility>
#include <limits>
#include <algorithm>
#include <opencv2/opencv.hpp>
#define CVPLOT_HEADER_ONLY
#include <CvPlot/cvplot.h>
#define CVUI_IMPLEMENTATION
#include "cvui.h"

#define WINDOW_NAME "HOG"
using namespace std;
class HOG {
public:
    HOG() {
        data[0] = 0.0f;
        data[1] = 0.0f;
        data[2] = 0.0f;
        data[3] = 0.0f;
    }
    //計算2個HOG之間的L2距離
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

//計算2張圖片的所有HOG之間的總L2距離
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

//回傳圖片的所有HOG(以2維vector存放)
vector<vector<HOG>> get_img_hogs(cv::Mat img,bool show_hist) {
    cv::Mat img_x_sobel = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_32FC1);
    cv::Mat img_y_sobel = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_32FC1);
    cv::Mat img_magnitude;
    cv::Mat img_angle;
    //sobel kernel
    cv::Mat x_sobel_kernel = (cv::Mat1s(3, 3) << 0, 0, 0, -1, 0, +1, 0, 0, 0);
    cv::Mat y_sobel_kernel = (cv::Mat1s(3, 3) << 0, -1, 0, 0, 0, 0, 0, +1, 0);
    //產生垂直與水平的sobel圖
    cv::filter2D(img, img_x_sobel, CV_32FC1, x_sobel_kernel, cv::Point(-1, -1), 0.0f, cv::BORDER_REPLICATE);
    cv::filter2D(img, img_y_sobel, CV_32FC1, y_sobel_kernel, cv::Point(-1, -1), 0.0f, cv::BORDER_REPLICATE);
    try
    {
        //算出magnitude與angle
        cv::cartToPolar(img_x_sobel, img_y_sobel, img_magnitude, img_angle, 1); //
        // ... Contents of your main
    }
    catch (cv::Exception& e)
    {
        cout << e.msg << endl;
    }
    vector<vector<HOG>> result;
    //掃過圖片中的每個子區域(4*4)，並產生HOG
    for (int i = 0, hog_i = 0; i < img_magnitude.rows; i += 4, hog_i++)
    {
        result.push_back(vector<HOG>());
        for (int j = 0, hog_j = 0; j < img_magnitude.cols; j += 4, hog_j++)
        {
            result[hog_i].push_back(HOG());

            //掃過16個像素計算出HOG
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
                    //計算該像素所在的區間，並加權分配給下個區間(splitA、splitB)
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
                    result[hog_i][hog_j].data[field_index] += splitA;
                    result[hog_i][hog_j].data[(field_index + 1) % 4] += splitB;
                }
            }
            //畫出HOG直方圖
            if (show_hist) {
                auto axes = CvPlot::plot(std::vector<double>{ result[hog_i][hog_j].data[0], result[hog_i][hog_j].data[1], result[hog_i][hog_j].data[2], result[hog_i][hog_j].data[3] }, "-o");
                cv::imshow("histogram " + to_string(hog_i) + " " + to_string(hog_j), axes.render(300, 400));
            }
        }
    }
    return result;
}

int main() {
    cvui::init(WINDOW_NAME);
    cv::Mat frame = cv::Mat(cv::Size(550, 580), CV_32FC1);

    //cv::Mat origin = cv::imread("Lenna.jpg", CV_32F);

    //初始化所需圖片
    cv::Mat img = cv::imread("ABC01.jpg", CV_32FC1);
    cv::Mat target = cv::imread("t1.jpg", CV_32FC1);
    cv::Mat result_img = cv::imread("t1.jpg", CV_8UC1);
    cv::Mat identify_img = cv::imread("p1.jpg", CV_8UC1);
    cv::Mat img_x_sobel = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_32FC1);
    cv::Mat img_y_sobel = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_32FC1);
    cv::Mat img_magnitude;
    cv::Mat img_angle;

    //圖片預處理
    cv::resize(img, img, cv::Size(16, 16));
    cv::resize(target, target, cv::Size(128, 128));
    cv::resize(result_img, result_img, cv::Size(128, 128));
    
    //cv::threshold(identify_img, identify_img, 150, 255, cv::THRESH_BINARY);
    cv::threshold(identify_img, identify_img,150,255, cv::THRESH_BINARY_INV);//二值化
    cv::resize(identify_img, identify_img, cv::Size(300, 32));
    cv::cvtColor(result_img, result_img,cv::COLOR_GRAY2RGB);

    //////////////////////////<part 1>///////////////////////////////////////

    //將單個字母圖片的所有HOG算出，存入alpha_hogs
    vector<vector<HOG>> alpha_hogs = get_img_hogs(img,false);
    //儲存每個ROI錨點對應到的L2距離
    vector<pair<float, cv::Point>> all_result;
    //計算每個ROI與字母的L2距離，用雙層迴圈掃過所有16x16的區域
    for (int i = 0; i < target.rows-16; i++) {
        for (int j = 0; j < target.cols-16; j++) {
            //產生ROI(切割出16x16的區域)
            cv::Mat ROI = target(cv::Rect(j, i, 16, 16));
            //計算此ROI的所有HOG
            vector<vector<HOG>> ROI_hogs = get_img_hogs(ROI,false);
            //計算ROI與字母的L2距離
            float result = get_hogs_distance(alpha_hogs,ROI_hogs);
            //放入vector
            all_result.push_back(make_pair(result,cv::Point(i,j)));
        }
    }
    //排序取前六名
    std::sort(all_result.begin(), all_result.end(), [](auto& left, auto& right) {
        return left.first < right.first;
        });
    for (int num = 0; num < 6; num++) {
        int i_base = all_result[num].second.x;
        int j_base = all_result[num].second.y;
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                //藍綠通道值減半，使該區域變紅
                result_img.at < cv::Vec3b >(i_base + i, j_base + j)[0] /= 2;
                result_img.at < cv::Vec3b >(i_base + i, j_base + j)[1] /= 2;
            }
        }
    }
    //顯示結果
    cv::imshow("result_img", result_img);
    

    //////////////////////////<part 2>///////////////////////////////////////


    cv::imshow("identify_img", identify_img);
    //用陣列儲存每個字母的所有HOG
    vector<vector<HOG>> all_alpha_hogs[26];
    for (int i = 1; i <= 26; i++) {
        //產生圖片檔名(ABCXX.jpg)
        string zero = i < 10 ? "0" : "";
        string filename = "ABC" + zero + to_string(i) + ".jpg";
        cv::Mat alpha_img = cv::imread(filename, CV_32FC1);
        //圖片縮放
        cv::resize(alpha_img, alpha_img, cv::Size(16, 16));
        //產生HOG放入陣列
        all_alpha_hogs[i - 1] = get_img_hogs(alpha_img,false);
    }
    string output;
    cv::Mat labels;
    //產生連通圖，count為連通區域數量
    int count = cv::connectedComponents(identify_img, labels);
    cout << "count " << count << endl;
    //配置結果字串的大小(連通區域數量)
    output.resize(count);
    //存放每個連通區域與字母的L2距離最小值
    vector<float> min_result_components;
    //配置大小(連通區域數量)
    min_result_components.reserve(count);
    //初始值設為float可表達之最大值
    for (int i = 0; i < count; i++) { min_result_components[i]= std::numeric_limits<float>::max();}
    //雙層迴圈掃過每個像素，若遇到label，則產生ROI計算L2距離
    for (int i = 0; i < labels.rows-16; i++) {
        for (int j = 0; j < labels.cols-16; j++) {
            //為0時是空白處，不處理
            if (labels.at<ushort>(i, j) == 0) continue;
            //產生ROI
            cv::Mat ROI = identify_img(cv::Rect(j, i, 16, 16));
            //計算ROI的所有HOG
            vector<vector<HOG>> ROI_hogs = get_img_hogs(ROI,false);
            
            //ROI的HOG與A~Z每個字母的HOG計算L2距離
            for (int alpha = 0; alpha < 26; alpha++) {
                float result = get_hogs_distance(ROI_hogs, all_alpha_hogs[alpha]);
                //若該L2距離比之前遇過的都小，則取代為最小，並將字元放入輸出字串
                if (result < min_result_components[labels.at<ushort>(i, j)]) {
                    min_result_components[labels.at<ushort>(i, j)] = result;
                    output[labels.at<ushort>(i, j)] = alpha + 'A';
                }
            }
        }
    }
    //輸出結果
    cout << "output " << output << endl;
    while (true) {
        if (cv::waitKey(20) == 27) {
            break;
        }
    }
    return 0;
}