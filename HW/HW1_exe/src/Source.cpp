#include <iostream>
#include <string>
#include <random>
#include <fstream>
#include <algorithm>
#include <time.h>
#include <chrono>
#include <limits>
#include <opencv2/opencv.hpp>
#define CVPLOT_HEADER_ONLY
#include <CvPlot/cvplot.h>
#define CVUI_IMPLEMENTATION
#include "cvui.h"

#define WINDOW_NAME "B10815057_���t�q_�m��v���B�z_�@�~1"

//�������T�d�򤺩Ҧ��ȹ������T�Ȫ�table (��Ӫ�)
void get_gaussian_noise_table(double* rand_mapping_table, int gaussian_range, double gaussian_sigma) {    
    double y_sum = 0.0f;
    double cumulative = 0.0f;
    //���ͨC�ӭȪ��������T�A�å[�`�_��
    for (int x = -gaussian_range, index = 0; x <= gaussian_range; x++, index++) {
        double y = exp(-(double)(x * x) / (2.0f * gaussian_sigma * gaussian_sigma));
        y_sum += y;
        rand_mapping_table[index] = y;
    }
    //���]�L�@�M��X���T�`�ȡA�M��~��o����v
    //��X�C�Ӱ������T�����v�A�M��N�֥[�᪺���v�@����ӭ�
    for (int x = -gaussian_range, index = 0; x <= gaussian_range; x++, index++) {
        rand_mapping_table[index] /= y_sum;
        cumulative += rand_mapping_table[index];
        rand_mapping_table[index] = cumulative;
    }
}

//�ǤJ�@�H���Ȩîھڹ�Ӫ�^�����T��
int gaussian_noise(double* rand_mapping_table, int gaussian_range, double random_val) {
    int index = 0;
    int x;
    //���Ĥ@�Ӥj��ۤv�����v�ȡA�^�Ǹ����T��
    for (x = -gaussian_range; x <= gaussian_range; x++, index++) {
        if (rand_mapping_table[index] > random_val) break;
    }
    if (x > gaussian_range)x = gaussian_range;
    return x;
}

constexpr int RAND_DOUBLE_MIN = 0;
constexpr int RAND_DOUBLE_MAX = 1;
std::random_device rd;
std::default_random_engine eng(rd());
std::uniform_real_distribution<float> distr(RAND_DOUBLE_MIN, RAND_DOUBLE_MAX);

double random01() {
    return distr(eng);
}

int main() {
    srand(time(NULL));
    std::locale loc = std::locale::global(std::locale(""));
    cvui::init(WINDOW_NAME);
    cv::Mat frame = cv::Mat(cv::Size(1490, 520), CV_8UC1); //�m���CV_8UC3
    std::string filename = "ntust_gray.jpg";
    cv::Mat source_img = cv::imread(filename, 0);
    cv::Mat destination_img = cv::imread(filename, 0);
    bool have_noise = false;
    double gaussian_sigma = 25.0f;
    double pre_gaussian_sigma = gaussian_sigma;
    int gaussian_range = 100;
    int pre_gaussian_range = gaussian_range;
    double* rand_mapping_table = new double[gaussian_range * 2 + 1];
    double salt_pepper_N_percent = 0.1f;
    int filter_height = 3;
    int filter_width = 3;
    std::vector<cv::Point2d> transformation_points{ { 0, 0 },  { 1,1 } };
    auto axes = CvPlot::plot(transformation_points, "-o");
    cv::Mat test = axes.render(300, 400);
    cv::imshow(u8"Right click to reset", test);
    CvPlot::Window windowImage(u8"Right click to reset", axes, 300, 400);
    std::chrono::steady_clock::time_point pre_click = std::chrono::steady_clock::now();
    windowImage.setMouseEventHandler([&](const CvPlot::MouseEvent& mouseEvent) {
        std::chrono::steady_clock::time_point now_time = std::chrono::steady_clock::now();
        if ((mouseEvent.flags() & cv::MouseEventFlags::EVENT_FLAG_LBUTTON) && (std::chrono::duration_cast<std::chrono::milliseconds>(now_time - pre_click).count() > 100)) {
            if (mouseEvent.pos().x > 1.0f || mouseEvent.pos().x < 0.0f || mouseEvent.pos().y > 1.0f || mouseEvent.pos().y < 0.0f) return false;
            double x = mouseEvent.pos().x;
            double y = mouseEvent.pos().y;
            for (int i = 1; i < transformation_points.size(); i++) {
                if (transformation_points[i].x > x) {
                    // 1 / 255 ~= 0.004
                    if (abs(transformation_points[i].x - x) < 0.004f || abs(transformation_points[i - 1].x - x) < 0.004f) return false;
                    transformation_points.insert(transformation_points.begin() + i, cv::Point2d(x, y));
                    break;
                }
            }
            pre_click = now_time;
            axes = CvPlot::plot(transformation_points, "-o");
            return true;
        }
        if (mouseEvent.flags() & cv::MouseEventFlags::EVENT_FLAG_RBUTTON) {
            transformation_points.clear();
            transformation_points.push_back(cv::Point2d(0, 0));
            transformation_points.push_back(cv::Point2d(1, 1));
            axes = CvPlot::plot(transformation_points, "-o");
            return true;
        }
        return false;
        });
    get_gaussian_noise_table(rand_mapping_table, gaussian_range, gaussian_sigma);
    while (cv::getWindowProperty(WINDOW_NAME, 0) >= 0) {
        frame = cv::Scalar(49, 52, 49);
        cvui::printf(frame, 10, 10, "Source Image");
        if (!source_img.empty()) cvui::image(frame, 10, 30, source_img);
        if (!have_noise) {
            if (cvui::button(frame, 670, 170, 150, 30, "Gaussian Noise")) {
                //���h�j�鱽�L�Ҧ�pixel
                for (int i = 0; i < source_img.rows; i++) {
                    for (int j = 0; j < source_img.cols; j++) {
                        //���o���T��
                        int noise = gaussian_noise(rand_mapping_table, gaussian_range, random01());
                        int base_color = source_img.at<uchar>(i, j);
                        int result = base_color + noise;
                        //��W�L255�Τp��0���Ȱ��ҥ~�B�z
                        if (result > 255) result = 255;
                        else if (result < 0) result = 0;
                        //��[�J���T�᪺�ȩ��ؼйϤ���pixel
                        destination_img.at<uchar>(i, j) = result;
                    }
                }
                have_noise = true;
            }
            cvui::printf(frame, 670, 30, "Gaussian Range");
            cvui::counter(frame, 700, 60, &gaussian_range);
            if (gaussian_range > 255) { gaussian_range = 255; }
            else if (gaussian_range < 0) { gaussian_range = 0; }
            cvui::printf(frame, 670, 100, "Gaussian Sigma");
            cvui::trackbar(frame, 670, 120, 150, &gaussian_sigma, (double)0.1f, (double)100.0f);
            cvui::printf(frame, 670, 240, "N percent noise");
            cvui::trackbar(frame, 670, 260, 150, &salt_pepper_N_percent, (double)0.0f, (double)1.0f);
            if (cvui::button(frame, 670, 310, 150, 30, "Salt Pepper Noise")) {
                //���h�j�鱽�L�Ҧ�pixel
                for (int i = 0; i < source_img.rows; i++) {
                    for (int j = 0; j < source_img.cols; j++) {
                        //�Y0~1�H���Ȥj����w��N�A���ݧ���
                        //�Y0~1�H���Ȥp����w��N�A�H�U50%���v��ܶ�(0)�Υ�(255)
                        destination_img.at<uchar>(i, j) = random01() > salt_pepper_N_percent ? source_img.at<uchar>(i, j) : (random01() > 0.5f ? 255 : 0);
                    }
                }
                have_noise = true;
            }
        }
        else {
            if (cvui::button(frame, 670, 170, 150, 30, "Mean Filter")) {
                cv::Mat new_mat = cv::Mat::zeros(destination_img.rows, destination_img.cols, CV_8UC1);
                //���h�j�鱽�L�Ҧ�pixel
                for (int i = 0; i < destination_img.rows; i++) {
                    for (int j = 0; j < destination_img.cols; j++) {
                        int total = 0;
                        //�]�L��pixel����3x3�o��ϰ쪺�Ҧ�pixel
                        for (int k = 0; k < filter_height; k++) {
                            for (int g = 0; g < filter_width; g++) {
                                int x = j + g;
                                int y = i + k;
                                //�]�X�Ϥ��~����m���蹳�B�z
                                if (filter_height % 2) y -= (int)(filter_height / 2);
                                else y -= (int)(filter_height / 2) - 1;
                                if (filter_width % 2) x -= (int)(filter_width / 2);
                                else x -= (int)(filter_width / 2) - 1;

                                if (y < 0) y *= -1;
                                else if (y >= destination_img.rows) y = 2 * (destination_img.rows - 1) - y;
                                if (x < 0) x *= -1;
                                else if (x >= destination_img.cols) x = 2 * (destination_img.cols - 1) - x;

                                //�[�`�o�褤�Ҧ�pixel
                                total += destination_img.at<uchar>(y, x);
                            }
                        }
                        //�N�o�褤�Ҧ�pixel���Ȩ������A���ؼйϤ���pixel
                        new_mat.at<uchar>(i, j) = total / (filter_width * filter_height);
                    }
                }
                new_mat.copyTo(destination_img);
            }
            if (cvui::button(frame, 670, 310, 150, 30, "Median Filter")) {
                cv::Mat new_mat = cv::Mat::zeros(destination_img.rows, destination_img.cols, CV_8UC1);
                //�t�m�i�H�s���o�褤�Ҧ�pixel�Ȫ��Ŷ�
                int* all_val = new int[filter_height * filter_width];
                //���h�j�鱽�L�Ҧ�pixel
                for (int i = 0; i < destination_img.rows; i++) {
                    for (int j = 0; j < destination_img.cols; j++) {
                        int index = 0;
                        //�]�L��pixel����3x3�o��ϰ쪺�Ҧ�pixel
                        for (int k = 0; k < filter_height; k++) {
                            for (int g = 0; g < filter_width; g++) {
                                int x = j + g;
                                int y = i + k;
                                //�]�X�Ϥ��~����m���蹳�B�z
                                if (filter_height % 2) y -= (int)(filter_height / 2);
                                else y -= (int)(filter_height / 2) - 1;
                                if (filter_width % 2) x -= (int)(filter_width / 2);
                                else x -= (int)(filter_width / 2) - 1;

                                if (y < 0) y *= -1;
                                else if (y >= destination_img.rows) y = 2 * (destination_img.rows - 1) - y;
                                if (x < 0) x *= -1;
                                else if (x >= destination_img.cols) x = 2 * (destination_img.cols - 1) - x;

                                //�N�ȩ�J�s���o�誺�Ŷ�
                                all_val[index++] = destination_img.at<uchar>(y, x);

                            }
                        }
                        //�Ƨ��褤�Ҧ�pixel��
                        std::sort(all_val, all_val + (filter_height * filter_width));
                        //�N����ƪ�pixel�ȡA���ؼйϤ���pixel
                        new_mat.at<uchar>(i, j) = all_val[filter_height * filter_width / 2];
                    }
                }
                //����s���o��Ŷ����O����
                delete[] all_val;
                new_mat.copyTo(destination_img);
            }
        }
        if (cvui::button(frame, 670, 390, 150, 30, "Transformation")) {
            //���h�j�鱽�L�Ҧ�pixel
            for (int i = 0; i < source_img.rows; i++) {
                for (int j = 0; j < source_img.cols; j++) {
                    //�ҥ~�B�z�A��J��0��255(�̤j�P�̤p)�A��X�Y����J
                    if (source_img.at<uchar>(i, j) == 0 || source_img.at<uchar>(i, j) == 255) {
                        destination_img.at<uchar>(i, j) = source_img.at<uchar>(i, j);
                        continue;
                    }
                    //�p��pixel�ȹ���0~1�϶���x��
                    double input_x = (double)source_img.at<uchar>(i, j) / 255.0f;
                    double output_y = 0.0f;
                    //�]�L�C��transformation points
                    for (int k = 1; k < transformation_points.size(); k++) {
                        //����J��Ĥ@�Ӥj���Jx�Ȫ��I
                        if (input_x <= transformation_points[k].x) {
                            //���I�P�W���I��x�ȶZ��
                            double x_offset = transformation_points[k].x - transformation_points[k - 1].x;
                            //���I�P�W���I��y�ȶZ��
                            double y_offset = transformation_points[k].y - transformation_points[k - 1].y;
                            //�ҥ~�B�z�A�p�Gx�ȶZ���L�p�A�ӽu�q�������A�קK���͵L���j�A�ҥH�����I�P�W���I��y�ȥ���
                            if (x_offset < std::numeric_limits<double>::min()) {
                                output_y = (transformation_points[k].y + transformation_points[k - 1].y) / 2.0f;
                                break;
                            }
                            //�νu�ʤ�����X��Xy�ȡA�ø��X�j��
                            output_y = transformation_points[k - 1].y + y_offset * ((input_x - transformation_points[k - 1].x) / x_offset);
                            break;
                        }
                    }
                    //�N��Xy�ȭ��W255�o���Xpixel�ȡA���ؼйϤ���pixel
                    destination_img.at<uchar>(i, j) = output_y * 255.0f;
                }
            }
        }
        if (cvui::button(frame, 670, 480, 150, 30, "Reset")) {
            source_img.copyTo(destination_img);
            have_noise = false;
        }
        cvui::printf(frame, 840, 10, "Destination Image");
        if (!destination_img.empty()) cvui::image(frame, 840, 30, destination_img);

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
        cv::imshow(WINDOW_NAME, frame);
        if (cv::waitKey(20) == 27) {
            break;
        }
    }
    return 0;
}