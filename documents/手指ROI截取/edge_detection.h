//
// Created by xhb on 18-7-28.
//

#ifndef INC_3D_FINGER_VEIN_EDGE_DETECTION_H
#define INC_3D_FINGER_VEIN_EDGE_DETECTION_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <numeric>
#include <cmath>

using namespace std;
using namespace cv;
using namespace Eigen;

extern int min_y, max_y;
void addMask(cv::Mat &srcImage, vector<int> &u_y, vector<int> &b_y);
vector<vector<int>> find_all_edges(cv::Mat img_1, cv::Mat img_2, cv::Mat img_3);
vector<vector<int>> edge_detect(cv::Mat img, int min_x, int max_x, int image_index);
vector<vector<int>> find_contours(cv::Mat edge_img);
bool isuseful(vector<int> test_pts);
int find_pts(vector<int> arr, int win_width);
vector<int> erase_outlier(int u_idx, int win_width, vector<int> u_y);
void disp_detected_edges(vector<vector<int>> u_b_y, cv::Mat img_1, cv::Mat img_2, cv::Mat img_3);

#endif //INC_3D_FINGER_VEIN_EDGE_DETECTION_H
