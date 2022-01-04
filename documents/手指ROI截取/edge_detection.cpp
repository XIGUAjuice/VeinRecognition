//
// Created by xhb on 18-7-28.
//

#include "edge_detection.h"

//int min_y = 91;
//int max_y = 490;
int min_y = 360;
int max_y = 1060;

vector<vector<int>> find_all_edges(cv::Mat img_1, cv::Mat img_2, cv::Mat img_3)
{

    int bias = 0;
    vector<vector<int>> u_b_1 = edge_detect(img_1, min_y, max_y, 1);
    vector<int> u_y_1 = u_b_1[0];
    vector<int> b_y_1 = u_b_1[1];

    vector<vector<int>> u_b_2 = edge_detect(img_2, min_y, max_y, 2);
    vector<int> u_y_2 = u_b_2[0];
    vector<int> b_y_2 = u_b_2[1];

    vector<vector<int>> u_b_3 = edge_detect(img_3, min_y, max_y, 3);
    vector<int> u_y_3 = u_b_3[0];
    vector<int> b_y_3 = u_b_3[1];

    // 直接出来的边缘坐标数组长度是408，因为滤波时加上了边缘，现在再去掉那个边缘
    // int gau_win_width = 9;   // 取奇数
    // min_x-floor((float)gau_win_width / 2) : max_x+floor((float)gau_win_width / 2)+1   左闭右开
    vector<int> temp_u_1, temp_b_1;
    vector<int> temp_u_2, temp_b_2;
    vector<int> temp_u_3, temp_b_3;
    for(int i=4;i<=403;i++) {
        temp_u_1.push_back(u_y_1[i] + bias);
        temp_b_1.push_back(b_y_1[i] - bias);
        temp_u_2.push_back(u_y_2[i] + bias);
        temp_b_2.push_back(b_y_2[i] - bias);
        temp_u_3.push_back(u_y_3[i] + bias);
        temp_b_3.push_back(b_y_3[i] - bias);
    }

    vector<vector<int>> u_b_y;
    u_b_y.push_back(temp_u_1);
    u_b_y.push_back(temp_b_1);
    u_b_y.push_back(temp_u_2);
    u_b_y.push_back(temp_b_2);
    u_b_y.push_back(temp_u_3);
    u_b_y.push_back(temp_b_3);

    return u_b_y;
}

vector<vector<int>> edge_detect(cv::Mat img, int min_x, int max_x, int image_index)
{
    // 先根据指定区域确定ROI
    // 边缘提取宽度为9.，所以这里先左右扩大4
    int gau_win_width = 25;   // 取奇数
    cv::Mat sub_img = img.colRange(min_x, max_x);
//    cout << "min_x-floor((float)gau_win_width / 2): " << min_x-floor((float)gau_win_width / 2) << endl;
//    cout << "max_x+floor((float)gau_win_width / 2): " << max_x+floor((float)gau_win_width / 2) << endl;
    img = sub_img;

    cv::Mat kernel_up = (cv::Mat_<char>(5, 9) <<
            -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0,  0,  0,  0,  0,  0,  0,  0,  0,
            1,  1,  1,  1,  1,  1,  1,  1,  1,
            1,  1,  1,  1,  1,  1,  1,  1,  1);
    cv::Mat kernel_down = (cv::Mat_<char>(5, 9) <<
            1,  1,  1,  1,  1,  1,  1,  1,  1,
            1,  1,  1,  1,  1,  1,  1,  1,  1,
            0,  0,  0,  0,  0,  0,  0,  0,  0,
            -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1);
    cv::Mat dst, img_roi_up, img_roi_down, dst_roi_up, dst_roi_down;
    int rows = img.rows;
    int cols = img.cols;
    img_roi_up = img.rowRange(0, round(rows/2));
    img_roi_down = img.rowRange(round(rows/2), rows);
    dst.create(img.size(), img.type());
    dst_roi_up = dst.rowRange(0, round(rows/2));
    dst_roi_down = dst.rowRange(round(rows/2), rows);

    // 使用自定义模板进行边缘检测
    cv::filter2D(img_roi_up, dst_roi_up, img.depth(), kernel_up);
    cv::filter2D(img_roi_down, dst_roi_down, img.depth(), kernel_down);

    // 边缘检测测试
    cv::imshow("edge detection", dst);
//    string win_name;
//    int2str(image_index, win_name);
//    cv::imshow(win_name, dst);

    // 自定义核
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 9));
    // 腐蚀膨胀，开运算
    cv::dilate(dst, dst, kernel);
    cv::erode(dst, dst, kernel);

    // 转换成uint8类型
    cv::convertScaleAbs(dst, dst);
   cv::imshow("dilate erode", dst);
   cv::waitKey();

    // 提取轮廓
    // [ux, ~, bx, ~] = find_contours(edge_img);
    vector<vector<int>> u_b_edge = find_contours(dst);
    vector<int> u_x = u_b_edge[0];
    vector<int> u_y = u_b_edge[1];
    vector<int> b_x = u_b_edge[2];
    vector<int> b_y = u_b_edge[3];

    // u_idx = find_pts(ux, gau_win_width);
    // b_idx = find_pts(bx, gau_win_width);
//    int gau_win_width = 9;   // 取奇数
    int u_idx = find_pts(u_y, gau_win_width);
    int b_idx = find_pts(b_y, gau_win_width);
//    cout << image_index << " u_idx: " << u_idx << endl;
//    cout << image_index << " b_idx: " << b_idx << endl;
    if(u_idx == 0 || b_idx == 0)
        cout << "error" << endl;
    else
    {
        u_y = erase_outlier(u_idx, gau_win_width*2, u_y);
        b_y = erase_outlier(b_idx, gau_win_width*2, b_y);
    }

    vector<vector<int>> ux_bx;
    ux_bx.push_back(u_y);
    ux_bx.push_back(b_y);

    return ux_bx;
}

// [u_edge_x, u_edge_y, b_edge_x, b_edge_y] = find_contours(edge_img)
vector<vector<int>> find_contours(cv::Mat edge_img)
{
    int rows = edge_img.rows;
    int cols = edge_img.cols;
    int mid = (int)floor(rows / 2);
    //int mid = 240;
//    int thre = 80;
    int thre = 130;
    cv::imshow("222", edge_img);
    cv::waitKey();
    cv::Mat edge_img_roi_up, edge_img_roi_down;
    edge_img_roi_up = edge_img.rowRange(0, mid);
    edge_img_roi_down = edge_img.rowRange(mid, rows);
    cv::Mat dst_img, dst_roi_up, dst_roi_down;
    dst_img.create(edge_img.size(), edge_img.type());
    dst_roi_up = dst_img.rowRange(0, mid);
    dst_roi_down = dst_img.rowRange(mid, rows);

    // 检测上下边缘
    cv::threshold(edge_img_roi_up, dst_roi_up, thre, 255, THRESH_BINARY);
    cv::threshold(edge_img_roi_down, dst_roi_down, thre, 255, THRESH_BINARY);
    // 开运算去除孤岛
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));//保证是奇数
    cv::morphologyEx(dst_roi_up, dst_roi_up, MORPH_OPEN, kernel);
    cv::morphologyEx(dst_roi_down, dst_roi_down, MORPH_OPEN, kernel);
    cv::imshow("222", dst_roi_up);
    cv::imshow("333", dst_roi_down);
    cv::waitKey();
    // cv::adaptiveThreshold(edge_img_roi_up, dst_roi_up, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 7, 11);
    // cv::adaptiveThreshold(edge_img_roi_down, dst_roi_down, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 7, 11);

//    cv::Mat kernel = (cv::Mat_<float>(3, 3) << -1, -1, -1, 2, 5, 2, -1, -1, -1);
//    cv::filter2D(edge_img_roi_down, edge_img_roi_down, edge_img_roi_down.type(), kernel);

//    cv::imshow("threshold", dst_img);

//    Eigen::MatrixXf u_edge_x, u_edge_y, b_edge_x, b_edge_y;
//    u_edge_x = Eigen::MatrixXf::Zero(1, mid);
//    u_edge_y = Eigen::MatrixXf::Zero(1, mid);
//    b_edge_x = Eigen::MatrixXf::Zero(1, cols-mid);
//    b_edge_y = Eigen::MatrixXf::Zero(1, cols-mid);
    vector<int> u_edge_x, u_edge_y, b_edge_x, b_edge_y;
    // 预先给这几个数组分配好内存，后面直接通过下标访问
    for(int j=0;j<cols;j++)
    {
        u_edge_x.push_back(0);
        u_edge_y.push_back(0);
        b_edge_x.push_back(0);
        b_edge_y.push_back(0);
    }
    cv::Mat mask(edge_img.rows, edge_img.cols, edge_img.type(), cv::Scalar(0));

    int i, j;
    uchar max=0;
    // 上边缘
    for(j=0;j<cols;j++)
    {
        for(i=mid-1;i>=0;i--)
        {
            if(dst_img.at<uchar>(i, j) > thre)
            {
                u_edge_x[j] = j;
                u_edge_y[j] = i;

//                cout << j << endl;
                mask.at<uchar>(i, j) = 255;
                break;
            }

            if(dst_img.at<uchar>(i, j) > max)
            {
                max = dst_img.at<uchar>(i, j);
            }
        }

        if(i == 0)
        {
            for(i=mid-1;i>=0;i--)
            {
                if(dst_img.at<uchar>(i, j) == max)
                {
                    u_edge_x[j] = j;
                    u_edge_y[j] = i;

                    mask.at<uchar>(i, j) = 255;
                    break;
                }
            }
        }
    }

    // 下边缘
    max = 0;
    for(j=0;j<cols;j++)
    {
        for(i=mid;i<rows;i++)
        {
            if(dst_img.at<uchar>(i, j) > thre)
            {
                b_edge_x[j] = j;
                b_edge_y[j] = i;

                mask.at<uchar>(i, j) = 255;
                break;
            }

            if(dst_img.at<uchar>(i, j) > max)
            {
                max = dst_img.at<uchar>(i, j);
            }
        }

        if(i == rows-1)
        {
            for(i=mid;i<rows;i++)
            {
                if(dst_img.at<uchar>(i, j) == max)
                {
                    b_edge_x[j] = j;
                    b_edge_y[j] = i;

                    mask.at<uchar>(i, j) = 255;
                    break;
                }
            }
        }
    }


//    Eigen::MatrixXf u_b_edge = Eigen::MatrixXf::Zero(4, mid);
//    u_b_edge << u_edge_x, u_edge_y, b_edge_x, b_edge_y;
    vector<vector<int>> u_b_edge;
    u_b_edge.push_back(u_edge_x);
    u_b_edge.push_back(u_edge_y);
    u_b_edge.push_back(b_edge_x);
    u_b_edge.push_back(b_edge_y);

    cv::imshow("mask", mask);
    cv::waitKey();
    return u_b_edge;
}

bool isuseful(vector<int> test_pts)
{
//    int thre = 6;
    int thre = 6;
    int pt_size = test_pts.size();
    bool flag = true;
    vector<int> dx;
    for(int i=0;i<pt_size - 1;i++)
    {
        // dx[i] = test_pts[i+1] - test_pts[i];
        dx.push_back(test_pts[i+1] - test_pts[i]);
    }
    int sum_dx = accumulate(dx.begin(), dx.end(), 0);
    if(sum_dx > thre)
    {
        flag = false;
    }
    return flag;
}

int find_pts(vector<int> arr, int win_width)
{
    bool flag = false;
    int arr_len = arr.size();
    int start_index = arr_len / 2;
    int end_index, new_index;
    vector<int> test_pts;
    int pt;

    while(flag == false)
    {
        end_index = start_index + win_width;
        for(int i=start_index;i<=end_index;i++)
            test_pts.push_back(arr[i]);
            //test_pts[i-start_index] = arr[i];
        flag = isuseful(test_pts);
        if(flag == false)
        {
            new_index = start_index - win_width / 2;
            if(new_index < 0)
                break;
            start_index = new_index;
        }

    }

    if(flag == false)
    {
        start_index = arr_len / 2;
    }
    while(flag == false)
    {
        end_index = start_index + win_width;
        if(end_index > arr_len)
        {
            pt = 0;
            break;
        }
        for(int i=start_index;i<=end_index;i++)
            test_pts.push_back(arr[i]);
            //test_pts[i-start_index] = arr[i];
        flag = isuseful(test_pts);
        if(flag == false)
            start_index = start_index + win_width / 2;
    }

    if(flag == true)
        pt = start_index + win_width / 2;
    //cout << pt << endl;
    return pt;
}

vector<int> erase_outlier(int u_idx, int win_width, vector<int> u_y)
{

    int idx = ceil((float)win_width / 2);
    float u_dif, u_predict, tmp;
    // 左遍历
    for (int i = u_idx; i >= 1; i--)
    {
        u_dif = 0;
        tmp = 0;
        for(int j=1;j<=idx-1;j++)
        {
            tmp = u_y[i+idx+j] - u_y[i+j];
            if (abs(tmp) <= idx)
                u_dif += tmp / idx;
        }
        u_predict = u_y[i] + u_dif / (idx - 1);
        if(abs(u_y[i-1] - u_predict) > 5){
            // cout << "up" << endl;
            // cout << u_y[i-1] << endl;
            // cout << u_predict << endl;
            u_y[i-1] = round(u_predict);
        }
    }
    // 右遍历
    for (int i = u_idx; i <= u_y.size() - 1; i++)
    {
        u_dif = 0;
        tmp = 0;
        for(int j=1;j<=idx-1;j++)
        {
            tmp = u_y[i-win_width+j] - u_y[i-idx+j+1];
            if (abs(tmp) <= idx)
                u_dif += tmp / idx;
        }
        u_predict = u_y[i] + u_dif / (idx - 1);
        if(abs(u_y[i+1] - u_predict) > 5){
            // cout << "down" << endl;
            // cout << u_y[i+1] << endl;
            // cout << u_predict << endl;
            u_y[i+1] = round(u_predict);
        }
            
    }

    return u_y;
}

void disp_detected_edges(vector<vector<int>> u_b_y, cv::Mat img_1, cv::Mat img_2, cv::Mat img_3)
{
    vector<int> u_y_1 = u_b_y[0];
    vector<int> b_y_1 = u_b_y[1];
    vector<int> u_y_2 = u_b_y[2];
    vector<int> b_y_2 = u_b_y[3];
    vector<int> u_y_3 = u_b_y[4];
    vector<int> b_y_3 = u_b_y[5];

    int edge_len = u_y_1.size();
    cv::cvtColor(img_1, img_1, COLOR_GRAY2BGR);
    cv::cvtColor(img_2, img_2, COLOR_GRAY2BGR);
    cv::cvtColor(img_3, img_3, COLOR_GRAY2BGR);

    for(int j=0;j<edge_len;j++)
    {
//        cout << u_y_1[j] << endl;
        cv::circle(img_1, cv::Point(j+min_y-1, u_y_1[j]), 1, cv::Scalar(255, 0, 0), 2, 8);
        cv::circle(img_1, cv::Point(j+min_y-1, b_y_1[j]), 1, cv::Scalar(0, 255, 0), 2, 8);

        cv::circle(img_2, cv::Point(j+min_y-1, u_y_2[j]), 1, cv::Scalar(255, 0, 0), 2, 8);
        cv::circle(img_2, cv::Point(j+min_y-1, b_y_2[j]), 1, cv::Scalar(0, 255, 0), 2, 8);

        cv::circle(img_3, cv::Point(j+min_y-1, u_y_3[j]), 1, cv::Scalar(255, 0, 0), 2, 8);
        cv::circle(img_3, cv::Point(j+min_y-1, b_y_3[j]), 1, cv::Scalar(0, 255, 0), 2, 8);
    }

    cv::imshow("disp_img_1", img_1);
    cv::imshow("disp_img_2", img_2);
    cv::imshow("disp_img_3", img_3);
}

void addMask(cv::Mat &srcImage, vector<int> &u_y, vector<int> &b_y)
{
    int cols = srcImage.cols;
    int rows = srcImage.rows;
    int maxRow = (int)floor(rows / 2);
    int minRow = (int)floor(rows / 2);

    for(int i = 0; i < cols; i++)
    {
        for (int j = 0; j < u_y[i]; j++) {
            srcImage.at<uchar>(j,i) = 0;
        }
        if (minRow > u_y[i] && u_y[i] -10 > 30)
            minRow = u_y[i];

        for (int k = rows - 1; k > b_y[i]; k--) {
            srcImage.at<uchar>(k,i) = 0;
        }
        if (maxRow < b_y[i] && b_y[i] + 10 < rows - 30)
            maxRow = b_y[i];
    }
    // cout << minRow << endl;
    // cout << maxRow <<endl;
    srcImage = srcImage.rowRange(minRow - 10, maxRow + 10);
}


