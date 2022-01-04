/***********************************************************************
*
*工程说明：该工程用于手指静脉图像边缘提取以及为边缘之外的部分打上mask
*依赖： opencv3.4.11+
*Date: 2020-9-15
*Author: gqc
*
************************************************************************/
#include <iostream>
#include <opencv2/opencv.hpp>
#include "common/FileSort.h"
#include "edge_detection.h"
#include <time.h>
#include <algorithm>

using namespace std;
using namespace cv;


string SAVE =  "Z:\\data\\finger_vein\\SCUT_LFMB\\SCUT LFMB-HQFV\\SegMask\\Dark_Session2\\";
// string LOAD =  "Z:\\data\\finger_vein\\SCUT_LFMB\\SCUT LFMB-HQFV\\Dark_Session2\\Pictures";
string LOAD =  "D:\\CameraSDK\\1.png";

void int2str(const int &int_temp, string &string_temp)
{
	stringstream stream;
	stream << int_temp;
	string_temp = stream.str();   //此处也可以用 stream>>string_temp  
}

/********************************************
 * 字符串分割
 *******************************************/
void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while(std::string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2-pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if(pos1 != s.length())
        v.push_back(s.substr(pos1));
}

/*********************************************
*查找字符c最后出现的位置
*
**********************************************/
int findchar(string str, unsigned char c)
{
	if (str.length() <= 0)
	{
		return 0;
	}

	for (int i = str.length()-1; i >= 0; i--)
	{
		if (str[i] == c)
		{
			return i+1;
		}
	}
	return 0;
}

// int main()
// {
//     printf("start capture");
//     VideoCapture capture(0);
//     Mat frame;
//     // capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);//宽度 
//     // capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);//高度
//     // capture.set(CV_CAP_PROP_FPS, 30);//帧数
//     int count_frame = 0;
//     string fileName;
//     while(1)
//     {
//         count_frame++;
//         capture >> frame;
//         fileName = to_string(count_frame) + ".bmp";
//         imwrite(fileName, frame);
//     }
// }


// int main()
// {
// 	vector <string> imgVector;
// 	findFile((char*)LOAD.c_str(), "jpg", imgVector); //遍历文件夹中的bmp格式图像
// 	vector<string>::iterator it;
// 	for (it = imgVector.begin(); it != imgVector.end(); it++)
// 	{
//             string tmpStr = *it;
//             Mat srcImg = imread(tmpStr.c_str(), 0);
//             //vector<string> stringVec;
//             //SplitString(tmpStr, stringVec,"/");
//             int pos = findchar(tmpStr, '\\');
//             string s_name = tmpStr.substr(pos, tmpStr.length() - pos);
//             cout << s_name<<endl;
//             //************预处理过程********************
//             if (!srcImg.empty())
//             {
//                 int gau_win_width = 25;   // 取奇数
//                 cv::Mat outImg = srcImg.colRange(min_y, max_y);
//                 cv::GaussianBlur(srcImg, srcImg, cv::Size(5, 5), 2, 2);

//                 vector<vector<int>> u_b_1 = edge_detect(srcImg, min_y, max_y, 1);
//                 vector<int> u_y = u_b_1[0];
//                 vector<int> b_y = u_b_1[1];
//                 addMask((Mat &) outImg,(vector<int> &) u_y, (vector<int> &) b_y);
//                 string save_name = SAVE + s_name;
//                 imwrite(save_name, outImg);   //保存ROI图像
//             }
// 	}
//     cout << "finish" << endl;
// 	return 0;
// }


int main()
{
    Mat srcImg = imread(LOAD.c_str(), 0);
    //vector<string> stringVec;
    //SplitString(tmpStr, stringVec,"/");
    imshow("22", srcImg);
    waitKey();
    //************预处理过程********************
    if (!srcImg.empty())
    {
        int gau_win_width = 25;   // 取奇数
        cv::Mat outImg = srcImg.colRange(min_y, max_y);
        cv::GaussianBlur(srcImg, srcImg, cv::Size(5, 5), 2, 2);
        int pos = findchar(LOAD, '\\');
        vector<vector<int>> u_b_1 = edge_detect(srcImg, min_y, max_y, 1);
        vector<int> u_y = u_b_1[0];
        vector<int> b_y = u_b_1[1];
        addMask((Mat &) outImg,(vector<int> &) u_y, (vector<int> &) b_y);
        imshow("11", outImg);
        waitKey();
        string s_name = LOAD.substr(pos, LOAD.length() - pos);
        cout << s_name<<endl;
        string save_name = SAVE + s_name;
        imwrite(save_name, outImg);   //保存ROI图像
    }
}
