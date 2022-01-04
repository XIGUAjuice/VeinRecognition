#ifndef _MATCHLBP_H_
#define _MATCHLBP_H_
#include <opencv2/opencv.hpp>
using namespace cv;
/**********************************************************************************
*函数：matchBlockHistograms
*功能：直方图匹配。
*Input
	BlockHistograms1：待匹配直方图1；
	BlockHistograms2：待匹配直方图2；
	BlockRows：行分块；
	BlockCols：列分块。
*Output
	Distance：匹配分数值。
*********************************************************************************/
double matchBlockHistograms(int* BlockHistograms1, int* BlockHistograms2, unsigned char BlockRows, unsigned char BlockCols);

/**********************************************************************************
*函数:matchImage
*功能：Image 图像大小要归一化，然后经过高斯模糊处理以便去除噪声，实验中选用的是大小为5*5、标准差为1.2的高斯核连续处理图像四次
*Input
	Image1：待匹配图像1；
	Rows1：图像1的高；
	Cols1：图像1的宽；
	Image2：待匹配图像2；
	Rows2：图像2的高；
	Cols2：图像2的宽；
	Radius：LBP半径；
	BlockRows：行分块；
	BlockCols：列分块。
*Output
	Distance：匹配分数值
*********************************************************************************/
double matchImage(unsigned char* Image1, int Rows1, int Cols1, unsigned char* Image2, int Rows2, int Cols2, unsigned char Radius, unsigned char BlockRows, unsigned char BlockCols);

double matchLBP(Mat srcImg, Mat dstImg);
#endif // !_MATCHLBP_H_
