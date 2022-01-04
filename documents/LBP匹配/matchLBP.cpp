#include "matchLBP.h"
#include "LBP.h"
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

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
double matchBlockHistograms(int* BlockHistograms1, int* BlockHistograms2, unsigned char BlockRows, unsigned char BlockCols)
{
	double chiVal = 0;
	double histVal1, histVal2;
	int histSize = BlockRows*BlockCols * 59;
	int cnt = 0;		//匹配计数

	for (int i = 0; i < histSize; i++)
	{
		histVal1 = double(BlockHistograms1[i]);
		histVal2 = double(BlockHistograms2[i]);
		//直方图相交方法  相交率
		if (histVal1 == 0 && histVal2 == 0)
		{
			continue;
		}
		else
		{
			cnt++;
		}
			
		if (histVal1 <= histVal2)
			chiVal += histVal1 * 2 / (histVal1 + histVal2);
		else
			chiVal += histVal2 * 2 / (histVal1 + histVal2);

		//卡方距离统计方法
		/*if(histVal1 ==0 && histVal2 ==0)
			continue;
		double diffScore = abs(histVal1 - histVal2);
		chiVal += (diffScore * diffScore)/ (histVal1 + histVal2);*/
	}
	return chiVal / cnt;
}

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
double matchImage(unsigned char* Image1, int Rows1, int Cols1, unsigned char* Image2, int Rows2, int Cols2, unsigned char Radius, unsigned char BlockRows, unsigned char BlockCols)
{
	int* BlockHistograms1 = new int[BlockRows*BlockCols * 59];
	int* BlockHistograms2 = new int[BlockRows*BlockCols * 59];

	getBlockHistograms(Image1, Rows1, Cols1, BlockHistograms1, Radius, BlockRows, BlockCols);
	getBlockHistograms(Image2, Rows2, Cols2, BlockHistograms2, Radius, BlockRows, BlockCols);

	double Distance = 0;
	Distance = matchBlockHistograms(BlockHistograms1, BlockHistograms2, BlockRows, BlockCols);

	delete BlockHistograms1;
	delete BlockHistograms2;

	return Distance;
}


double matchLBP(Mat srcImg, Mat dstImg)
{
	int normWidth = 128;
	int normHeight = 64;
	int numCols = 6;
	int numRows = 3;

	//int normWidth = 251;
	//int normHeight = 152;
	//int numCols = 10;
	//int numRows = 6;

	int radius = 1;
	resize(dstImg, dstImg, Size(normWidth, normHeight));
	resize(srcImg, srcImg, Size(normWidth, normHeight));

	GaussianBlur(dstImg, dstImg, Size(5, 5), 2.0, 2.0);
	GaussianBlur(srcImg, srcImg, Size(5, 5), 2.0, 2.0);
	
	double Score = matchImage(srcImg.data, srcImg.rows, srcImg.cols, dstImg.data, dstImg.rows, dstImg.cols, radius, numRows, numCols);
	return Score;
}


