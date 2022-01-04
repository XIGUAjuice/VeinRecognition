#include "matchLBP.h"
#include "LBP.h"
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

/**********************************************************************************
*������matchBlockHistograms
*���ܣ�ֱ��ͼƥ�䡣
*Input
	BlockHistograms1����ƥ��ֱ��ͼ1��
	BlockHistograms2����ƥ��ֱ��ͼ2��
	BlockRows���зֿ飻
	BlockCols���зֿ顣
*Output
	Distance��ƥ�����ֵ��
*********************************************************************************/
double matchBlockHistograms(int* BlockHistograms1, int* BlockHistograms2, unsigned char BlockRows, unsigned char BlockCols)
{
	double chiVal = 0;
	double histVal1, histVal2;
	int histSize = BlockRows*BlockCols * 59;
	int cnt = 0;		//ƥ�����

	for (int i = 0; i < histSize; i++)
	{
		histVal1 = double(BlockHistograms1[i]);
		histVal2 = double(BlockHistograms2[i]);
		//ֱ��ͼ�ཻ����  �ཻ��
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

		//��������ͳ�Ʒ���
		/*if(histVal1 ==0 && histVal2 ==0)
			continue;
		double diffScore = abs(histVal1 - histVal2);
		chiVal += (diffScore * diffScore)/ (histVal1 + histVal2);*/
	}
	return chiVal / cnt;
}

/**********************************************************************************
*����:matchImage
*���ܣ�Image ͼ���СҪ��һ����Ȼ�󾭹���˹ģ�������Ա�ȥ��������ʵ����ѡ�õ��Ǵ�СΪ5*5����׼��Ϊ1.2�ĸ�˹����������ͼ���Ĵ�
*Input
	Image1����ƥ��ͼ��1��
	Rows1��ͼ��1�ĸߣ�
	Cols1��ͼ��1�Ŀ�
	Image2����ƥ��ͼ��2��
	Rows2��ͼ��2�ĸߣ�
	Cols2��ͼ��2�Ŀ�
	Radius��LBP�뾶��
	BlockRows���зֿ飻
	BlockCols���зֿ顣
*Output
	Distance��ƥ�����ֵ
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


