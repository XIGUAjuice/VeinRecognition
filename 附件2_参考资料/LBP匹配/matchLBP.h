#ifndef _MATCHLBP_H_
#define _MATCHLBP_H_
#include <opencv2/opencv.hpp>
using namespace cv;
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
double matchBlockHistograms(int* BlockHistograms1, int* BlockHistograms2, unsigned char BlockRows, unsigned char BlockCols);

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
double matchImage(unsigned char* Image1, int Rows1, int Cols1, unsigned char* Image2, int Rows2, int Cols2, unsigned char Radius, unsigned char BlockRows, unsigned char BlockCols);

double matchLBP(Mat srcImg, Mat dstImg);
#endif // !_MATCHLBP_H_
