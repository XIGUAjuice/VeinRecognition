#ifndef LBP_H
#define LBP_H
/***********************************************************************
*������SubImg
*���ܣ���ͼ���зָ��ָ����С��
*������
	pSrc��ԭʼͼ��ָ�룻
	pDst���ֿ�ͼ��ָ�룻
	SrcRows��ԭʼͼ���
	RowsFirst���ֿ����Ͻǵ���ԭͼ�е�������
	Rowslength���ֿ�ߣ�
	ColsFirst���ֿ����Ͻǵ���ԭͼ�е�������
	Colslength���ֿ��
*����ֵ����
***********************************************************************/
void SubImg(unsigned char* pSrc,unsigned char* pDst,int SrcRows,int RowsFirst,int Rowslength,int ColsFirst,int Colslength);

/***********************************************************************
*������LBPofPixel
*���ܣ�����ÿһ�ֿ���ĳһ���ص��LBP code��
*������
	image��ԭʼͼ��
	imageRows��ԭʼͼ��Ŀ�
	pointRows��ָ�����ص����ڵ�������
	pointCols��ָ�����ص����ڵ�������
	LBPdata���õ���LBP code��
	UniformFlag����uniform�ͱ�ʶ��
	Numof1��������8�����ڴ��ڸõ�����صĸ�����ͳ��0��57�������������
	RotateNum���任Ϊ��Сֵ��Ҫ����ת������
	Radius��LBP�뾶
*����ֵ����
***********************************************************************/
void LBPofPixel(unsigned char* image,int imageRows,int pointRows,int pointCols,unsigned char* LBPdata,unsigned char* UniformFlag,int* Numof1,int* RotateNum, unsigned char Radius);

/***********************************************************************
*������getLBPHistogram
*���ܣ�ͳ��ÿһ�ֿ��ֱ��ͼ��
*������
	SubImage���ֿ���ͼ��
	SubImgRows����ͼ��ߣ�
	SubImgCols����ͼ���
	LBPHistogram���õ��ֿ��ֱ��ͼ��
	Radius��LBP�뾶��
*����ֵ����
***********************************************************************/
void getLBPHistogram(unsigned char* SubImage,int SubImgRows,int SubImgCols,int* LBPHistogram, unsigned char Radius);

/***********************************************************************
*������getBlockHistograms
*���ܣ���ȡ�ֿ�ֱ��ͼ��
*������
	image����ƥ��ֱ��ͼ1��
	ImageRows��ͼ��ߣ�
	ImageCols��ͼ���
	BlockHistograms���õ�ֱ��ͼ��
	Radius��LBP�뾶��
	BlockRows���зֿ飻
	BlockCols���зֿ顣
*����ֵ����
***********************************************************************/
void getBlockHistograms(unsigned char* image,int ImageRows,int ImageCols,int* BlockHistograms, unsigned char Radius, unsigned char BlockRows, unsigned char BlockCols);


#endif