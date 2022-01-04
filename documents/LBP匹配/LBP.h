#ifndef LBP_H
#define LBP_H
/***********************************************************************
*函数：SubImg
*功能：从图像中分割出指定的小块
*参数：
	pSrc：原始图像指针；
	pDst：分块图像指针；
	SrcRows：原始图像宽；
	RowsFirst：分块左上角点在原图中的行数；
	Rowslength：分块高；
	ColsFirst：分块左上角点在原图中的列数；
	Colslength：分块宽。
*返回值：空
***********************************************************************/
void SubImg(unsigned char* pSrc,unsigned char* pDst,int SrcRows,int RowsFirst,int Rowslength,int ColsFirst,int Colslength);

/***********************************************************************
*函数：LBPofPixel
*功能：计算每一分块内某一像素点的LBP code。
*参数：
	image：原始图像；
	imageRows：原始图像的宽；
	pointRows：指定像素点所在的行数；
	pointCols：指定像素点所在的列数；
	LBPdata：得到的LBP code；
	UniformFlag：非uniform型标识；
	Numof1：该像素8邻域内大于该点的像素的个数（统计0和57这两种情况）；
	RotateNum：变换为最小值需要的旋转次数；
	Radius：LBP半径
*返回值：空
***********************************************************************/
void LBPofPixel(unsigned char* image,int imageRows,int pointRows,int pointCols,unsigned char* LBPdata,unsigned char* UniformFlag,int* Numof1,int* RotateNum, unsigned char Radius);

/***********************************************************************
*函数：getLBPHistogram
*功能：统计每一分块的直方图。
*参数：
	SubImage：分块子图像；
	SubImgRows：子图像高；
	SubImgCols：子图像宽；
	LBPHistogram：得到分块的直方图；
	Radius：LBP半径。
*返回值：空
***********************************************************************/
void getLBPHistogram(unsigned char* SubImage,int SubImgRows,int SubImgCols,int* LBPHistogram, unsigned char Radius);

/***********************************************************************
*函数：getBlockHistograms
*功能：获取分块直方图。
*参数：
	image：待匹配直方图1；
	ImageRows：图像高；
	ImageCols：图像宽；
	BlockHistograms：得到直方图；
	Radius：LBP半径；
	BlockRows：行分块；
	BlockCols：列分块。
*返回值：空
***********************************************************************/
void getBlockHistograms(unsigned char* image,int ImageRows,int ImageCols,int* BlockHistograms, unsigned char Radius, unsigned char BlockRows, unsigned char BlockCols);


#endif