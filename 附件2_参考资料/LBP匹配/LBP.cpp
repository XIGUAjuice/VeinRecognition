#include "LBP.h"
#include<iostream>
using namespace std;
#define NORM_WIDTH 128//240
#define NORM_HEIGHT 64//120

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
void SubImg(unsigned char* pSrc,unsigned char* pDst,int SrcRows,int RowsFirst,int Rowslength,int ColsFirst,int Colslength)
{
	int i,j;
	for (i=0;i<Rowslength;i++)
	{
		for (j=0;j<Colslength;j++)
		{
			*(pDst+i*Colslength+j)=*(pSrc+(i+RowsFirst)*SrcRows+j+ColsFirst);
		}
	}
}

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
void LBPofPixel(unsigned char* image,int imageRows,int pointRows,int pointCols,unsigned char* LBPdata,unsigned char* UniformFlag,int* Numof1,int* RotateNum, unsigned char Radius)
{
	//获取对应的像素值
	unsigned char data[9];
	data[0] = *(image+pointRows*imageRows+pointCols);
	data[1] = *(image+pointRows*imageRows+pointCols+Radius);
	data[2] = *(image+(pointRows+Radius)*imageRows+pointCols+Radius);
	data[3] = *(image+(pointRows+Radius)*imageRows+pointCols);
	data[4] = *(image+(pointRows+Radius)*imageRows+pointCols-Radius);
	data[5] = *(image+pointRows*imageRows+pointCols-Radius);
	data[6] = *(image+(pointRows-Radius)*imageRows+pointCols-Radius);
	data[7] = *(image+(pointRows-Radius)*imageRows+pointCols);
	data[8] = *(image+(pointRows-Radius)*imageRows+pointCols+Radius);
	//计算LBP值,同时标志该LBP值是不是Uniform型的
	unsigned char Flag = 0;
	int num = 0;
	*Numof1 = 0;
	//初始化LBPdata
	(*LBPdata) = 0;
	for(unsigned char i=8;i>=1;i--)
	{
		(*LBPdata) = (*LBPdata) << 1;
		if(data[i] >= data[0])//1
		{
			(*Numof1) += 1;
			(*LBPdata) = (*LBPdata) | 0x01;
			if(i==8)//记录第8位的状态
			{
				Flag = 1;
			}
			else
			{
				if(Flag == 0)
					num++;
				Flag = 1;
			}
		}
		else//0
		{
			if(i==8)//记录第16位的状态
			{
				Flag = 0;
			}
			else
			{
				if(Flag == 1)
					num++;
				Flag = 0;
			}
		}
	}
	//判断第1位和最后一位是否一样
	if(((((*LBPdata)&0x80)!=0) && (((*LBPdata)&0x01)==0))
		|| ((((*LBPdata)&0x80)==0) && (((*LBPdata)&0x01)!=0)))
		num++;
	if(num <= 2)
		*UniformFlag = 1;
	else
		*UniformFlag = 0;
	//求取旋转次数
	if(*UniformFlag == 0)
		return ;
	if((*Numof1)==0 || (*Numof1)==8)
		return;
	unsigned char temp = 0;
	for(unsigned char i=0;i<(*Numof1);i++)
	{
		temp = temp << 1;
		temp = temp | 0x01;
	}
	for(unsigned char i=0;i<8;++i)//循环旋转比较
	{
		if(temp == (*LBPdata))
		{
			*RotateNum = i;
			break;
		}

		//循环移位
		if( (temp & 0x80) == 0x00)//末位补0
		{
			temp = temp << 1;
			temp = temp & 0xFE;
		}
		else//末位补1
		{
			temp = temp << 1;
			temp = temp | 0x01;
		}
	
	}
}

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
void getLBPHistogram(unsigned char* SubImage,int SubImgRows,int SubImgCols,int* LBPHistogram, unsigned char Radius)
{
	for(unsigned char i=0;i<59;++i)//总共有58种uniform LBP
	{
		LBPHistogram[i]=0;
	}

	//统计uniform LBP
	unsigned char LBPData = 0;//LBP code
	unsigned char UniformFlag = 0;
	int Numof1 = 0;
	int RotateNum = 0;
	for(int i=Radius;i<SubImgRows-Radius;++i)
	{
		for(int j=Radius;j<SubImgCols-Radius;++j)
		{
			LBPofPixel(SubImage,SubImgCols,i,j,&LBPData,&UniformFlag,&Numof1,&RotateNum,Radius);
			if (UniformFlag == 0)//非uniform型不统计
			{
				LBPHistogram[58] += 1;
				continue;	
			}

			if(Numof1 == 0)
				LBPHistogram[0] += 1;
			else if(Numof1 == 8)
				LBPHistogram[57] += 1;
			else
			{
				LBPHistogram[8*(Numof1-1)+1+RotateNum] += 1;
			}
		}
	}

}

/***********************************************************************
*函数：getBlockHistograms
*功能：串接分块直方图。
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
void getBlockHistograms(unsigned char* image,int ImageRows,int ImageCols,int* BlockHistograms, unsigned char Radius, unsigned char BlockRows, unsigned char BlockCols)
{
	//保存分块情况
//	BlockSize = size;

	//int SubRows = static_cast<int>((ImageRows-2*Radius) / BlockRows);
	//int SubCols = static_cast<int>((ImageCols-2*Radius) / BlockCols);
	int SubRows = static_cast<int>(ImageRows  / BlockRows);
	int SubCols = static_cast<int>(ImageCols / BlockCols);
	int imageMax;

	for(int i=0;i<BlockRows;++i)
	{
		for(int j=0;j<BlockCols;++j)
		{
			//获取子图像
			imageMax=int(float((float)ImageRows*ImageCols/BlockRows/BlockCols) + 100);
			unsigned char* subImage=new unsigned char[imageMax];
//			SubImg(image,subImage,ImageCols,i*SubRows,SubRows+Radius*2,j*SubCols,SubCols+Radius*2
			SubImg(image, subImage, ImageCols, i*SubRows, SubRows, j*SubCols, SubCols );

			int LBP_Histogram[59];
			getLBPHistogram(subImage,SubRows,SubCols,LBP_Histogram,Radius);

			for (int k=0;k<59;k++)
			{
				BlockHistograms[k+(i*BlockCols+j)*59]=LBP_Histogram[k];
			}

			delete subImage;
		}
	}
}

