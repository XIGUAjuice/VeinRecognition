#include "LBP.h"
#include<iostream>
using namespace std;
#define NORM_WIDTH 128//240
#define NORM_HEIGHT 64//120

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
void LBPofPixel(unsigned char* image,int imageRows,int pointRows,int pointCols,unsigned char* LBPdata,unsigned char* UniformFlag,int* Numof1,int* RotateNum, unsigned char Radius)
{
	//��ȡ��Ӧ������ֵ
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
	//����LBPֵ,ͬʱ��־��LBPֵ�ǲ���Uniform�͵�
	unsigned char Flag = 0;
	int num = 0;
	*Numof1 = 0;
	//��ʼ��LBPdata
	(*LBPdata) = 0;
	for(unsigned char i=8;i>=1;i--)
	{
		(*LBPdata) = (*LBPdata) << 1;
		if(data[i] >= data[0])//1
		{
			(*Numof1) += 1;
			(*LBPdata) = (*LBPdata) | 0x01;
			if(i==8)//��¼��8λ��״̬
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
			if(i==8)//��¼��16λ��״̬
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
	//�жϵ�1λ�����һλ�Ƿ�һ��
	if(((((*LBPdata)&0x80)!=0) && (((*LBPdata)&0x01)==0))
		|| ((((*LBPdata)&0x80)==0) && (((*LBPdata)&0x01)!=0)))
		num++;
	if(num <= 2)
		*UniformFlag = 1;
	else
		*UniformFlag = 0;
	//��ȡ��ת����
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
	for(unsigned char i=0;i<8;++i)//ѭ����ת�Ƚ�
	{
		if(temp == (*LBPdata))
		{
			*RotateNum = i;
			break;
		}

		//ѭ����λ
		if( (temp & 0x80) == 0x00)//ĩλ��0
		{
			temp = temp << 1;
			temp = temp & 0xFE;
		}
		else//ĩλ��1
		{
			temp = temp << 1;
			temp = temp | 0x01;
		}
	
	}
}

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
void getLBPHistogram(unsigned char* SubImage,int SubImgRows,int SubImgCols,int* LBPHistogram, unsigned char Radius)
{
	for(unsigned char i=0;i<59;++i)//�ܹ���58��uniform LBP
	{
		LBPHistogram[i]=0;
	}

	//ͳ��uniform LBP
	unsigned char LBPData = 0;//LBP code
	unsigned char UniformFlag = 0;
	int Numof1 = 0;
	int RotateNum = 0;
	for(int i=Radius;i<SubImgRows-Radius;++i)
	{
		for(int j=Radius;j<SubImgCols-Radius;++j)
		{
			LBPofPixel(SubImage,SubImgCols,i,j,&LBPData,&UniformFlag,&Numof1,&RotateNum,Radius);
			if (UniformFlag == 0)//��uniform�Ͳ�ͳ��
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
*������getBlockHistograms
*���ܣ����ӷֿ�ֱ��ͼ��
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
void getBlockHistograms(unsigned char* image,int ImageRows,int ImageCols,int* BlockHistograms, unsigned char Radius, unsigned char BlockRows, unsigned char BlockCols)
{
	//����ֿ����
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
			//��ȡ��ͼ��
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

