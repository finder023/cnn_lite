

#ifndef READMNIST_H
#define READMNIST_H

//���ݿ����ݶ�ȡ

#include <iostream>
#include <fstream>
#include <vector>
#include "cnn.h"
#include "convertImg.h"

using namespace std;


struct ImageFileHeader{
	unsigned char MagicNumber[4];
	unsigned char NumberOfImages[4];
	unsigned char NumberOfRows[4];
	unsigned char NumberOfColums[4];
};// Mnistͼ���ļ�ͷ

struct LabelFileHeader{
	unsigned char MagicNumber[4];
	unsigned char NumberOfItems[4];
};//Mnist��ǩ�ļ�ͷ


class ReadMnist{

	char* filePathT;
	char* filePathL;

	int ConvertCharArrayToInt(unsigned char* array, int LengthOfArray);


public:
	vector< vector<unsigned char> >* dataT;//ѵ�����ݼ�ָ��
	vector< vector<unsigned char> >* dataL;//�������ݼ�ָ��

	//������������ֵ����

	ReadMnist(){}

	//��ȡͼ���ļ�
	void readFile(vector< vector<unsigned char> >* _dataT, vector< vector<unsigned char> >* _dataL,
		char* _filePathT, char* _filePathL);

};	//end of ReadMnist




#endif
