

#ifndef READMNIST_H
#define READMNIST_H

//数据库内容读取

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
};// Mnist图像文件头

struct LabelFileHeader{
	unsigned char MagicNumber[4];
	unsigned char NumberOfItems[4];
};//Mnist标签文件头


class ReadMnist{

	char* filePathT;
	char* filePathL;

	int ConvertCharArrayToInt(unsigned char* array, int LengthOfArray);


public:
	vector< vector<unsigned char> >* dataT;//训练数据集指针
	vector< vector<unsigned char> >* dataL;//测试数据集指针

	//析构函数，赋值变量

	ReadMnist(){}

	//读取图像文件
	void readFile(vector< vector<unsigned char> >* _dataT, vector< vector<unsigned char> >* _dataL,
		char* _filePathT, char* _filePathL);

};	//end of ReadMnist




#endif
