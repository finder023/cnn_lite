
#include "readMnist.h"

using namespace std;

inline int ReadMnist::ConvertCharArrayToInt(unsigned char* array, int LengthOfArray)
{
	if (LengthOfArray < 0)
	{
		return -1;
	}
	int result = static_cast<signed int>(array[0]);
	for (int i = 1; i < LengthOfArray; i++)
	{
		result = (result << 8) + array[i];
	}
	return result;
}//char 类型转 int，这个unsigned char存储比较特殊



void ReadMnist::readFile(vector< vector<unsigned char> >* _dataT, vector< vector<unsigned char> >* _dataL,
	char* _filePathT, char* _filePathL)
{

	dataT = _dataT;
	dataL = _dataL;
	filePathT = _filePathT;
	filePathL = _filePathL;



	//读取二进制文件
	ifstream fileT(filePathT, ios::binary);
	ifstream fileL(filePathL, ios::binary);

	ImageFileHeader imageFileHeader;
	LabelFileHeader labelFileHeader;

	fileT.read((char *)(&imageFileHeader), sizeof(imageFileHeader));//读取数据头
	fileL.read((char *)(&labelFileHeader), sizeof(labelFileHeader));//读取数据头

	//下面的变量并不是全部有用
	int magicNumberT = ConvertCharArrayToInt(imageFileHeader.MagicNumber, 4);
	int imageNumber = ConvertCharArrayToInt(imageFileHeader.NumberOfImages, 4);
	int rowsNumber = ConvertCharArrayToInt(imageFileHeader.NumberOfRows, 4);
	int columsNumber = ConvertCharArrayToInt(imageFileHeader.NumberOfColums, 4);
	int allPixNum = imageNumber * rowsNumber * columsNumber;

	int magicNumberL = ConvertCharArrayToInt(labelFileHeader.MagicNumber, 4);
	int itemNumber = ConvertCharArrayToInt(labelFileHeader.NumberOfItems, 4);

	//读取图像内容，以vector< vector<unsigned char> > 形式储存在内存中
	//imageNumber
	for (int i = 0; i<imageNumber; ++i)
	{
		//每个图像由28*28=784个像素点组成
		unsigned char tmpT[784];
		fileT.read((char*)&tmpT, 784 * sizeof(unsigned char));
		vector<unsigned char> tmpDataT(tmpT, tmpT + 784);
		dataT->push_back(tmpDataT);
		tmpDataT.clear();

		//读取标签文件，这里直接表示为网络目标输出，如果图片标签为3，则存储为{0,0,0,2,0,0,0,0,0,0}
		unsigned char tmpL[10] = { 0 };
		unsigned char label;
		fileL.read((char*)&label, sizeof(label));
		tmpL[label] = MARK;
		vector<unsigned char> tmpDataL(tmpL, tmpL + 10);
		dataL->push_back(tmpDataL);
		tmpDataL.clear();
	}
	fileT.close();
	fileL.close();

}




