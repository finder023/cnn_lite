
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
}//char ����ת int�����unsigned char�洢�Ƚ�����



void ReadMnist::readFile(vector< vector<unsigned char> >* _dataT, vector< vector<unsigned char> >* _dataL,
	char* _filePathT, char* _filePathL)
{

	dataT = _dataT;
	dataL = _dataL;
	filePathT = _filePathT;
	filePathL = _filePathL;



	//��ȡ�������ļ�
	ifstream fileT(filePathT, ios::binary);
	ifstream fileL(filePathL, ios::binary);

	ImageFileHeader imageFileHeader;
	LabelFileHeader labelFileHeader;

	fileT.read((char *)(&imageFileHeader), sizeof(imageFileHeader));//��ȡ����ͷ
	fileL.read((char *)(&labelFileHeader), sizeof(labelFileHeader));//��ȡ����ͷ

	//����ı���������ȫ������
	int magicNumberT = ConvertCharArrayToInt(imageFileHeader.MagicNumber, 4);
	int imageNumber = ConvertCharArrayToInt(imageFileHeader.NumberOfImages, 4);
	int rowsNumber = ConvertCharArrayToInt(imageFileHeader.NumberOfRows, 4);
	int columsNumber = ConvertCharArrayToInt(imageFileHeader.NumberOfColums, 4);
	int allPixNum = imageNumber * rowsNumber * columsNumber;

	int magicNumberL = ConvertCharArrayToInt(labelFileHeader.MagicNumber, 4);
	int itemNumber = ConvertCharArrayToInt(labelFileHeader.NumberOfItems, 4);

	//��ȡͼ�����ݣ���vector< vector<unsigned char> > ��ʽ�������ڴ���
	//imageNumber
	for (int i = 0; i<imageNumber; ++i)
	{
		//ÿ��ͼ����28*28=784�����ص����
		unsigned char tmpT[784];
		fileT.read((char*)&tmpT, 784 * sizeof(unsigned char));
		vector<unsigned char> tmpDataT(tmpT, tmpT + 784);
		dataT->push_back(tmpDataT);
		tmpDataT.clear();

		//��ȡ��ǩ�ļ�������ֱ�ӱ�ʾΪ����Ŀ����������ͼƬ��ǩΪ3����洢Ϊ{0,0,0,2,0,0,0,0,0,0}
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




