#ifndef NETMAKER_H
#define NETMAKER_H

#include "cnn.h"
#include "readMnist.h"
#include <iostream>
#include <vector>
#include <fstream>
#include "distortion.h"




class NetMaker{
	int id;
	std::ofstream writefile;
	std::ofstream testfile;
	double MSE;
	bool isCnn;

	ReadMnist readFile;
	ReadMnist readtestFile;

	Distortion distortion;

	double min_eta;

	void computeInfo();
	

public:
	CNNnetwork cnnNet;
	std::vector< std::vector<unsigned char> > inputData;
	std::vector< std::vector<unsigned char> > outputData;
	std::vector< std::vector<unsigned char> > testData;
	std::vector< std::vector<unsigned char> > testLabel;

#if LOG

	std::ofstream writeTxtData;
#endif
	

	CnnHeader cnnHeader;
	NnHeader nnHeader;


	//����Netmaker�����ͬʱ����Network����
	NetMaker(unsigned* _k) : MSE(0) , min_eta(0.00005){

		isCnn = 1;
		for (unsigned i = 0; i < 5; i++) {
			cnnHeader.kernels[i] = _k[i];
		}
		
		computeInfo();

		cnnNet.createCnn(cnnHeader);
	}

	NetMaker(std::vector<int>& _k) :MSE(0) 
	{
		isCnn = 0;
		nnHeader.i = _k.size();
		for (unsigned i = 0; i < _k.size(); i++) {
			nnHeader.nodes[i] = _k[i];
		}

		computeInfo();

		cnnNet.createNn(nnHeader);
	}

	//��ȡѵ���Ͳ������ݣ������ο�readMnist.h
	void getSampleData(); 

	//ѵ������������
	void train();

	//������ȷ�ʣ�ֻ��֪���������10��ֵ���δ�������0-9�����ƶȣ����������-1��1�����ƶ�Խ��ֵԽ��Ȼ���������Ҳ�����׶���
	void test(); 

	//�洢ѵ������Ȩֵ�����ؽ�����
	void weightSave(); 

	//��Ӧ��ȡ�洢��Ȩֵ
	void weightRead(); 
};


///////////////////////////////////


#endif	//NetMaker