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


	//创建Netmaker对象的同时创建Network对象
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

	//读取训练和测试数据，函数参考readMnist.h
	void getSampleData(); 

	//训练函数无疑问
	void train();

	//计算正确率，只需知道计算输出10个值依次代表数字0-9的相似度，理论输出是-1到1，相似度越高值越大，然后这个函数也就容易读了
	void test(); 

	//存储训练还得权值，不必介绍了
	void weightSave(); 

	//对应读取存储的权值
	void weightRead(); 
};


///////////////////////////////////


#endif	//NetMaker