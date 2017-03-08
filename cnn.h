#ifndef CNN_H
#define CNN_H

#include <iostream>
#include <vector>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits.h>
#include "configure.h"

//程序中用到的特殊标记和sigmoid激活函数，是tanh函数的拉伸变形

#define ULONG_MAX_ 0xffffffffUL

#if LOGISTIC

inline double sigmoid(double x) { return 1.0 / (1 + exp(-1.0 * x)); }
inline double dsigmoid(double x) { return sigmoid(x) * (1.0 - sigmoid(x)); }
inline double randWeight() { return 0.05*((double)(2.0 * rand()) / RAND_MAX - 1.0); }
#define MARK 1

#else

inline double sigmoid(double x) { return 1.7159*tanh(0.66666667*x); }
inline double dsigmoid(double x) { return 0.66666667 / 1.7159*(1.7159 + x)*(1.7159 - x); }
inline double randWeight() { return 0.05*((double)(2.0 * rand()) / RAND_MAX - 1.0); }
#define MARK 2

#endif //	!SIGMOID








struct CnnHeader {
	unsigned layer[4];
	unsigned nodes[5];
	unsigned kernels[5];
	unsigned maps[5];
};

struct NnHeader {
	int i;
	unsigned nodes[6];
	unsigned layer[5];
};

class CNNconnection{
public:
	
	unsigned weightIdx;	//权值ID
	unsigned neuralIdx;	//神经元ID
	CNNconnection(unsigned _neuralIdx, unsigned _weightIdx) : weightIdx(_weightIdx), neuralIdx(_neuralIdx) {
	}
};



class CNNneural{
public:
	double output;	//神经元输出值
	double sum;
	std::vector<CNNconnection> m_connection;	//连接
};



class CNNlayer{
public:
	CNNlayer* preLayer;
	std::vector<CNNneural> m_neurals;
	std::vector<double> m_weights;
	std::vector<double> m_gradient;

	CNNlayer(CNNlayer* pLayer){
		preLayer = pLayer;
	}

	void calculate(); 

	void backPropagate(std::vector<double>& dErr_wrt_dXn, std::vector<double>& dErr_wrt_dXnm1, double eta);

};



class CNNnetwork{
public:
	unsigned nLayer;
	std::vector<unsigned> nodes;
	std::vector<double> actualOutput;
	double etaLearningRate;
	unsigned iterNum;	//迭代次数

	std::vector<CNNlayer*> m_layers;

	CNNnetwork()
	{
		etaLearningRate = 0.001;

	}

	void createCnn(CnnHeader& _header); 

	void createNn(NnHeader& _header);

	void calculate(std::vector<double>& inputvector, std::vector<double>& outputvector);

	void backpropagate(std::vector<double>& actualOutput, std::vector<double> targetOutput);

	void classifer(std::vector<unsigned char>& inVect, std::vector<double>& outVect);

};



#endif