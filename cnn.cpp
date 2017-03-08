#include "cnn.h"


using namespace std;


///////////////////////////////////////////
//		
//			calss CNNlayer
//
//////////////////////////////////////////


void CNNlayer::calculate(){
	vector<CNNneural>::iterator nit;
	vector<CNNconnection>::iterator cit;

	double dSum = 0;
	

	for (nit = m_neurals.begin(); nit < m_neurals.end(); nit++){
		CNNneural& n = *nit;
		cit = n.m_connection.begin();

		dSum = m_weights[cit->weightIdx];//第一个是偏置
		//计算输入与权值乘积的和


		for (cit++; cit < n.m_connection.end(); cit++){
			if (cit->neuralIdx == ULONG_MAX_) {
				dSum += m_weights[cit->weightIdx];
			}
			else {
				dSum += (m_weights[cit->weightIdx])*(preLayer->m_neurals[cit->neuralIdx].output);
			}
		}
		//计算输出
		n.output = sigmoid(dSum);
		n.sum = dSum;

	}

}



void CNNlayer::backPropagate(vector<double>& dErr_wrt_dXn, vector<double>& dErr_wrt_dXnm1, double eta){

	unsigned ii;
	unsigned kk;
	unsigned nIndex;
	double output;

	//输入和的梯度
	vector<double> dErr_wrt_dYn(m_neurals.size());

	//权值的梯度
	vector<double> dErr_wrt_dWn(m_weights.size(), 0.0);

	vector<CNNneural>::iterator nit;
	vector<CNNconnection>::iterator cit;

	//计算输入和的梯度
	for (ii = 0; ii < m_neurals.size(); ii++){
		output = m_neurals[ii].output;
		dErr_wrt_dYn[ii] = dsigmoid(output) * dErr_wrt_dXn[ii];
	}

	//计算权值的梯度
	ii = 0;
	for (nit = m_neurals.begin(); nit < m_neurals.end(); nit++){

		CNNneural& n = *nit;

		for (cit = n.m_connection.begin(); cit < n.m_connection.end(); cit++){
			kk = cit->neuralIdx;
			if (kk == ULONG_MAX_){
				output = 1.0;	//偏置
			}
			else{
				output = preLayer->m_neurals[kk].output;
			}
			dErr_wrt_dWn[cit->weightIdx] += dErr_wrt_dYn[ii] * output;
		}
		ii++;
	}

	//计算上层误差

	ii = 0;
	for (nit = m_neurals.begin(); nit < m_neurals.end(); nit++){
		CNNneural& n = *nit;

		for (cit = n.m_connection.begin(); cit < n.m_connection.end(); cit++){
			kk = cit->neuralIdx;

			if (kk != ULONG_MAX_){
				nIndex = kk;

				dErr_wrt_dXnm1[nIndex] += dErr_wrt_dYn[ii] * m_weights[cit->weightIdx];
			}
		}
		ii++;
	}

	//更新权值
	for (vector<double>::size_type i = 0; i != m_weights.size(); i++){
		m_weights[i] -= eta*dErr_wrt_dWn[i];

	}

	


}


///////////////////////////////////////////
//		
//			calss CNNnetwork
//
//////////////////////////////////////////



//这是程序最难的以部分了，设置各个神经元的链接，尤其是卷积层的，最好看看卷积的原理在读代码吧，卷积和opencv的滤波器是一样的，只不过卷积的核不是固定的，它也是若干权值组成，能进行学习更新数值，这部分和图像联系起来，否则抽象到难以理解



void CNNnetwork::createCnn(CnnHeader &_header)
{
	unsigned int ii, jj, kk, ll;
	int xx, yy;


	//layer input
	CNNlayer* pLayer = new CNNlayer(NULL);
	m_layers.push_back(pLayer);
	for (ii = 0; ii < _header.nodes[0]; ii++) {
		CNNneural sneural;
		sneural.output = 0;
		pLayer->m_neurals.push_back(sneural);
	}

	//设置卷积层连接

	for (xx = 1; xx < 3; xx++) {


		pLayer = new CNNlayer(pLayer);
		m_layers.push_back(pLayer);

		for (ii = 0; ii < _header.nodes[xx]; ii++) {
			CNNneural sneural;
			sneural.output = 0;
			pLayer->m_neurals.push_back(sneural);
		}

		for (ii = 0; ii < _header.layer[xx - 1]; ii++) {

			pLayer->m_weights.push_back(randWeight());

			pLayer->m_gradient.push_back(0.0);
		}

		for (ii = 0; ii < _header.kernels[xx]; ii++) {
			for (jj = 0; jj < _header.maps[xx]; jj++) {
				for (kk = 0; kk < _header.maps[xx]; kk++) {

					CNNneural& n = pLayer->m_neurals[kk + jj * _header.maps[xx] + ii * _header.maps[xx] * _header.maps[xx]];

					for (yy = 0; yy < _header.kernels[xx - 1]; yy++) {

						int iNumWeight = _header.kernels[xx - 1] * ii * 26 + yy * 26;
						CNNconnection sconnections(ULONG_MAX_, iNumWeight++);
						n.m_connection.push_back(sconnections);	//添加偏置

						for (ll = 0; ll < 25; ll++) {
							n.m_connection.push_back(CNNconnection(yy * _header.maps[xx - 1] * _header.maps[xx - 1] + (2 * jj + ll / 5)*_header.maps[xx - 1] + 2 * kk + ll % 5, iNumWeight++));
						}

					}
				}
			}
		}

	}

	//设置全连接层连接

	for (int xx = 3; xx < 5; xx++) {

		pLayer = new CNNlayer(pLayer);
		m_layers.push_back(pLayer);

		for (ii = 0; ii < _header.nodes[xx]; ii++) {
			CNNneural sneural;
			sneural.output = 0;
			pLayer->m_neurals.push_back(sneural);
		}

		for (ii = 0; ii < _header.layer[xx - 1]; ii++) {
			pLayer->m_weights.push_back(randWeight());

			pLayer->m_gradient.push_back(0.0);
		}

		int iNumWeight = 0;

		for (int fm = 0; fm < _header.nodes[xx]; fm++) {
			CNNneural& n = pLayer->m_neurals[fm];
			CNNconnection sconnections(ULONG_MAX_, iNumWeight++);
			n.m_connection.push_back(sconnections);

			for (ii = 0; ii < _header.nodes[xx - 1]; ii++) {
				n.m_connection.push_back(CNNconnection(ii, iNumWeight++));
			}
		}

	}

}

void CNNnetwork::createNn(NnHeader &_header)
{
	int ii, jj, kk, ll;


	//layer input
	CNNlayer* pLayer = new CNNlayer(NULL);
	m_layers.push_back(pLayer);
	for (ii = 0; ii < _header.nodes[0]; ii++) {
		CNNneural sneural;
		sneural.output = 0;
		pLayer->m_neurals.push_back(sneural);
	}

	//other layer

	for (ii = 1; ii < _header.i; ii++) {
		pLayer = new CNNlayer(pLayer);
		m_layers.push_back(pLayer);

		for (jj = 0; jj < _header.nodes[ii]; jj++) {
			CNNneural sneural;
			sneural.output = 0;
			pLayer->m_neurals.push_back(sneural);
		}

		for (jj = 0; jj < _header.layer[ii - 1]; jj++) {

			pLayer->m_weights.push_back(randWeight());

			pLayer->m_gradient.push_back(0.0);
		}

		int iNumWeight = 0;

		for (int fm = 0; fm < _header.nodes[ii]; fm++) {
			CNNneural& n = pLayer->m_neurals[fm];
			CNNconnection sconnections(ULONG_MAX_, iNumWeight++);
			n.m_connection.push_back(sconnections);

			for (jj = 0; jj < _header.nodes[ii - 1]; jj++) {
				n.m_connection.push_back(CNNconnection(jj, iNumWeight++));
			}
		}
	}

}





void CNNnetwork::calculate(vector<double>& inputVector, vector<double>& outputVector){


	vector<CNNlayer*>::iterator lit = m_layers.begin();
	vector<CNNneural>::iterator nit;


	nit = (*lit)->m_neurals.begin();
	int count = 0;

	//输入层神经元的输出就是像素点值
	for (; nit < (*lit)->m_neurals.end(); nit++){
		nit->output = inputVector[count];
		count++;
	}


	for (lit++; lit < m_layers.end(); lit++){
		(*lit)->calculate();
	}
	//cout << "cccccc" << endl;

	//计算输出
	lit = m_layers.end();
	lit--;

	nit = (*lit)->m_neurals.begin();

	for (unsigned ii = 0; ii < (*lit)->m_neurals.size(); ii++){
		outputVector[ii] = nit->output;
		nit++;
	}


}



void CNNnetwork::backpropagate(vector<double>& actualOutput, vector<double> targetOutput){

	vector<CNNlayer*>::iterator lit = m_layers.end() - 1;

	vector<double> dErr_wrt_dXlast((*lit)->m_neurals.size());
	vector< vector<double> > differentials;

	unsigned iSize = m_layers.size();

	differentials.resize(iSize);

	unsigned ii;

	//计算最后一层误差

	for (ii = 0; ii < (*lit)->m_neurals.size(); ii++){
		dErr_wrt_dXlast[ii] = actualOutput[ii] - targetOutput[ii];
	}


	differentials[iSize - 1] = dErr_wrt_dXlast;

	//其他层误差设为0
	for (ii = 0; ii < iSize - 1; ii++){
		differentials[ii].resize(m_layers[ii]->m_neurals.size(), 0.0);
	}

	//开始迭代计算

	lit = m_layers.end() - 1;
	ii = iSize - 1;

	for (; lit > m_layers.begin(); lit--){
		(*lit)->backPropagate(differentials[ii], differentials[ii - 1], etaLearningRate);

		ii--;
	}

	differentials.clear();
}



void CNNnetwork::classifer(vector<unsigned char>& inVect, vector<double>& outVect){
	vector<double> in(841, 0.0);
	int ii, jj;


	for (ii = 0; ii < 28; ii++) {
		for (jj = 0; jj < 28; jj++) {
			in[1 + jj + 29 * (ii + 1)] = (double)inVect[jj + ii * 28] / 128.0 - 1.0;
		}
	}

	//cout << 
	calculate(in, outVect);
}

