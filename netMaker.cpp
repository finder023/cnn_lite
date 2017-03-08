#include "netMaker.h"

using namespace std;

void NetMaker::getSampleData(){

	char trainFilePath[] = "data/trainimages.dat";
	char trainLabelPath[] = "data/trainlabels.dat";
	char testFilePath[] = "data/trainimages.dat";
	char testLabelPath[] = "data/trainlabels.dat";

	

	readFile.readFile(&inputData, &outputData, trainFilePath, trainLabelPath);
	readtestFile.readFile(&testData, &testLabel, testFilePath, testLabelPath);
}


void NetMaker::computeInfo()
{
	if (isCnn) {
		for (int i = 0; i < 5; i++) {
			switch (i)
			{
			case 0:
				cnnHeader.nodes[i] = cnnHeader.kernels[i] * 29 * 29;
				cnnHeader.maps[i] = 29;
				break;
			case 1:
				cnnHeader.nodes[i] = cnnHeader.kernels[i] * 13 * 13;
				cnnHeader.layer[i - 1] = cnnHeader.kernels[i] * (5 * 5 + 1);
				cnnHeader.maps[i] = 13;
				break;
			case 2:
				cnnHeader.nodes[i] = cnnHeader.kernels[i] * 5 * 5;
				cnnHeader.layer[i - 1] = cnnHeader.kernels[i - 1] * cnnHeader.kernels[i] * (5 * 5 + 1);
				cnnHeader.maps[i] = 5;
				break;
			default:
				cnnHeader.nodes[i] = cnnHeader.kernels[i];
				cnnHeader.layer[i - 1] = cnnHeader.nodes[i] * (cnnHeader.nodes[i - 1] + 1);
				cnnHeader.maps[i] = 1;
				break;
			}
		}
	}
	else {
		for (int i = 1; i < nnHeader.i; i++) {
			nnHeader.layer[i - 1] = nnHeader.nodes[i] * (nnHeader.nodes[i - 1] + 1);
		}
	}
}


void NetMaker::train()
{
	//创建训练需要变量，看名字就知道干什么的了

	vector<double> inputVector;
	vector<double> targetOutputVector;
	vector<double> actualOutputVector;


	if (isCnn) {
		inputVector = vector<double>(cnnHeader.nodes[0], 0.0);
		targetOutputVector = vector<double>(cnnHeader.nodes[4], 0.0);
		actualOutputVector = vector<double>(cnnHeader.nodes[4], 0.0);
	}
	else {
		inputVector = vector<double>(nnHeader.nodes[0], 0.0);
		targetOutputVector = vector<double>(nnHeader.nodes[nnHeader.i - 1], 0.0);
		actualOutputVector = vector<double>(nnHeader.nodes[nnHeader.i - 1], 0.0);
	}

	int ii, jj;
	int trainTimes = 0;

#if LOG
	char g[20] = { 0 };

	sprintf(g, "log_%f.txt", cnnNet.etaLearningRate);

	writeTxtData.open(g);
#endif // LOG


	//一百次跌代训练
	double MSEpre = ULONG_MAX_;
	int biggerTimes = 0;
	while (biggerTimes < 3) {

		cout << "trainTimes: " << ++trainTimes << endl;
		for (size_t t = 0; t < inputData.size(); t++) {

			//初始化输入向量，为了节省内存，只用了一个向量
			for (ii = 0; ii < inputVector.size(); ii++) {
				inputVector[ii] = 0.0;
			}

			//读取图片是28*28，输入图片是29*29，所以输入图片四周像素点设为0，并且将数值归一化为-1（纯黑）到1（纯白）之间，
			for (ii = 0; ii < 28; ii++) {
				for (jj = 0; jj < 28; jj++) {
					inputVector[1 + jj + 29 * (ii + 1)] = (double)inputData[t][jj + ii * 28] / 128.0 - 1.0;
				}
			}
			//同样目标输出归一化为-1到1之间
			for (ii = 0; ii < targetOutputVector.size(); ii++) {
				targetOutputVector[ii] = (double)outputData[t][ii] - 1.0;
			}


			distortion.GenerateDistortionMap();
			distortion.ApplyDistortionMap(inputVector);

			//网络前向计算，填充计算输出向量
			cnnNet.calculate(inputVector, actualOutputVector);
			//误差后向计算，并更新权值
			cnnNet.backpropagate(actualOutputVector, targetOutputVector);
		}

		test();

		//当MSE连续三次变大的时候认为有过拟合现象，此时应停止训练
		if (MSE >= MSEpre) {
			biggerTimes++;
			weightSave();
		}
		else {
			biggerTimes = 0;
		}
		if (cnnNet.etaLearningRate > min_eta){
			if (trainTimes % 2 == 0)
				cnnNet.etaLearningRate *= 0.794183335;
		}

		MSEpre = MSE;
	}

	weightSave();

}



void NetMaker::test()
{
	int rightNum = 0;
	vector<double> output;
	if (isCnn) {
		output = vector<double>(cnnHeader.nodes[4], 0.0);
	}
	else
		output = vector<double>(nnHeader.nodes[nnHeader.i - 1], 0.0);

	MSE = 0;
	for (size_t i = 0; i < testData.size(); i++) {
		cnnNet.classifer(testData[i], output);

		double maxElem = -2;
		unsigned elemLocation = 0;
		for (unsigned j = 0; j < output.size(); j++) {
			if (maxElem < output[j]) {
				maxElem = output[j];
				elemLocation = j;
			}
		}

		int targetLocation = 0;
		while (!testLabel[i][targetLocation])
			++targetLocation;
		if (elemLocation == targetLocation)
			++rightNum;

		double mse = 0.0;

		for (unsigned x = 0; x < output.size(); x++) {
			mse += (testLabel[i][x] - 1 - output[x])*(testLabel[i][x] - 1 - output[x]);
		}
		mse = mse * 0.5 / output.size();
		MSE += mse;
	}

	MSE /= testData.size();
	cout << "MSE: " << MSE << endl;
	cout << "Accuracy: " << (double)rightNum / testData.size() << endl;
	cout << "LearningRate: " << cnnNet.etaLearningRate << endl;

#if LOG
	writeTxtData << MSE << "      ";
	writeTxtData << (double)rightNum / testData.size() << "       " << cnnNet.etaLearningRate << endl;
#endif

}


void NetMaker::weightSave() {


	if (isCnn) {
		writefile.open("data/weightCnn.dat", ios::binary);
		writefile.write((char*)(&cnnHeader), sizeof(cnnHeader));
	}
	else {
		writefile.open("data/weightNn.dat", ios::binary);
		writefile.write((char*)(&nnHeader), sizeof(nnHeader));
	}

	for (size_t i = 1; i < cnnNet.m_layers.size(); i++) {
		for (size_t j = 0; j < cnnNet.m_layers[i]->m_weights.size(); j++) {
			writefile.write((char*)(&(cnnNet.m_layers[i]->m_weights[j])), sizeof(double));
		}
	}
	//cout << "x " << x << endl;
	writefile.close();

}



void NetMaker::weightRead()
{
	if (isCnn) {
		CnnHeader header;
		ifstream readfile_("data/weightCnn9887.dat", ios::binary);


		readfile_.read((char*)(&header), sizeof(header));



		double* layer[4];

		for (int i = 0; i < 4; i++) {
			layer[i] = (double*)malloc(sizeof(double)*header.layer[i]);
			readfile_.read((char*)(layer[i]), sizeof(double)*header.layer[i]);
			cnnNet.m_layers[i + 1]->m_weights = vector<double>(layer[i], layer[i] + header.layer[i]);
		}


		readfile_.close();
	}
	else {
		NnHeader header;
		ifstream readfile_("data/weightNn.dat", ios::binary);


		readfile_.read((char*)(&header), sizeof(header));

		double** layer = (double**)malloc(sizeof(double*)*(nnHeader.i - 1));

		for (int i = 0; i < nnHeader.i - 1; i++) {
			layer[i] = (double*)malloc(sizeof(double)*header.layer[i]);
			readfile_.read((char*)(layer[i]), sizeof(double)*header.layer[i]);
			cnnNet.m_layers[i + 1]->m_weights = vector<double>(layer[i], layer[i] + header.layer[i]);
		}
	}

}
