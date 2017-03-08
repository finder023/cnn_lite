#include "netMaker.h"
#include <iostream>
#include <vector>
#include <time.h>


using namespace std;


#if CVSHOW

#include <opencv2/opencv.hpp>
using namespace cv;

#endif // CVSHOW


int main(){

	//clock_t startTime = clock();
	unsigned netNodes[5] = { 1, 6, 50, 100, 10};

	int _nodes[3] = { 841, 100,  10 };

	vector<int> nodes(_nodes, _nodes + 3);

	NetMaker net(netNodes);

	net.getSampleData();
	
	net.train();


	net.weightSave();

	
#if CVSHOW 
	double output[10];	//这个是空数组，直接填

	vector<unsigned char> image(28*28,0);	//这个是图像内容，需要赋值

	Mat img = Mat::zeros(Size(28, 28), CV_8UC1);

	int a = 1;

	namedWindow("img", 0);

	while ((char)waitKey(1000) != 27){
		
		clock_t startTime = clock();

		for (int i = 0; i < 28; i++){
			for (int j = 0; j < 28; j++){
				img.at<uchar>(i, j) = net.inputData[a][28 * i + j];
			}
		}

		//对图像进行识别
		net.network.classifer(net.inputData[a++], output);

		double maxElem = 0;
		unsigned elemLocation = 0;
		for (unsigned j = 0; j < 10; j++){
			if (maxElem < output[j]){
				maxElem = output[j];
				elemLocation = j;
			}
		}

		cout << "识别结果: " << elemLocation << " " << endl;

		clock_t endTime = clock();
		cout << "Run time: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << " S" << endl;

		imshow("img", img);
	}


	
#endif //CVSHOW
	
}

