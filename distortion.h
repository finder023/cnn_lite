#ifndef DISTORTION_H
#define DISTORTION_H

#include <iostream>
#include <vector>
#include <cstdlib>
#include <math.h>




#define UNIFORM_PLUS_MINUS_ONE ( (double)(2.0 * rand())/RAND_MAX - 1.0 )
#define GAUSSIAN_FIELD_SIZE 21
#define FALSE 0
#define TRUE 1


#define RANDOM 1
#define ROTATION 1
#define GAUSSIAN_ 1
#define SCALE 1

#define CHANGE_ 0

typedef int BOOL;

class Distortion
{
	double m_dMaxScaling;
	double m_dElasticScaling;
	double m_dMaxRotation;
	double m_dElasticSigma;
	double m_dRandom;

public:

	int m_cCount;
	int m_cCols;
	int m_cRows;

	double m_GaussianKernel[GAUSSIAN_FIELD_SIZE][GAUSSIAN_FIELD_SIZE];

	//double* m_DispH;  // horiz distortion map array
	//double* m_DispV;  // vert distortion map array
	std::vector<double> m_DispH;
	std::vector<double> m_DispV;

	Distortion();

	double& At(std::vector<double>& p, int row, int col);


	void GenerateDistortionMap(double severityFactor = 1.0);
	//void ApplyDistortionMap(double* inputVector);
	void ApplyDistortionMap(std::vector<double>& inputVector);


};


#endif



