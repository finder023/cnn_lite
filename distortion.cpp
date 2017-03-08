#include "distortion.h"

using namespace std;

Distortion::Distortion()
{
	m_dElasticScaling = 12.0; //0.5
	m_dMaxScaling = 15.0;  // like 20.0 for 20%
	m_dMaxRotation = 15.0;  // like 20.0 for 20 degrees
	m_dElasticSigma = 4.0;  // higher numbers are more smooth and less distorted; Simard uses 4.0
	m_dRandom = 0.65;

	m_cCols = 29;
	m_cRows = 29;

	m_cCount = m_cCols * m_cRows;

	//m_DispH = new double[m_cCount];
	//m_DispV = new double[m_cCount];
	m_DispH = vector<double>(m_cCount, 0.0);
	m_DispV = vector<double>(m_cCount, 0.0);

	int iiMid = GAUSSIAN_FIELD_SIZE / 2;  // GAUSSIAN_FIELD_SIZE is strictly odd

	double twoSigmaSquared = 2.0 * (m_dElasticSigma)* (m_dElasticSigma);
	//twoSigmaSquared = 1.0 / twoSigmaSquared;
	double twoPiSigma = 1.0 / (twoSigmaSquared * 3.1415926535897932384626433832795);

	for (int col = 0; col<GAUSSIAN_FIELD_SIZE; ++col)
	{
		for (int row = 0; row<GAUSSIAN_FIELD_SIZE; ++row)
		{
			m_GaussianKernel[row][col] = twoPiSigma *
				(exp(-(((row - iiMid)*(row - iiMid) + (col - iiMid)*(col - iiMid)) / twoSigmaSquared)));
		}
	}
}




//double& Distortion::At(double* p, int row, int col)  // zero-based indices, starting at bottom-left
//{
//	double x = 0.0;
//	int location = row * m_cCols + col;
//	if (location >= 0 && location < m_cCount && row < m_cRows && row >= 0 && col < m_cCols && col >= 0)
//		return p[location];
//	return x;
//	
//}

double& Distortion::At(vector<double>& p, int row, int col)  // zero-based indices, starting at bottom-left
{
	double x = 0.0;
	int location = row * m_cCols + col;
	//if (location >= 0 && location < m_cCount && row < m_cRows && row >= 0 && col < m_cCols && col >= 0)
	return p[location];
	//return x;

}



void Distortion::GenerateDistortionMap(double severityFactor /* =1.0 */)
{
	int row, col;
	//double* uniformH = new double[m_cCount];
	//double* uniformV = new double[m_cCount];
	m_DispH = vector<double>(m_cCount, 0.0);
	m_DispV = vector<double>(m_cCount, 0.0);


	vector<double> uniformH(m_cCount, 0);
	vector<double> uniformV(m_cCount, 0);


	for (col = 0; col<m_cCols; ++col)
	{
		for (row = 0; row<m_cRows; ++row)
		{
			At(uniformH, row, col) = UNIFORM_PLUS_MINUS_ONE;
			At(uniformV, row, col) = UNIFORM_PLUS_MINUS_ONE;
		}
	}
	//这里是设置每个横向和纵向的变换范围，接下来使用高斯核处理
	// filter with gaussian

#if GAUSSIAN_
	double fConvolvedH, fConvolvedV;
	double fSampleH, fSampleV;
	double elasticScale = severityFactor * m_dElasticScaling;
	int xxx, yyy, xxxDisp, yyyDisp;
	int iiMid = GAUSSIAN_FIELD_SIZE / 2;  // GAUSSIAN_FIELD_SIZE is strictly odd

	for (col = 0; col<m_cCols; ++col)
	{
		for (row = 0; row<m_cRows; ++row)
		{
			fConvolvedH = 0.0;
			fConvolvedV = 0.0;

			for (xxx = 0; xxx<GAUSSIAN_FIELD_SIZE; ++xxx)
			{
				for (yyy = 0; yyy<GAUSSIAN_FIELD_SIZE; ++yyy)
				{
					xxxDisp = col - iiMid + xxx;
					yyyDisp = row - iiMid + yyy;

					if (xxxDisp<0 || xxxDisp >= m_cCols || yyyDisp<0 || yyyDisp >= m_cRows)
					{
						fSampleH = 0.0;
						fSampleV = 0.0;
					}
					else
					{
						fSampleH = At(uniformH, yyyDisp, xxxDisp);
						fSampleV = At(uniformV, yyyDisp, xxxDisp);
					}

					fConvolvedH += fSampleH * m_GaussianKernel[yyy][xxx];
					fConvolvedV += fSampleV * m_GaussianKernel[yyy][xxx];
				}
			}

			At(m_DispH, row, col) = elasticScale * fConvolvedH;
			At(m_DispV, row, col) = elasticScale * fConvolvedV;
		}
	}



	//next, the scaling of the image by a random scale factor
	//Horizontal and vertical directions are scaled independently


#endif	//GAUSSIAN_

	int iMid = m_cRows / 2;

#if SCALE

	double dSFHoriz = severityFactor * m_dMaxScaling / 100.0 * UNIFORM_PLUS_MINUS_ONE;  // m_dMaxScaling is a percentage
	double dSFVert = severityFactor * m_dMaxScaling / 100.0 * UNIFORM_PLUS_MINUS_ONE;  // m_dMaxScaling is a percentage




	for (row = 0; row<m_cRows; ++row)
	{
		for (col = 0; col<m_cCols; ++col)
		{
			At(m_DispH, row, col) += dSFHoriz * (col - iMid);
			At(m_DispV, row, col) -= dSFVert * (iMid - row);  // negative because of top-down bitmap
		}
	}

#endif //SCALE


#if RANDOM

	for (row = 0; row < m_cRows; ++row){
		for (col = 0; col < m_cCols; ++col){
			At(m_DispH, row, col) += UNIFORM_PLUS_MINUS_ONE * m_dRandom;
			At(m_DispV, row, col) += UNIFORM_PLUS_MINUS_ONE * m_dRandom;
		}
	}

#endif	//RANDOM

	// finally, apply a rotation

#if ROTATION
	double angle = severityFactor * m_dMaxRotation * UNIFORM_PLUS_MINUS_ONE;
	angle = angle * 3.1415926535897932384626433832795 / 180.0;  // convert from degrees to radians

	double cosAngle = cos(angle);
	double sinAngle = sin(angle);

	for (row = 0; row<m_cRows; ++row)
	{
		for (col = 0; col<m_cCols; ++col)
		{
			/*At(m_DispH, row, col) += (col - iMid) * (cosAngle - 1) - (iMid - row) * sinAngle;
			At(m_DispV, row, col) -= (iMid - row) * (cosAngle - 1) + (col - iMid) * sinAngle; */ // negative because of top-down bitmap
			At(m_DispH, row, col) += (col - iMid) * (cosAngle - 1) + (row - iMid) * sinAngle;
			At(m_DispV, row, col) += (row - iMid) * (cosAngle - 1) - (col - iMid) * sinAngle;
		}
	}

#endif //ROTATION
}



void Distortion::ApplyDistortionMap(vector<double>& inputVector)
{
	// applies the current distortion map to the input vector

	// For the mapped array, we assume that 0.0 == background, and 1.0 == full intensity information
	// This is different from the input vector, in which +1.0 == background (white), and 
	// -1.0 == information (black), so we must convert one to the other

	std::vector< std::vector< double > >   mappedVector(m_cRows, std::vector< double >(m_cCols, 0.0));

	int sourceRow, sourceCol;
	double fracRow, fracCol;
	double w1, w2, w3, w4;
	double sourceValue;
	int row, col;
	int sRow, sCol, sRowp1, sColp1;
	BOOL bSkipOutOfBounds;

	for (row = 0; row<m_cRows; ++row)
	{
		for (col = 0; col<m_cCols; ++col)
		{
			// the pixel at sourceRow, sourceCol is an "phantom" pixel that doesn't really exist, and
			// whose value must be manufactured from surrounding real pixels (i.e., since 
			// sourceRow and sourceCol are floating point, not ints, there's not a real pixel there)
			// The idea is that if we can calculate the value of this phantom pixel, then its 
			// displacement will exactly fit into the current pixel at row, col (which are both ints)
			//at表示读取像素值，m_diopV是一个竖直方向的变换图像。
			sourceRow = row + floor(At(m_DispV, row, col));
			sourceCol = col + floor(At(m_DispH, row, col));


			// weights for bi-linear interpolation

			fracRow = At(m_DispV, row, col) - (double)floor(At(m_DispV, row, col));
			fracCol = At(m_DispH, row, col) - (double)floor(At(m_DispH, row, col));
			//这里是取小数点的操作。取值范围在-0.5~0.5之间

			w4 = fracRow * fracCol;
			w2 = (1.0 - fracRow) * fracCol;
			w3 = fracRow * (1 - fracCol);
			w1 = (1 - fracRow) *(1 - fracCol);

			//floor(0.5);//小于或等于x的最大整数
			//ceil(0.3);//大于或等于x的最下整数
			//取得若干权值
			// limit indexes

			/*
			while (sourceRow >= m_cRows ) sourceRow -= m_cRows;
			while (sourceRow < 0 ) sourceRow += m_cRows;

			while (sourceCol >= m_cCols ) sourceCol -= m_cCols;
			while (sourceCol < 0 ) sourceCol += m_cCols;
			*/
			bSkipOutOfBounds = FALSE;

			if ((sourceRow + 1.0) >= m_cRows)	bSkipOutOfBounds = TRUE;
			if (sourceRow < 0)				bSkipOutOfBounds = TRUE;

			if ((sourceCol + 1.0) >= m_cCols)	bSkipOutOfBounds = TRUE;
			if (sourceCol < 0)				bSkipOutOfBounds = TRUE;

			if (bSkipOutOfBounds == FALSE)
			{
				// the supporting pixels for the "phantom" source pixel are all within the 
				// bounds of the character grid.
				// Manufacture its value by bi-linear interpolation of surrounding pixels

				sRow = sourceRow;
				sCol = sourceCol;

				sRowp1 = sRow + 1;
				sColp1 = sCol + 1;

				while (sRowp1 >= m_cRows) sRowp1 -= m_cRows;
				while (sRowp1 < 0) sRowp1 += m_cRows;

				while (sColp1 >= m_cCols) sColp1 -= m_cCols;
				while (sColp1 < 0) sColp1 += m_cCols;

				// perform bi-linear interpolation
				//果然是上下左右
				sourceValue = w1 * At(inputVector, sRow, sCol) +
					w2 * At(inputVector, sRow, sColp1) +
					w3 * At(inputVector, sRowp1, sCol) +
					w4 * At(inputVector, sRowp1, sColp1);

			}
			else
			{
				// At least one supporting pixel for the "phantom" pixel is outside the
				// bounds of the character grid. Set its value to "background"

				sourceValue = -1.0;  // "background" color in the -1 -> +1 range of inputVector
			}

			mappedVector[row][col] = sourceValue;  // conversion to 0->1 range we are using for mappedVector

		}
	}

	// now, invert again while copying back into original vector

	for (row = 0; row<m_cRows; ++row)
	{
		for (col = 0; col<m_cCols; ++col)
		{
			At(inputVector, row, col) = mappedVector[row][col];
		}
	}

}




