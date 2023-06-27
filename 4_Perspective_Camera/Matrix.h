#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <assert.h>
#include "geometry.h"

using namespace std;

const int default_dimension = 4;

struct Matrix
{
	vector<vector<float>> m;
	int row = 4;
	int col = 4;
	Matrix(int row_ = default_dimension, int col_ = default_dimension) : row(row_), col(col_)
	{
		m = vector<vector<float>>(row, vector<float>(col, 0));
	}
	std::vector<float>& Matrix::operator[](const int i)
	{
		assert(i >= 0 && i < row);
		return m[i];
	}


	Matrix operator*(const Matrix& a)
	{
		/*assert(row == a.col);
		Matrix result(row, a.col);
		for (int i = 0; i < row; i++) {
		for (int j = 0; j < a.col; j++) {
		result.m[i][j] = 0.f;
		for (int k = 0; k < col; k++) {
		result.m[i][j] += m[i][k] * a.m[k][j];
		}
		}
		}
		return result;*/

		assert(col == a.row);

		Matrix result(row, a.col);


		// 4x4 x 4x1

		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < a.col; ++j)
			{
				result.m[i][j] = 0.0f;
				for (int k = 0; k < col; ++k)
					result.m[i][j] += m[i][k] * a.m[k][j];
			}
		}
		return result;
	}

	Matrix transpose() {
		Matrix result(col, row);
		for (int i = 0; i < row; i++)
			for (int j = 0; j < col; j++)
				result[j][i] = m[i][j];
		return result;
	}

	Matrix inverse()
	{
		assert(row == col);
		// augmenting the square matrix with the identity matrix of the same dimensions A => [AI]
		Matrix result(row, col * 2);
		for (int i = 0; i<row; i++)
			for (int j = 0; j<col; j++)
				result[i][j] = m[i][j];
		for (int i = 0; i<row; i++)
			result[i][i + col] = 1;
		// first pass
		for (int i = 0; i<row - 1; i++) {
			// normalize the first row
			for (int j = result.col - 1; j >= 0; j--)
				result[i][j] /= result[i][i];
			for (int k = i + 1; k<row; k++) {
				float coeff = result[k][i];
				for (int j = 0; j<result.col; j++) {
					result[k][j] -= result[i][j] * coeff;
				}
			}
		}
		// normalize the last row
		for (int j = result.col - 1; j >= row - 1; j--)
			result[row - 1][j] /= result[row - 1][row - 1];
		// second pass
		for (int i = row - 1; i>0; i--) {
			for (int k = i - 1; k >= 0; k--) {
				float coeff = result[k][i];
				for (int j = 0; j<result.col; j++) {
					result[k][j] -= result[i][j] * coeff;
				}
			}
		}
		// cut the identity matrix back
		Matrix truncate(row, col);
		for (int i = 0; i<row; i++)
			for (int j = 0; j<col; j++)
				truncate[i][j] = result[i][j + col];
		return truncate;
	}

	friend std::ostream& operator<<(std::ostream& s, Matrix& m) {
		for (int i = 0; i < m.row; i++)
		{
			for (int j = 0; j < m.col; j++)
			{
				s << m[i][j];
				if (j < m.col - 1) s << "\t";
			}
			s << "\n";
		}
		return s;
	}
};
Matrix identity_matrix(int dimension = 4)
{
	Matrix d(dimension, dimension);

	for (int i = 0; i < dimension; ++i)
		d[i][i] = 1;
	return d;
}

//Matrix Operation

//matrix to vector
//4x1 matrix, 4 rows, 1 col
vec3 m2v(Matrix m)
{
	//return vec3(m[0][0] / m[3][0], m[1][0] / m[3][0], m[2][0] / m[3][0]);
	float inv_z = 1.0f / m[3][0];
	return vec3(m[0][0] * inv_z, m[1][0] * inv_z, m[2][0] * inv_z);
}
//vector to matrix
//4x1 matrix, 4 rows, 1 col (again!)
Matrix v2m(vec3 v)
{
	Matrix m(4, 1);
	m[0][0] = v.x;
	m[1][0] = v.y;
	m[2][0] = v.z;
	m[3][0] = 1.0f;
	return m;
}

//perspective projection matrix
/*
//why z' = z / (1 - z / c)
x' = x / (1 - z / c)
y' = y / (1 - z / c)

z / z' = (c - z) / c

z' / z = c / (c - z)

z' = c z /(c - z)

z' = z / (1 - z/c)


x / x' = y / y' = z / z' = (c - z) / c

x' / x = y' / y = z' / z = c / (c - z) = 1 / (1 - z / c)

x' = x / (1 - z / c)

*/

/*
1 0 0 0    x               x
0 1 0 0    y               y
0 0 1 0    z               z
0 0 r 1    1              rz + 1

z / (rz + 1) = z'

z / z' = rz + 1;

also z / z' = (c - z) / c

so (c - z) / c = (rz + 1)

rz = -z / c

r = -1/c

c = (eye - center);
*/
Matrix perspective_projection(vec3 eye, vec3 center)
{
	Matrix m = identity_matrix();
	m[3][2] = -1.0f / (eye - center).length();//-0.05;//-1.0f / (eye - center).length();

	return m;
}

//Look At Matrix
/*
y
|
|   .P
O---------x
/
/
z

vec3 OP = (P - O) = (eye - center)

Transform

1  0  0  PO.x
0  1  0  PO.y
0  0  1  PO.z
0  0  0  1

Model View

x0 x1 x2 0      OP.x
y0 y1 y2 0		OP.y
z0 z1 z2 0		OP.z
0  0  0	 1		0



To get the coordinate of P, we project OP on each coordinate
in other words it is dot product between each axis and OP

why not bake model_view and transform into on matrix?
(they do not conflict with each other)

because we have to first transform it to the new origin, then transform
the newly create vector in to new coordinate

*/

Matrix lookat(vec3 eye, vec3 center, vec3 up)
{
	vec3 z = (eye - center).norm(); // OP vector
	vec3 x = (up.cross(z)).norm();
	vec3 y = (z.cross(x)).norm();
	Matrix model_view = identity_matrix();
	Matrix transform = identity_matrix();

	for (int i = 0; i < 3; ++i)
	{
		model_view[0][i] = x[i];
		model_view[1][i] = y[i];
		model_view[2][i] = z[i];

		model_view[i][3] = -center[i];
		//transform[i][3] = -center[i];
	}

	//model_view = model_view * transform;

	return model_view;
	/*model_view = {
	{x[0], x[1], x[2], 0},
	{y[0], y[1], y[2], 0}
	{z[0], z[1], z[2], 0}
	{  0 ,    0,    0, 1}
	};

	//convert P to OP vector by minus P from O or "eye"
	transform = {
	{1, 0, 0, -eye[0]}
	{0, 1, 0, -eye[1]}
	{0, 0, 1, -eye[2]}
	{0, 0, 0,  1}
	};*/
}

//viewport matrix
/*
given a viewport w, h, d start at [x, y], create a matrix that transform point
(a, b, c) to viewport w, h, with far plane = d

matrix:
w/2  0  0    x + w/2     a
0  h/2  0    y + h/2     b
0  0    d/2  d/2         c
0  0    0    1           1

wa / 2 + x + w / 2 =    x + w * (a + 1) / 2

hb / 2 + y + h / 2 =    y + h * (b + 1) / 2

dc/2 + d/2        =    d * (c + 1) / 2

1                 =   1
*/
//float depth;
Matrix viewport(int x, int y, int w, int h, int depth)
{
	Matrix v = identity_matrix(4);

	v[0][3] = x + w / 2.0f;
	v[1][3] = y + h / 2.0f;
	v[2][3] = depth / 2.0f;

	v[0][0] = w / 2.0f;
	v[1][1] = h / 2.0f;
	v[2][2] = depth / 2.0f;

	return v;
}
#endif // !_MATRIX_H_


