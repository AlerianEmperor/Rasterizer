#ifndef _MATHS_LINEAR_H_
#define _MATHS_LINEAR_H_
#include <math.h>
#include <vector>
#include <iostream>

using namespace std;

const float pi = 3.1415926;
const float angle_to_radian = 0.0174532f; // pi / 180.0f
const float inf = 1e20;
const float eps = 1e-4;

double inline __declspec (naked) __fastcall sqrt14(double n)
{
	_asm fld qword ptr[esp + 4]
		_asm fsqrt
	_asm ret 8
}

float radians(float angleInDegrees)
{
	return angleInDegrees * (acos(-1.0f) / 180.0);
}

float max(float a, float b) { return a > b ? a : b; }
float min(float a, float b) { return a < b ? a : b; }
int max(int a, int b) { return a > b ? a : b; }
int min(int a, int b) { return a < b ? a : b; }


struct vec2
{
	union
	{
		struct
		{
			float x, y;
		};
		float e[2];
	};

	vec2() { x = 0.0f, y = 0.0f; }
	vec2(float v) : x(v), y(v) {}
	vec2(float x_, float y_) : x(x_), y(y_) {}
	float& operator[](int i) { return e[i]; }
	vec2 operator-() { return vec2(-x, -y); }
	vec2 operator+=(vec2& v) { x += v.x; y += v.y; return *this; }
	vec2 operator-=(vec2& v) { x -= v.x; y -= v.y; return *this; }
	vec2 operator*=(vec2& v) { x *= v.x; y *= v.y; return *this; }
	vec2 operator*=(float& a) { x *= a; y *= a; return *this; }
	vec2 operator/=(float& a) { float inv_a = 1.0f / a; x *= inv_a; y *= inv_a; return *this; }

	friend vec2 operator+(vec2& v1, vec2& v2) { return vec2(v1.x + v2.x, v1.y + v2.y); }
	friend vec2 operator-(vec2& v1, vec2& v2) { return vec2(v1.x - v2.x, v1.y - v2.y); }
	friend vec2 operator*(float& a, vec2& v) { return vec2(a * v.x, a * v.y); }
	friend vec2 operator*(vec2& v, float& a) { return vec2(a * v.x, a * v.y); }
	friend vec2 operator/(vec2& v, float& a) { float inv_a = 1.0f / a; return vec2(v.x * inv_a, v.y * inv_a); }

	float length2() { return x * x + y * y; }
	float length() { return sqrt14(x * x + y * y); }
};

struct vec2i
{
	union
	{
		struct
		{
			int x, y;
		};
		int e[2];
	};

	vec2i() { x = 0, y = 0; }
	vec2i(int v) : x(v), y(v) {}
	vec2i(int x_, int y_) : x(x_), y(y_) {}
	int& operator[](int i) { return e[i]; }

	friend vec2i operator+(vec2i& v1, vec2i& v2) { return vec2i(v1.x + v2.x, v1.y + v2.y); }
	friend vec2i operator-(vec2i& v1, vec2i& v2) { return vec2i(v1.x - v2.x, v1.y - v2.y); }
	friend vec2i operator*(float& a, vec2i& v) { return vec2i(a * v.x, a * v.y); }
	friend vec2i operator*(vec2i& v, float& a) { return vec2i(a * v.x, a * v.y); }
};



struct vec3
{
	union
	{
		struct
		{
			float x, y, z;
		};
		float e[3];
	};
	vec3() { x = 0.0f, y = 0.0f, z = 0.0f; }
	vec3(float v) : x(v), y(v), z(v) {}
	vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
	float& operator[](int i) { return e[i]; }
	vec3 operator-() { return vec3(-x, -y, -z); }
	vec3 operator+=(vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
	vec3 operator-=(vec3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
	vec3 operator*=(vec3& v) { x *= v.x; y *= v.y; z *= v.z; return *this; }
	vec3 operator*=(const float& a) { x *= a; y *= a; z *= a; return *this; }
	vec3 operator/=(float& a) { float inv_a = 1.0f / a; x *= inv_a; y *= inv_a; z *= inv_a; return *this; }

	friend vec3 operator+(vec3& v1, vec3& v2) { return vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z); }
	friend vec3 operator-(vec3& v1, vec3& v2) { return vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z); }
	friend vec3 operator*(const vec3& v1, const vec3& v2) { return vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z); }
	friend vec3 operator*(const float& a, vec3& v) { return vec3(a * v.x, a * v.y, a * v.z); }
	friend vec3 operator*(vec3& v, const float& a) { return vec3(a * v.x, a * v.y, a * v.z); }
	friend vec3 operator/(vec3& v, const float& a) { float inv_a = 1.0f / a; return vec3(v.x * inv_a, v.y * inv_a, v.z * inv_a); }

	float length2() { return x * x + y * y + z * z; }
	float length() { return sqrt14(x * x + y * y + z * z); }

	vec3 __fastcall norm() { float l = 1.0f / sqrt14(x*x + y*y + z*z); return *this * l; }
	void __fastcall normalize() { float l = 1.0f / sqrt14(x*x + y*y + z*z); *this *= l; }
};

float __fastcall dot(vec3& v1, vec3& v2) { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }
vec3 __fastcall cross(vec3& v1, vec3& v2) { return{ v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x }; }

struct vec3i
{
	union
	{
		struct
		{
			int x, y, z;
		};
		int e[3];
	};
	vec3i() { x = 0, y = 0, z = 0; }
	vec3i(int v) : x(v), y(v), z(v) {}
	vec3i(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}

	int& operator[](int i) { return e[i]; }
};

struct vec4
{
	union
	{
		struct
		{
			float x, y, z, w;
		};
		float e[4];
	};

	vec4() { x = 0.0f, y = 0.0f, z = 0.0f, w = 0.0f; }
	vec4(float v) : x(v), y(v), z(v), w(v) {}
	vec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
	vec4(vec3 v, float w_) : x(v.x), y(v.y), z(v.z), w(w_) {}
	float& operator[](int i) { return e[i]; }
	vec4 operator-() { return vec4(-x, -y, -z, -w); }
	vec4 operator+=(vec4& v) { x += v.x; y += v.y; z += v.z; w += v.w; return *this; }
	vec4 operator-=(vec4& v) { x -= v.x; y -= v.y; z -= v.z; w -= v.w; return *this; }
	vec4 operator*=(vec4& v) { x *= v.x; y *= v.y; z *= v.z; w *= v.w; return *this; }
	vec4 operator*=(float& a) { x *= a; y *= a; z *= a; w *= a; return *this; }
	vec4 operator/=(float& a) { float inv_a = 1.0f / a; x *= inv_a; y *= inv_a; z *= inv_a; w *= inv_a; return *this; }

	friend vec4 operator+(vec4& v1, vec4& v2) { return vec4(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w); }
	friend vec4 operator-(vec4& v1, vec4& v2) { return vec4(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w); }
	friend vec4 operator*(float& a, vec4& v) { return vec4(a * v.x, a * v.y, a * v.z, a * v.w); }
	friend vec4 operator*(vec4& v, float& a) { return vec4(a * v.x, a * v.y, a * v.z, a * v.w); }
	friend vec4 operator/(vec4& v, float& a) { float inv_a = 1.0f / a; return vec4(v.x * inv_a, v.y * inv_a, v.z * inv_a, v.w * inv_a); }
};

float __fastcall dot(vec4& v1, vec3& v2) { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }

vec3 proj3(vec4& v)
{
	return vec3(v.x, v.y, v.z);
}

struct mat3
{
	//vec3 row[3];
	vec3 m[3] = { vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0) };
	mat3() {};
	mat3(vec3 v0, vec3 v1, vec3(v2))
	{
		m[0] = v0;
		m[1] = v1;
		m[2] = v2;
	}

	//vec3 operator[](int i) { return m[i]; }
	vec3& operator[](int i) { return m[i]; }
	//mat3 identity() { mat3 mat; mat[0][0] = 1.0f; mat[1][1] = 1.0f; mat[2][2] = 1.0f; }

	mat3 operator*(float& s)
	{
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
				m[i][j] *= s;
		return *this;
	}

	vec3 operator*(vec3& v)
	{
		vec3 result(0);
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
				result[i] += m[i][j] * v[j];
		return result;
	}

	mat3 transpose()
	{
		mat3 trans;

		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
				trans[i][j] = m[j][i];
		return trans;
	}

	//3D math primer for game development, 2nd, page 184 /845
	float determine()
	{
		return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) +
			m[0][1] * (m[1][2] * m[2][0] - m[1][0] * m[2][2]) +
			m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
	}
	//3D math primer for game development, 2nd, page 190 / 845
	//inverse(M) = adjoint(M) / determine(M)
	mat3 adjoint()
	{
		mat3 adj;

		adj[0][0] = +(m[1][1] * m[2][2] - m[2][1] * m[1][2]);
		adj[0][1] = -(m[1][0] * m[2][2] - m[2][0] * m[1][2]);
		adj[0][2] = +(m[1][0] * m[2][1] - m[2][0] * m[1][1]);
		adj[1][0] = -(m[0][1] * m[2][2] - m[2][1] * m[0][2]);
		adj[1][1] = +(m[0][0] * m[2][2] - m[2][0] * m[0][2]);
		adj[1][2] = -(m[0][0] * m[2][1] - m[2][0] * m[0][1]);
		adj[2][0] = +(m[0][1] * m[1][2] - m[1][1] * m[0][2]);
		adj[2][1] = -(m[0][0] * m[1][2] - m[1][0] * m[0][2]);
		adj[2][2] = +(m[0][0] * m[1][1] - m[1][0] * m[0][1]);

		return adj;
	}

	//inverse_transpose(M) = adjoint(M)^T / determine(M)
	//use this function for normal matrix computation
	mat3 inverse_transpose()
	{
		mat3 adj = adjoint();
		float det = determine();
		float inv_det = 1.0f / det;

		return adj * inv_det;
	}
	mat3 inverse()
	{
		return (*this).inverse_transpose().transpose();
	}
};

//3D math primer for game development, 2nd, page 184 /845
float determine(mat3& m)
{
	return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) +
		m[0][1] * (m[1][2] * m[2][0] - m[1][0] * m[2][2]) +
		m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}
//3D math primer for game development, 2nd, page 190 / 845
//inverse(M) = adjoint(M) / determine(M)
mat3 adjoint(mat3& m)
{
	mat3 adj;

	adj[0][0] = +(m[1][1] * m[2][2] - m[2][1] * m[1][2]);
	adj[0][1] = -(m[1][0] * m[2][2] - m[2][0] * m[1][2]);
	adj[0][2] = +(m[1][0] * m[2][1] - m[2][0] * m[1][1]);
	adj[1][0] = -(m[0][1] * m[2][2] - m[2][1] * m[0][2]);
	adj[1][1] = +(m[0][0] * m[2][2] - m[2][0] * m[0][2]);
	adj[1][2] = -(m[0][0] * m[2][1] - m[2][0] * m[0][1]);
	adj[2][0] = +(m[0][1] * m[1][2] - m[1][1] * m[0][2]);
	adj[2][1] = -(m[0][0] * m[1][2] - m[1][0] * m[0][2]);
	adj[2][2] = +(m[0][0] * m[1][1] - m[1][0] * m[0][1]);

	return adj;
}

struct mat4
{
	//vec4 m[4] = { vec4(0, 0, 0, 0), vec4(0, 0, 0, 0), vec4(0, 0, 0, 0), vec4(0, 0, 0, 0) };

	float m[16];// = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	//vector<float> m;

	//float m[4][4] = { {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0} };


	mat4() 
	{ 
		//m.resize(16, 0);
		m[0] = 0;
		m[1] = 0;
		m[2] = 0;
		m[3] = 0;

		m[4] = 0;
		m[5] = 0;
		m[6] = 0;
		m[7] = 0;

		m[8] = 0;
		m[9] = 0;
		m[10] = 0;
		m[11] = 0;

		m[12] = 0;
		m[13] = 0;
		m[14] = 0;
		m[15] = 0;
	};
	//mat4(vec4 v0, vec4 v1, vec4 v2, vec4 v3) { m[0] = v0; m[1] = v1; m[2] = v2; m[3] = v3; }
	mat4(vec4 v0, vec4 v1, vec4 v2, vec4 v3)
	{
		//m.resize(16);
		/*m[0][0] = v0.x; m[0][1] = v0.y; m[0][2] = v0.z; m[0][3] = v0.w;
		m[1][0] = v1.x; m[1][1] = v1.y; m[1][2] = v1.z; m[1][3] = v1.w;
		m[2][0] = v2.x; m[2][1] = v2.y; m[2][2] = v2.z; m[2][3] = v2.w;
		m[3][0] = v3.x; m[3][1] = v3.y; m[3][2] = v3.z; m[3][3] = v3.w;
		*/
		/*m[0][0] = v0.x; m[0][1] = v1.y; m[0][2] = v2.z; m[0][3] = v3.w;
		m[1][0] = v0.x; m[1][1] = v1.y; m[1][2] = v2.z; m[1][3] = v3.w;
		m[2][0] = v0.x; m[2][1] = v1.y; m[2][2] = v2.z; m[2][3] = v3.w;
		m[3][0] = v0.x; m[3][1] = v1.y; m[3][2] = v2.z; m[3][3] = v3.w;
		*/
		m[0] = v0.x; m[1] = v0.y, m[2] = v0.z, m[3] = v0.w;
		m[4] = v1.x; m[5] = v1.y, m[6] = v1.z, m[7] = v1.w;
		m[8] = v2.x; m[9] = v2.y, m[10] = v2.z, m[11] = v2.w;
		m[12] = v3.x; m[13] = v3.y, m[14] = v3.z, m[15] = v3.w;

		//m[1] = v1;
		//m[2] = v2;
		//m[3] = v3;
	}

	//vec4& operator[](int i) { return vec4(m[i][0], m[i][1], m[i][2], m[i][3]); }

	//vec4& operator[](int i) { return vec4(m[0][i], m[1][i], m[2][i], m[3][i]); }

	//vec4& operator[](const int& i) { return m[i]; }

	//float operator[](int i) { return m[i]; }

	friend mat4 operator*(mat4& m1, float& v)
	{
		mat4 result;
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				result.m[i * 4 + j] = m1.m[i * 4 + j] * v;
				//result[i][j] = m1[i][j] * v;
		return result;
	}
	friend vec4 operator*(mat4& m1, vec4& v)
	{
		vec4 result;

		/*for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
		result[i] += m1[i][j] * v[j];*/

		//#pragma omp parallel for schedule(guided)
		for (int i = 0; i < 4; ++i)
			result[i] = m1.m[4 * i] * v[0] + m1.m[4 * i + 1] * v[1] + m1.m[4 * i + 2] * v[2] + m1.m[4 * i + 3] * v[3];

		//result[0] = m1[0][0] * v[0] + m1[0][1] * v[1] + m1[0][2] * v[2] + m1[0][3] * v[3];
		//result[1] = m1[1][0] * v[0] + m1[1][1] * v[1] + m1[1][2] * v[2] + m1[1][3] * v[3];
		//result[2] = m1[2][0] * v[0] + m1[2][1] * v[1] + m1[2][2] * v[2] + m1[2][3] * v[3];
		//result[3] = m1[3][0] * v[0] + m1[3][1] * v[1] + m1[3][2] * v[2] + m1[3][3] * v[3];


		return result;
	}
	friend mat4 operator*(mat4& m1, mat4& m2)
	{
		mat4 result;
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				for (int k = 0; k < 4; ++k)
					result.m[i * 4 + j] += m1.m[i * 4 + k] * m2.m[k * 4 + j];

					//result[i][j] += m1[i][k] * m2[k][j];
		return result;
	}
	mat4 transpose()
	{
		mat4 trans;
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				trans.m[i * 4 + j] = m[j * 4 + i];
				//trans[i][j] = m[j][i];
		return trans;
	}
	//minor cofactor 3D math primer for graphics and game development, page 185 / 845 on search tab 
	float minor(mat4 mat, int r, int c)
	{
		//cut down matrix
		mat3 result;

		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
			{
				int new_r = i < r ? i : i + 1;
				int new_c = j < c ? j : j + 1;

				result.m[i * 4 + j] = mat.m[new_r * 4 + new_c];
				//result[i][j] = mat[new_r][new_c];
			}
		return determine(result);
	}
	float cofactor(mat4 mat, int r, int c)
	{
		int sign = (r + c) & 1 ? -1 : 1;

		float minor_det = minor(mat, r, c);

		return sign * minor_det;
	}

	mat4 adjoint(mat4 mat)
	{
		mat4 result;
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				result.m[i * 4 + j] = cofactor(mat, i, j);
				//result[i][j] = cofactor(mat, i, j);
		return result;
	}

	//page 186 / 845
	mat4 inverse_transpose()
	{
		//M = sum(m[i][j] * C[i][j]) = sum(m[i][j] * (-1)^(i + j) * det[i][j])

		mat4 adj = adjoint(*this);

		float det = 0;
		//for (int j = 0; j < 4; ++j)
		//	det += m[j] * adj.m[j];
			//det += m[0][j] * adj[0][j];

		float inv_det = 1.0f / det;

		mat4 result;

		//cout << result.m[0];
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				result.m[i * 4 + j] = adj.m[i * 4 + j] * inv_det;
				//result[i][j] = adj[i][j] * inv_det;
		return result;
	}
	mat4 inverse()
	{
		return inverse_transpose();
	}
};

mat3 mat4_to_mat3(mat4& m4)
{
	mat3 m3;

	m3.m[0] = vec3(m4.m[0], m4.m[1], m4.m[2]);
	m3.m[1] = vec3(m4.m[4], m4.m[5], m4.m[6]);
	m3.m[2] = vec3(m4.m[8], m4.m[9], m4.m[10]);

	return m3;
}

//----------------MODEL MATRIX----------------------

mat4 mat4_identity()
{
	mat4 m;

	//m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.0f;

	m.m[0] = m.m[5] = m.m[10] = m.m[15] = 1.0f;

	return m;
}

mat4 translate(float tx, float ty, float tz)
{
	mat4 m;

	/*m[0][3] = tx;
	m[1][3] = ty;
	m[2][3] = tz;*/

	/*m[3][0] = tx;
	m[3][1] = ty;
	m[3][2] = tz;*/

	/*m = mat4(vec4(1.0f, 0.0f, 0.0f, 0.0f),
	vec4(0.0f, 1.0f, 0.0f, 0.0f),
	vec4(0.0f, 0.0f, 1.0f, 0.0f),
	vec4(tx  , ty  , tz  , 1.0f));//*/


	m = mat4(vec4(1.0f, 0.0f, 0.0f, tx),
		vec4(0.0f, 1.0f, 0.0f, ty),
		vec4(0.0f, 0.0f, 1.0f, tz),
		vec4(0.0f, 0.0f, 0.0f, 1.0f));//*/
	return m;
}

/*
	0	1	2	3
	4	5	6	7
	8	9	10	11
	12	13	14	15
*/
//https://en.wikipedia.org/wiki/Rotation_matrix
mat4 rotate_x(float angle)
{
	mat4 m = mat4_identity();

	angle *= angle_to_radian;

	float c = cosf(angle);
	float s = sinf(angle);

	//row wise access is faster
	/*m[1][1] = c; m[1][2] = -s;
	m[2][1] = s; m[2][2] = c;*/

	m.m[5] = c;	m.m[6] = -s;
	m.m[9] = s;	m.m[10] = c;

	return m;
}

mat4 rotate_y(float angle)
{
	mat4 m = mat4_identity();

	angle *= angle_to_radian;

	float c = cosf(angle);
	float s = sinf(angle);

	//m[0][0] = c; m[0][2] = s;
	//m[2][0] = -s; m[2][2] = c;

	m.m[0] = c;	m.m[2] = s;
	m.m[8] = -s;	m.m[10] = c;

	return m;
}

mat4 rotate_z(float angle)
{
	mat4 m = mat4_identity();

	angle *= angle_to_radian;

	float c = cosf(angle);
	float s = sinf(angle);

	//m[0][0] = c; m[0][1] = -s;
	//m[1][0] = s; m[1][1] = c;

	m.m[0] = c;	m.m[1] = -s;
	m.m[4] = s;	m.m[5] = c;

	return m;
}

mat4 transpose(mat4 &m)
{
	mat4 trans;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			trans.m[4 * i + j] = m.m[4 * j + i];
			//trans[i][j] = m[j][i];
	return trans;
}

mat4 rotate(float angle, float x, float y, float z)
{
	//mat4 result;

	const float x2 = x * x;
	const float y2 = y * y;
	const float z2 = z * z;
	float rads = float(angle) * 0.0174532925f;
	const float c = cosf(rads);
	const float s = sinf(rads);
	const float omc = 1.0f - c;

	mat4 result(vec4(x2 * omc + c, y * x * omc + z * s, x * z * omc - y * s, 0),
				vec4(x * y * omc - z * s, y2 * omc + c, y * z * omc + x * s, 0),
				vec4(x * z * omc + y * s, y * z * omc - x * s, z2 * omc + c, 0),
				vec4(0, 0, 0, 1));

	return transpose(result);
}

mat4 rotate(float x, float y, float z)
{
	return rotate(z, 0.0f, 0.0f, 1.0f) * rotate(y, 0.0f, 1.0f, 0.0f) * rotate(x, 1.0f, 0.0f, 0.0f);
}

mat4 scale(float sx, float sy, float sz)
{
	mat4 m;

	//m[0][0] = sx;
	//m[1][1] = sy;
	//m[2][2] = sz;
	//m[3][3] = 1.0f;

	m.m[0] = sx;
	m.m[5] = sy;
	m.m[10] = sz;
	m.m[15] = 1.0f;


	return m;
}

//----------VIEW MATRIX----------------

//https://www.songho.ca/opengl/gl_camera.html
mat4 lookAt(vec3 eye, vec3 target, vec3 up)
{
	vec3 f = (eye - target).norm();
	vec3 l = cross(up, f).norm();
	vec3 u = cross(f, l);

	mat4 m(vec4(l.x, l.y, l.z, -dot(l, eye)),
		   vec4(u.x, u.y, u.z, -dot(u, eye)),
		   vec4(f.x, f.y, f.z, -dot(f, eye)),
		   vec4(0, 0, 0, 1.0f));

	return m;

	/*m[0] = vec4(l.x, u.x, f.x, 0.0f);
	m[1] = vec4(l.y, u.y, f.y, 0.0f);
	m[2] = vec4(l.z, u.z, f.z, 0.0f);
	m[3] = vec4(-dot(l, eye), -dot(u, eye), -dot(f, eye), 1.0f);*/

	//mat4 m;

	/*vec3 f = (target - eye);
	vec3 upN = up.norm();
	vec3 s = cross(f, upN);
	vec3 u = cross(s, f);
	mat4 M = mat4(vec4(s[0], u[0], -f[0], 0),
	vec4(s[1], u[1], -f[1], 0),
	vec4(s[2], u[2], -f[2], 0),
	vec4(0, 0, 0, 1));
	*/
	/* mat4 M = mat4(vec4(s[0], s[1], s[2], 0),
	vec4(u[0], u[1], u[2], 0),
	vec4(-f[0], -f[1], -f[2], 0),
	vec4(0, 0, 0, 1));
	*/
	//return M * translate(-eye.x, -eye.y, -eye.z);
}

//----------PROJECTION MATRIX-------------

//https://www.songho.ca/opengl/gl_projectionmatrix.html
//----------perspective--------------
mat4 perspective(float l, float r, float t, float b, float n, float f)
{
	mat4 m;

	/*m[0][0] = 2.0f * n / (r - l);		m[2][0] = (r + l) / (r - l);
	m[1][1] = 2.0f * n / (t - b);		m[2][1] = (t + b) / (t - b);
	m[2][2] = -(f + n) / (f - n);		m[3][2] = -2.0f * f * n / (f - n);
	m[2][3] = -1.0f;*/

	/*m[0][0] = 2.0f * n / (r - l);		m[0][2] = (r + l) / (r - l);
	m[1][1] = 2.0f * n / (t - b);		m[1][2] = (t + b) / (t - b);
	m[2][2] = -(f + n) / (f - n);		m[2][3] = -2.0f * f * n / (f - n);
	m[3][2] = -1.0f;
	*/

	m.m[0] = 2.0f * n / (r - l);		m.m[2] = (r + l) / (r - l);
	m.m[5] = 2.0f * n / (t - b);		m.m[6] = (t + b) / (t - b);
	m.m[10] = -(f + n) / (f - n);		m.m[11] = -2.0f * f * n / (f - n);
	m.m[14] = -1.0f;

	return m;
}

//why perspective_vertical_fov and perspective_horizontal_fov have to use -fov 
//but perspective dont have to do this

//because perspective use frustume base so it does not flip image along y axis
//but perspective_fov use pin hole model it flip image along y axis
mat4 perspective_vertical_fov(float fov, float aspect_ratio, float near, float far)
{
	float tangent = tanf(radians(fov * 0.5f));

	float top = near * tangent;
	float right = top * aspect_ratio;

	mat4 m;

	/*m[0][0] = near / right;
	m[1][1] = near / top;
	m[2][2] = -(near + far) / (far - near);
	m[3][2] = -1;
	m[2][3] = -2.0f * near * far / (far - near);
	m[3][3] = 0;
	*/

	m.m[0] = near / right;
	m.m[5] = near / top;
	m.m[10] = -(near + far) / (far - near);
	m.m[14] = -1;
	m.m[11] = -2.0f * near * far / (far - near);
	m.m[15] = 0;

	return m;
}

mat4 perspective_horizontal_fov(float fov, float aspect_ratio, float near, float far)
{
	float tangent = tanf(radians(fov * 0.5f));

	float right = near * tangent;
	float top = right / aspect_ratio;

	mat4 m;

	/*m[0][0] = near / right;
	m[1][1] = near / top;
	m[2][2] = -(near + far) / (far - near);
	m[3][2] = -1;
	m[2][3] = -2.0f * near * far / (far - near);
	m[3][3] = 0;
	*/

	m.m[0] = near / right;
	m.m[5] = near / top;
	m.m[10] = -(near + far) / (far - near);
	m.m[14] = -1;
	m.m[11] = -2.0f * near * far / (far - near);
	m.m[15] = 0;

	return m;
}

mat4 perspective_sb7(float fov, float aspect, float n, float f)
{
	//fov = fov * angle_to_radian;

	float q = 1.0f / tanf(radians(0.5f * fov));
	float A = q / aspect;
	float B = (n + f) / (n - f);
	float C = (2.0f * n * f) / (n - f);

	/*mat4 result;

	result[0] = vec4(A, 0.0f, 0.0f, 0.0f);
	result[1] = vec4(0.0f, q, 0.0f, 0.0f);
	result[2] = vec4(0.0f, 0.0f, B, C);
	result[3] = vec4(0.0f, 0.0f, -1.0f, 0.0f);
	//*/

	/*result[0] = vec4(A, 0.0f, 0.0f, 0.0f);
	result[1] = vec4(0.0f, q, 0.0f, 0.0f);
	result[2] = vec4(0.0f, 0.0f, B, -1.0f);
	result[3] = vec4(0.0f, 0.0f, C, 0.0f);//*/

	mat4 result(vec4(A, 0.0f, 0.0f, 0.0f),
		vec4(0.0f, q, 0.0f, 0.0f),
		vec4(0.0f, 0.0f, B, -1.0f),
		vec4(0.0f, 0.0f, C, 0.0f));

	return result;
}



//https://github.com/ssloy/tinyrenderer/wiki/Lesson-4:-Perspective-projection
mat4 simple_perpspective(float c)
{
	mat4 m = mat4_identity();

	//m[2][3] = c;

	return m;
}

//-------ortho-------
mat4 ortho(float l, float r, float t, float b, float n, float f)//left, right, top, bot, near, far
{
	mat4 m;

	//write like this can be faster, but I keep the original formula
	//float inv_rl = 1.0f / (r - l); //float inv_tb = 1.0f / (t - b);

	/*m[0][0] = 2.0f / (r - l); m[0][3] = -(r + l) / (r - l);
	m[1][1] = 2.0f / (t - b); m[1][3] = -(t + b) / (t - b);
	m[2][2] = 2.0f / (f - n); m[2][3] = -(f + n) / (f - n);
	m[3][3] = 1.0f;*/

	/*m[0][0] = 2.0f / (r - l); m[3][0] = -(r + l) / (r - l);
	m[1][1] = 2.0f / (t - b); m[3][1] = -(t + b) / (t - b);
	m[2][2] = 2.0f / (f - n); m[3][2] = -(f + n) / (f - n);
	m[3][3] = 1.0f;
	*/

	m.m[0] = 2.0f / (r - l); m.m[12] = -(r + l) / (r - l);
	m.m[5] = 2.0f / (t - b); m.m[13] = -(t + b) / (t - b);
	m.m[10] = 2.0f / (f - n); m.m[14] = -(f + n) / (f - n);
	m.m[15] = 1.0f;

	return m;
}


//------------VIEWPORT--------------------
mat4 viewport(int x, int y, int w, int h, float depth)
{
	mat4 m = mat4_identity();

	/*m[0][3] = x + w / 2;
	m[1][3] = y + h / 2;
	//m[2][3] = depth / 2;

	m[0][0] = w / 2;
	m[1][1] = h / 2;
	//m[2][2] = 1;////depth / 2;
	*/

	m.m[3] = x + w / 2;
	m.m[7] = y + h / 2;
	//m[2][3] = depth / 2;

	m.m[0] = w / 2;
	m.m[5] = h / 2;

	return m;
	/*
	w/2	 0	  0	  x + w / 2		a			a * w / 2 + x + w / 2 = x + (a + 1) * w / 2
	0	 h/2  0   y + h / 2     b									y + (b + 1) * h / 2
	0    0   d/2  d / 2			c									d * (c + 1) / 2  c
	0	 0   0	  1				1									1
	*/
}

static int is_back_facing(vec3 ndc_pos[3])
{
	vec3 a = ndc_pos[0];
	vec3 b = ndc_pos[1];
	vec3 c = ndc_pos[2];
	float signed_area = a.x * b.y - a.y * b.x +
		b.x * c.y - b.y * c.x +
		c.x * a.y - c.y * a.x;   //|AB AC|
	return signed_area <= 0;
}

void clamp_vec3(vec3& c)
{
	c.x = min(255.0f, max(c.x, 0.0f));
	c.y = min(255.0f, max(c.y, 0.0f));
	c.z = min(255.0f, max(c.z, 0.0f));
}
#endif // !_MATHS_H_
