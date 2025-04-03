#ifndef _MATHS_H_
#define _MATHS_H_
#include <math.h>
#include <array>
#include <omp.h>
#include <smmintrin.h>

using namespace std;

const float pi = 3.1415926;
const float gammar_power = 1.0f / 2.2f;
const float ipi = 1.0f / pi;
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

float minf(const float& a, const float& b) { return a < b ? a : b; }
float maxf(const float& a, const float& b) { return a > b ? a : b; }

void minf2(int& a, float& b) { a = a < b ? a : b; }
void maxf2(int& a, float& b) { a = a > b ? a : b; }

float max(float a, float b) { return a > b ? a : b; }
float min(float a, float b) { return a < b ? a : b; }
int max(int a, int b) { return a > b ? a : b; }
int min(int a, int b) { return a < b ? a : b; }

float clamp(float& v, const float& low, const float& high)
{
	return max(low, min(v, high));
}

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
	friend vec2 operator*(vec2& v, const float& a) { return vec2(a * v.x, a * v.y); }
	friend vec2 operator/(vec2& v, float& a) { float inv_a = 1.0f / a; return vec2(v.x * inv_a, v.y * inv_a); }

	float dot(vec2& v) { return x * v.x + y * v.y; }

	float length2() { return x * x + y * y; }
	float length() { return sqrt14(x * x + y * y); }//sqrt14
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
/*
float Q_rsqrt(float& number)
{
	long i;
	float x2, y;
	//const float threehalfs = 1.5F;

	x2 = number * 0.5F;
	y = number;
	i = *(long *)&y;                       // evil floating point bit level hacking
	i = 0x5f3759df - (i >> 1);               // what the fuck?
	y = *(float *)&i;
	y = y * (1.5f - (x2 * y * y));   // 1st iteration
										   //	y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

	return y;
}
*/

float Q_rsqrt(float& y)
{
	long i;
	float x2;
	//const float threehalfs = 1.5F;

	x2 = y * 0.5f;
	//y = number;
	i = *(long *)&y;                       // evil floating point bit level hacking
	i = 0x5f3759df - (i >> 1);               // what the fuck?
	y = *(float *)&i;
	y *= (1.5f - (x2 * y * y));   // 1st iteration
									 //	y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

	return y;
}

struct vec3
{
	union
	{
		struct
		{
			float x, y, z;
		};
		float e[3];
		//array<float, 3> e;
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
	void __fastcall normalize() { float l = 1.0f / sqrt14(x*x + y*y + z*z); *this *= l; }//

	//vec3 __fastcall norm() { float l = (x*x + y*y + z*z); l = Q_rsqrt(l); return *this * l; }
	//void __fastcall normalize() { float l = (x*x + y*y + z*z); l = Q_rsqrt(l); *this *= l; }

};

float __fastcall dot(vec3& v1, vec3& v2) { return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]; }
vec3 __fastcall cross(vec3& v1, vec3& v2) { return{ v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x }; }


/*using sbyte = signed __int8;
using int16 = signed __int16;
using int32 = signed __int32;
using int64 = signed __int64;
using byte = unsigned __int8;
using uint16 = unsigned __int16;
using uint32 = unsigned __int32;
using uint64 = unsigned __int64;

#define property_get(type, name)												\
	__declspec(property(get = get_##name))										\
	type name;																	\
	type get_##name() const

#define property_getset(type, name)												\
	__declspec(property(get = get_##name, put = set_##name))					\
	type name;																	\
	type get_##name() const

#define property_set(type, name)												\
	void set_##name(type value)

_MM_ALIGN16 struct vec3
{
	union
	{
		struct
		{
			float x;
			float y;
			float z;
		};
		struct
		{
			float e[3];
		};
		__m128 MM;
	};

	property_get(float, Length)
	{
		return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(MM, MM, 0x71)));
	}
	property_get(float, InverseLength)
	{
		return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_dp_ps(MM, MM, 0x71)));
	}

	vec3() :
		MM(_mm_setzero_ps())
	{
	}
	vec3(__m128 mm) :
		MM(mm)
	{
	}
	explicit vec3(float uniform) :
		MM(_mm_set_ps(0, uniform, uniform, uniform))
	{
	}
	explicit vec3(float x, float y, float z) :
		MM(_mm_set_ps(0, z, y, x))
	{
	}

	float& operator[](int i) { return e[i]; }

	vec3 norm()
	{
		return _mm_mul_ps(MM, _mm_rsqrt_ps(_mm_dp_ps(MM, MM, 0x7f)));
	}
	void normalize()
	{
		MM = _mm_mul_ps(MM, _mm_rsqrt_ps(_mm_dp_ps(MM, MM, 0x7f)));
	}
	float DotProduct(const vec3 &other) const
	{
		return _mm_cvtss_f32(_mm_dp_ps(MM, other.MM, 0x71));
	}
	vec3 CrossProduct(const vec3 &other) const
	{
		return _mm_sub_ps(
			_mm_mul_ps(_mm_shuffle_ps(MM, MM, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(other.MM, other.MM, _MM_SHUFFLE(3, 1, 0, 2))),
			_mm_mul_ps(_mm_shuffle_ps(MM, MM, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(other.MM, other.MM, _MM_SHUFFLE(3, 0, 2, 1)))
		);
	}

	vec3 operator +(const vec3 &other) const
	{
		return _mm_add_ps(MM, other.MM);
	}
	vec3 operator -(const vec3 &other) const
	{
		return _mm_sub_ps(MM, other.MM);
	}
	vec3 operator -() const
	{
		return _mm_sub_ps(_mm_setzero_ps(), MM);
	}
	vec3 operator *(const vec3 &other) 
	{
		return _mm_mul_ps(MM, other.MM);
	}
	vec3 operator *(float scalar) 
	{
		return _mm_mul_ps(MM, _mm_set1_ps(scalar));
	}

	friend vec3 operator*(const float& a, vec3& v) { return _mm_mul_ps(_mm_set1_ps(a), v.MM);}
	friend vec3 operator*(vec3& v, const float& a) { return _mm_mul_ps(_mm_set1_ps(a), v.MM);}

	vec3 operator /(const vec3 &other) const
	{
		return _mm_div_ps(MM, other.MM);
	}
	vec3 operator /(float scalar) const
	{
		return _mm_div_ps(MM, _mm_set1_ps(scalar));
	}
	vec3& operator +=(const vec3 &other)
	{
		MM = _mm_add_ps(MM, other.MM);
		return *this;
	}
	vec3& operator -=(const vec3 &other)
	{
		MM = _mm_sub_ps(MM, other.MM);
		return *this;
	}
	vec3& operator *=(const vec3 &other)
	{
		MM = _mm_mul_ps(MM, other.MM);
		return *this;
	}
	vec3& operator *=(float scalar)
	{
		MM = _mm_mul_ps(MM, _mm_set1_ps(scalar));
		return *this;
	}
	vec3& operator /=(const vec3 &other)
	{
		MM = _mm_div_ps(MM, other.MM);
		return *this;
	}
	vec3& operator /=(float scalar)
	{
		MM = _mm_div_ps(MM, _mm_set1_ps(scalar));
		return *this;
	}
	bool operator ==(const vec3 &other) const
	{
		return x == other.x && y == other.y && z == other.z;
	}
	bool operator !=(const vec3 &other) const
	{
		return x != other.x || y != other.y || z != other.z;
	}
	void* operator new[](size_t size)
	{
		return _aligned_malloc(size, 16);
	}
		void operator delete[](void *ptr)
	{
		if (ptr)
		{
			_aligned_free(ptr);
		}
	}

	float length() { return _mm_cvtss_f32(_mm_dp_ps(MM, MM, 0x71)); }
};


float dot(vec3& v1, vec3 &other) 
{
	return _mm_cvtss_f32(_mm_dp_ps(v1.MM, other.MM, 0x71));
}
vec3 cross(vec3& v1, vec3 &other) 
{
	return _mm_sub_ps(
		_mm_mul_ps(_mm_shuffle_ps(v1.MM, v1.MM, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(other.MM, other.MM, _MM_SHUFFLE(3, 1, 0, 2))),
		_mm_mul_ps(_mm_shuffle_ps(v1.MM, v1.MM, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(other.MM, other.MM, _MM_SHUFFLE(3, 0, 2, 1)))
	);
}
*/
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

	friend vec3i operator+(vec3i& v1, vec3i& v2) { return vec3i(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z); }
	friend vec3i operator*(vec3i& v, const float& a) { return vec3i(a * v.x, a * v.y, a * v.z); }
	vec3i operator-=(vec3i& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }

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
		//array<float, 4> e;
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
	friend vec4 operator*(const float& a, vec4& v) { return vec4(a * v.x, a * v.y, a * v.z, a * v.w); }
	friend vec4 operator*(vec4& v, const float& a) { return vec4(a * v.x, a * v.y, a * v.z, a * v.w); }
	friend vec4 operator/(vec4& v, float& a) { float inv_a = 1.0f / a; return vec4(v.x * inv_a, v.y * inv_a, v.z * inv_a, v.w * inv_a); }
};

float __fastcall dot(vec4& v1, vec3& v2) { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }

vec3 vec4_to_vec3(vec4& v)
{
	return vec3(v.x, v.y, v.z);
}

vec3 lerp(vec3& a, vec3& b, float& t)
{
	return a + (b - a) * t;
}

struct mat3
{
	//vec3 row[3];
	union
	{
		vec3 m[3] = { vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0) };

		struct
		{
			float e[9];
		};
	};
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

	friend vec3 operator*(mat3& m1, vec3& v)
	{
		vec3 result;
		/*for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
		//for(int i = 3; i--;)
		//	for(int j = 3; j--;)
			result[i] += m[i][j] * v[j];
		*/

		/*for (int i = 0; i < 3; ++i)
		{
			result[i] += m[i][0] * v[0];
			result[i] += m[i][1] * v[1];
			result[i] += m[i][2] * v[2];

			//result[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
		}*/

		for (int i = 0; i < 3; ++i)
			result[i] = m1.e[3 * i] * v[0] + m1.e[3 * i + 1] * v[1] + m1.e[3 * i + 2] * v[2];
		

		/*result[0] += m[0][0] * v[0];
		result[0] += m[0][1] * v[1];
		result[0] += m[0][2] * v[2];

		result[1] += m[1][0] * v[0];
		result[1] += m[1][1] * v[1];
		result[1] += m[1][2] * v[2];

		result[2] += m[2][0] * v[0];
		result[2] += m[2][1] * v[1];
		result[2] += m[2][2] * v[2];*/

		

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

	union
	{
		
		vec4 m[4] = { vec4(0, 0, 0, 0), vec4(0, 0, 0, 0), vec4(0, 0, 0, 0), vec4(0, 0, 0, 0) };
		struct
		{
			float e[16];
		};
	};
	//float m[4][4] = { {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0} };


	mat4() {};
	//mat4(vec4 v0, vec4 v1, vec4 v2, vec4 v3) { m[0] = v0; m[1] = v1; m[2] = v2; m[3] = v3; }
	mat4(vec4 v0, vec4 v1, vec4 v2, vec4 v3) 
	{ 
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
		m[0] = v0;
		m[1] = v1; 
		m[2] = v2; 
		m[3] = v3; 
	}

	//vec4& operator[](int i) { return vec4(m[i][0], m[i][1], m[i][2], m[i][3]); }

	//vec4& operator[](int i) { return vec4(m[0][i], m[1][i], m[2][i], m[3][i]); }

	vec4& operator[](const int& i) { return m[i]; }

	//float& operator[](const int& i) { return e[i]; }

	friend mat4 operator*(mat4& m1, float& v)
	{
		mat4 result;
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				result[i][j] = m1[i][j] * v;
		return result;
	}
	friend vec4 operator*(mat4& m1, vec4& v)
	{
		vec4 result;

		/*for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				result[i] += m1[i][j] * v[j];*/

		//#pragma omp parallel for schedule(guided)
		//for (int i = 0; i < 4; ++i)
		for (int i = 4; i--;)
			result[i] = m1.e[4 * i] * v[0] + m1.e[4 * i + 1] * v[1] + m1.e[4 * i + 2] * v[2] + m1.e[4 * i + 3] * v[3];

		//result[0] = m1.e[0] * v[0] + m1.e[1] * v[1] + m1.e[2] * v[2] + m1.e[3] * v[3];
		//result[1] = m1.e[4] * v[0] + m1.e[5] * v[1] + m1.e[6] * v[2] + m1.e[7] * v[3];
		//result[2] = m1.e[8] * v[0] + m1.e[9] * v[1] + m1.e[10] * v[2] + m1.e[11] * v[3];
		//result[3] = m1.e[12] * v[0] + m1.e[13] * v[1] + m1.e[14] * v[2] + m1.e[15] * v[3];

		/*for (int i = 0; i < 4; ++i)
		{
			result[i] += m1[i][0] * v[0];
			result[i] += m1[i][1] * v[1];
			result[i] += m1[i][2] * v[2];
			result[i] += m1[i][3] * v[3];

		}*/

		//result[i] = m1[i][0] * v[0] + m1[i][1] * v[1] + m1[i][2] * v[2] + m1[i][3] * v[3];
		
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
					result.e[4 * i + j] += m1.e[4 * i + k] * m2.e[4 * k + j];
					//result[i][j] += m1[i][k] * m2[k][j];
		return result;
	}
};

mat4 transpose(mat4& m)
{
	mat4 trans;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			//trans.e[4 * i + j] = e[4 * j + i];
			trans[i][j] = m[j][i];
	return trans;
}
//minor cofactor 3D math primer for graphics and game development, page 185 / 845 on search tab 
float minor(mat4& mat, int& r, int& c)
{
	//cut down matrix
	mat3 result;

	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
		{
			int new_r = i < r ? i : i + 1;
			int new_c = j < c ? j : j + 1;

			result[i][j] = mat[new_r][new_c];
		}
	return determine(result);
}
float cofactor(mat4& mat, int& r, int& c)
{
	int sign = (r + c) & 1 ? -1 : 1;

	float minor_det = minor(mat, r, c);

	return sign * minor_det;
}

mat4 adjoint(mat4& mat)
{
	mat4 result;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			result[i][j] = cofactor(mat, i, j);
	return result;
}

//page 186 / 845
mat4 inverse_transpose(mat4& m)
{
	//M = sum(m[i][j] * C[i][j]) = sum(m[i][j] * (-1)^(i + j) * det[i][j])

	mat4 adj = adjoint(m);

	float det = 0;
	for (int j = 0; j < 4; ++j)
		det += m[0][j] * adj[0][j];

	float inv_det = 1.0f / det;
	mat4 result;
	for (int i = 0; i < 16; ++i)
		result.e[i] = adj.e[i] * inv_det;
	return result;

	//result.e[i] = adj.e[i] * inv_det;
	//return result;
}
mat4 inverse(mat4& m)
{
	return transpose(inverse_transpose(m));
}

mat3 mat4_to_mat3(mat4& m4)
{
	mat3 m3;

	m3.m[0] = vec3(m4[0][0], m4[0][1], m4[0][2]);
	m3.m[1] = vec3(m4[1][0], m4[1][1], m4[1][2]);
	m3.m[2] = vec3(m4[2][0], m4[2][1], m4[2][2]);
	

	//mat3 m3(vec3(m4[0][0], m4[0][1], m4[0][2]), vec3(m4[1][0], m4[1][1], m4[1][2]), vec3(m4[2][0], m4[2][1], m4[2][2]));
	return m3;
}

//----------------MODEL MATRIX----------------------

mat4 mat4_identity()
{
	mat4 m;

	m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.0f;

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

//https://en.wikipedia.org/wiki/Rotation_matrix
mat4 rotate_x(float angle)
{
	mat4 m = mat4_identity();

	angle *= angle_to_radian;

	float c = cosf(angle);
	float s = sinf(angle);

	//row wise access is faster
	m[1][1] = c; m[1][2] = -s;
	m[2][1] = s; m[2][2] = c;

	return m;
}

mat4 rotate_y(float angle)
{
	mat4 m = mat4_identity();

	angle *= angle_to_radian;

	float c = cosf(angle);
	float s = sinf(angle);

	m[0][0] = c; m[0][2] = s;
	m[2][0] = -s; m[2][2] = c;

	return m;
}

mat4 rotate_z(float angle)
{
	mat4 m = mat4_identity();

	angle *= angle_to_radian;

	float c = cosf(angle);
	float s = sinf(angle);

	m[0][0] = c; m[0][1] = -s;
	m[1][0] = s; m[1][1] = c;

	return m;
}

/*mat4 transpose(mat4 &m)
{
	mat4 trans;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			trans[i][j] = m[j][i];
	return trans;
}*/

mat4 rotate(float angle, float x, float y, float z)
{
	mat4 result;

	const float x2 = x * x;
	const float y2 = y * y;
	const float z2 = z * z;
	float rads = float(angle) * 0.0174532925f;
	const float c = cosf(rads);
	const float s = sinf(rads);
	const float omc = 1.0f - c;

	result[0] = vec4(x2 * omc + c, y * x * omc + z * s, x * z * omc - y * s, 0);
	result[1] = vec4(x * y * omc - z * s, y2 * omc + c, y * z * omc + x * s, 0);
	result[2] = vec4(x * z * omc + y * s, y * z * omc - x * s, z2 * omc + c, 0);
	result[3] = vec4(0, 0, 0, 1);

	return transpose(result);
}

mat4 rotate(float x, float y, float z)
{
	return rotate(z, 0.0f, 0.0f, 1.0f) * rotate(y, 0.0f, 1.0f, 0.0f) * rotate(x, 1.0f, 0.0f, 0.0f);
}

mat4 scale(float sx, float sy, float sz)
{
	mat4 m;

	m[0][0] = sx;
	m[1][1] = sy;
	m[2][2] = sz;
	m[3][3] = 1.0f;

	return m;
}

//----------VIEW MATRIX----------------

//https://www.songho.ca/opengl/gl_camera.html
mat4 lookAt(vec3& eye, vec3& target, vec3& up)
{
	//de f nay thi phai de z xuoi, ko lay 1/z
	//vec3 f = (eye - target).norm();

	//de f nay thi lay 1 / z
	vec3 f = (target - eye).norm();
	vec3 l = cross(up, f).norm();
	vec3 u = cross(f, l);

	mat4 m;

	m[0] = vec4(l.x, l.y, l.z, -dot(l, eye));
	m[1] = vec4(u.x, u.y, u.z, -dot(u, eye));
	m[2] = vec4(f.x, f.y, f.z, -dot(f, eye));
	m[3] = vec4(0, 0, 0, 1.0f);

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
//near = 2, far = 5000, fov = 90

/*mat4 persepective_camera(vec3& position, vec3& target, const uint32_t image_height, const uint32_t image_width, float near, const float far, float fov, mat4& m_project_inv)
{
	float zoom = 1 / tan((fov / 2) * 3.1415926535897 / 180);

	float aspect = (float)image_width / (float)image_height;

	const float tan_half_fov = tan((fov / 2) * 3.1415926535897 / 180);
	const double z_range = far - near;

	mat4 m_project;

	m_project[0][0] = 1.0 / (tan_half_fov * aspect);     m_project[0][1] = 0;                                     m_project[0][2] = 0;                             m_project[0][3] = 0;
	m_project[1][0] = 0;                                 m_project[1][1] = 1.0 / tan_half_fov;                    m_project[1][2] = 0;                             m_project[1][3] = 0;
	m_project[2][0] = 0;                                 m_project[2][1] = 0;                                     m_project[2][2] = far / z_range;                 m_project[2][3] = 1;
	m_project[3][0] = 0;                                 m_project[3][1] = 0;                                     m_project[3][2] = (far * near) / -z_range;       m_project[3][3] = 0;

	m_project = transpose(m_project);

	m_project_inv = inverse(m_project);//glm::inverse(m_project);

	return m_project;
}

mat4 orthographic_camera(vec3& position, vec3& target, const uint32_t image_height, const uint32_t image_width, const float near, const float far, mat4& m_project_inv)
{
	const double z_range = far - near;

	mat4 m_project;

	m_project[0][0] = 128.0f / image_width;                m_project[0][1] = 0;                       m_project[0][2] = 0;                      m_project[0][3] = 0;
	m_project[1][0] = 0;                                     m_project[1][1] = 128.0f / image_height;      m_project[1][2] = 0;                      m_project[1][3] = 0;
	m_project[2][0] = 0;                                     m_project[2][1] = 0;                       m_project[2][2] = 1.0 / z_range;          m_project[2][3] = 0;
	m_project[3][0] = 0;                                     m_project[3][1] = 0;                       m_project[3][2] = near / z_range;       m_project[3][3] = 1;
	
	m_project = transpose(m_project);
	m_project_inv = inverse(m_project);
}
*/

//https://www.songho.ca/opengl/gl_projectionmatrix.html
//----------perspective--------------
mat4 perspective(float l, float r, float t, float b, float n, float f)
{
	mat4 m;

	/*m[0][0] = 2.0f * n / (r - l);		m[2][0] = (r + l) / (r - l);
	m[1][1] = 2.0f * n / (t - b);		m[2][1] = (t + b) / (t - b);
	m[2][2] = -(f + n) / (f - n);		m[3][2] = -2.0f * f * n / (f - n);
	m[2][3] = -1.0f;*/

	m[0][0] = 2.0f * n / (r - l);		m[0][2] = (r + l) / (r - l);
	m[1][1] = 2.0f * n / (t - b);		m[1][2] = (t + b) / (t - b);
	m[2][2] = -(f + n) / (f - n);		m[2][3] = -2.0f * f * n / (f - n);
	m[3][2] = -1.0f;

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

	m[0][0] = near / right;
	m[1][1] = near / top;
	m[2][2] = -(near + far) / (far - near);
	m[3][2] = -1;
	m[2][3] = -2.0f * near * far / (far - near);
	m[3][3] = 0;

	return m;
}

mat4 perspective_horizontal_fov(float fov, float aspect_ratio, float near, float far)
{
	float tangent = tanf(radians(fov * 0.5f));

	float right = near * tangent;
	float top = right / aspect_ratio;

	mat4 m;

	m[0][0] = near / right;
	m[1][1] = near / top;
	m[2][2] = -(near + far) / (far - near);
	m[3][2] = -1;
	m[2][3] = -2.0f * near * far / (far - near);
	m[3][3] = 0;

	return m;
}

mat4 perspective_sb7(float fov, float aspect, float n, float f)
{
	//fov = fov * angle_to_radian;

	float q = 1.0f / tanf(radians(0.5f * fov));
	float A = q / aspect;
	float B = (n + f) / (n - f);
	float C = (2.0f * n * f) / (n - f);

	mat4 result;

	result[0] = vec4(A, 0.0f, 0.0f, 0.0f);
	result[1] = vec4(0.0f, q, 0.0f, 0.0f);
	result[2] = vec4(0.0f, 0.0f, B, C);
	result[3] = vec4(0.0f, 0.0f, -1.0f, 0.0f);
	//*/

	/*result[0] = vec4(A, 0.0f, 0.0f, 0.0f);
	result[1] = vec4(0.0f, q, 0.0f, 0.0f);
	result[2] = vec4(0.0f, 0.0f, B, -1.0f);
	result[3] = vec4(0.0f, 0.0f, C, 0.0f);//*/

	return result;
}

mat4 PerspectiveCamera(const vec3& position, const vec3& target, const uint32_t image_height, const uint32_t image_width, const uint32_t near, const uint32_t far, const float fov, mat4& m_project_inv)
{
	//float fov_ = radians(fov);

	float m_zoom = 1 / tan((fov / 2) * 3.1415926535897 / 180);

	const float tan_half_fov = tan((fov / 2) * 3.1415926535897 / 180);
	const double z_range = far - near;

	float aspect = (float)image_width / (float)image_height;

	mat4 m_project = mat4_identity();

	m_project[0][0] = 1 / (tan_half_fov * aspect);     m_project[0][1] = 0;                                     m_project[0][2] = 0;                                     m_project[0][3] = 0;
	m_project[1][0] = 0;                               m_project[1][1] = 1 / tan_half_fov;                      m_project[1][2] = 0;                                     m_project[1][3] = 0;
	m_project[2][0] = 0;                               m_project[2][1] = 0;                                     m_project[2][2] = far / z_range;                        m_project[2][3] = 1;
	m_project[3][0] = 0;                               m_project[3][1] = 0;                                     m_project[3][2] = (far * near) / -z_range;       m_project[3][3] = 0;

	m_project = transpose(m_project);

	m_project_inv = inverse(m_project);

	return m_project;
}

mat4 OrthographicCamera(const vec3& position, const vec3& target, const uint32_t image_height, const uint32_t image_width, const uint32_t near, const uint32_t far)
{
	const double z_range = far - near;

	mat4 m_project;

	m_project[0][0] = 2.0 / image_width;                     m_project[0][1] = 0;                       m_project[0][2] = 0;                      m_project[0][3] = 0;
	m_project[1][0] = 0;                                     m_project[1][1] = 2.0 / image_height;      m_project[1][2] = 0;                      m_project[1][3] = 0;
	m_project[2][0] = 0;                                     m_project[2][1] = 0;                       m_project[2][2] = 1.0 / z_range;          m_project[2][3] = 0;
	m_project[3][0] = 0;                                     m_project[3][1] = 0;                       m_project[3][2] = near / z_range;       m_project[3][3] = 1;

	m_project = transpose(m_project);
	//m_project_inv = glm::inverse(m_project);

	return m_project;
}



//https://github.com/ssloy/tinyrenderer/wiki/Lesson-4:-Perspective-projection
mat4 simple_perpspective(float c)
{
	mat4 m = mat4_identity();

	m[2][3] = c;

	return m;
}

//-------ortho-------
mat4 ortho(float l, float r, float t, float b, float n, float f)//left, right, top, bot, near, far
{
	mat4 m = mat4_identity();

	/*m[0][0] = 2.0f * n / (r - l); m[0][2] = (r + l) / (r - l);
	m[1][1] = 2.0f * n / (t - b); m[1][2] = (t + b) / (t - b);
	m[2][2] = -(f + n) / (f - n); m[2][3] = -2.0f * f * n / (f - n);
	m[3][2] = -1.0f;
	m[3][3] = 0.0f;
	*/
	//m.transpose();


	//write like this can be faster, but I keep the original formula
	//float inv_rl = 1.0f / (r - l); //float inv_tb = 1.0f / (t - b);

	m[0][0] = 2.0f / (r - l); m[0][3] = -(r + l) / (r - l);
	m[1][1] = 2.0f / (t - b); m[1][3] = -(t + b) / (t - b);
	m[2][2] = -2.0f / (f - n); m[2][3] = -(f + n) / (f - n);
	m[3][3] = 1.0f;
	

	/*m[0][0] = 2.0f / (r - l); m[3][0] = -(r + l) / (r - l);
	m[1][1] = 2.0f / (t - b); m[3][1] = -(t + b) / (t - b);
	m[2][2] = 2.0f / (f - n); m[3][2] = -(f + n) / (f - n);
	m[3][3] = 1.0f;*/
	
	return m;
}

mat4 orthographic(float right, float top, float near, float far)
{
	mat4 m = mat4_identity();

	float z_range = far - near;

	m[0][0] = 1.0f / right;
	m[1][1] = 1.0f / top;
	m[2][2] = -2.0f / z_range;
	m[2][3] = -(near + far) / z_range;

	return m;
}

//------------VIEWPORT--------------------
mat4 viewport(int x, int y, int w, int h, float depth)
{
	mat4 m = mat4_identity();

	m[0][3] = x + w / 2;
	m[1][3] = y + h / 2;
	m[2][3] = 0.0f;//0.5f;//depth / 2;

	m[0][0] = w / 2;
	m[1][1] = h / 2;
	m[2][2] = 0.0f;// 0.5f;//1;////depth / 2;

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
	vec3 a = ndc_pos[0], b = ndc_pos[1], c = ndc_pos[2];
	//float signed_area = a.x * b.y - a.y * b.x +
	//	b.x * c.y - b.y * c.x +
	//	c.x * a.y - c.y * a.x;   //|AB AC|

	//float signed_area = a.x * (b.y - c.y) + a.y * ( c.x - b.x) +
	//	b.x * c.y - b.y * c.x;   


	float signed_area = a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y);

	//|AB AC|
	return signed_area <= 0;
}

/*static int is_back_facing(vec3i ndc_pos[3])
{
	vec3i a = ndc_pos[0];
	vec3i b = ndc_pos[1];
	vec3i c = ndc_pos[2];
	//float signed_area = a.x * b.y - a.y * b.x +
	//	b.x * c.y - b.y * c.x +
	//	c.x * a.y - c.y * a.x;   //|AB AC|

	//float signed_area = a.x * (b.y - c.y) + a.y * ( c.x - b.x) +
	//	b.x * c.y - b.y * c.x;   


	int signed_area = a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y);

	//|AB AC|
	return signed_area <= 0;
}*/

void clamp_vec3(vec3& c)
{
	c.x = min(255.0f, max(c.x, 0.0f));
	c.y = min(255.0f, max(c.y, 0.0f));
	c.z = min(255.0f, max(c.z, 0.0f));
}
#endif // !_MATHS_H_
