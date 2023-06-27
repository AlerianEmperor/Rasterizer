#ifndef _GEOMETRY_H_
#define _GEOMETRY_H_
//#include "Matrix.h"

static float maxf(const float& a, const float& b)
{
	return a > b ? a : b;
}

static float minf(const float& a, const float& b)
{
	return a < b ? a : b;
}

struct vec2
{
	float x;
	float y;

	vec2() {}
	vec2(float x_, float y_) : x(x_), y(y_) {}

	vec2 operator* (const float& v) const { return vec2(x * v, y * v); }
	vec2 operator+ (const vec2& v) const { return vec2(x + v.x, y + v.y); }
	vec2 operator-(const vec2& v) const { return vec2(x - v.x, y - v.y); }
	float operator[](const int& i) const { return (&x)[i]; }
};

struct vec2i
{
	int x;
	int y;

	vec2i() {}
	vec2i(float x_, float y_) : x(x_), y(y_) {}

	vec2i operator* (const int& v) const { return vec2i(x * v, y * v); }
	vec2i operator+ (const vec2i& v) const { return vec2i(x + v.x, y + v.y); }
	vec2i operator-(const vec2i& v) const { return vec2i(x - v.x, y - v.y); }
	int operator[](const int& i) const { return (&x)[i]; }
};

struct vec3
{
	vec3() : x(0), y(0), z(0) {}
	vec3(float v_) : x(v_), y(v_), z(v_) {}
	vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
	/*vec3(Matrix m)
	{
	float inv_z = 1.0f / m[3][0];

	x = m[0][0] * inv_z;
	y = m[1][0] * inv_z;
	z = m[2][0] * inv_z;
	}*/

	float x, y, z;


	vec3 __fastcall norm() const
	{
		float l = 1 / sqrtf(x*x + y*y + z*z); return *this * l;
	}

	void __fastcall normalize() const
	{
		float l = 1 / sqrtf(x*x + y*y + z*z); *this *= l;
	}

	float _fastcall dot(const vec3& v) const
	{
		return x * v.x + y * v.y + z * v.z;
	}
	vec3 __fastcall cross(const vec3& b) const
	{
		return{ y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x };
	}
	friend vec3 __fastcall operator^(const vec3& a, const vec3& b)
	{
		return{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
	}
	float _fastcall length() const { return sqrtf(x * x + y * y + z * z); }
	float _fastcall length2() const { return x * x + y * y + z * z; }

	float operator[](const int& i) const { return (&x)[i]; }


	friend vec3 __fastcall operator+(const vec3& a, const vec3& b)
	{
		return{ a.x + b.x, a.y + b.y, a.z + b.z };
	}
	friend vec3 __fastcall operator-(const vec3& a, const vec3& b)
	{

		return{ a.x - b.x, a.y - b.y, a.z - b.z };
	}
	friend vec3 __fastcall operator*(const vec3& a, const vec3& b)
	{

		return{ a.x * b.x, a.y * b.y, a.z * b.z };
	}

	friend vec3 operator*=(const vec3& a, const float& v)
	{
		//float ax = a.x, ay = a.y, az = a.z;
		return{ a.x * v, a.y * v, a.z * v };
	}

	friend vec3 __fastcall operator*(const vec3& a, const float& v) { return vec3(a.x * v, a.y * v, a.z * v); }
	friend vec3 __fastcall operator*(const float& v, const vec3& a) { return vec3(a.x * v, a.y * v, a.z * v); }
	vec3 operator+=(const vec3& v) { x += v.x; y += v.y; z += v.z;  return *this; }
	vec3 operator-=(const vec3& v) { x -= v.x; y -= v.y; z -= v.z;  return *this; }
	void operator*=(const float& value) { x *= value; y *= value; z *= value; }
	void operator*=(const vec3& value) { x *= value.x; y *= value.y; z *= value.z; }

	void operator/=(const vec3& value) { x /= value.x; y /= value.y; z /= value.z; }

	float maxc() const { return maxf(x, maxf(y, z)); }//float d = max(x, y); return max(d, z); }//{ return max(max(x, y), z); }//
	float minc() const { return minf(x, minf(y, z)); }//float d = min(x, y); return min(d, z); }//{ return min(min(x, y), z); }//

													  //float maxc() const { float d = max(x, y); return max(d, z); }//{ return max(max(x, y), z); }//
													  //float minc() const { float d = min(x, y); return min(d, z); }//{ return min(min(x, y), z); }//

	friend bool operator==(const vec3& a, const vec3& b)
	{
		return a.x == b.x && a.y == b.y && a.z == b.z;
	}

	float average() const
	{
		return (x + y + z) * 0.3333f;
		//return (x + y + z) / 3.0f;
	}

	friend vec3 operator/(const vec3& a, const vec3& b) { return{ a.x / b.x, a.y / b.y, a.z / b.z }; }
	friend vec3 operator/=(const vec3& a, const float& v) { return{ a.x / v, a.y / v, a.z / v }; }
	friend vec3 operator-(const vec3& a) { return{ -a.x, -a.y, -a.z }; }

	friend vec3 operator/(const vec3& a, const float& v) { return{ a.x / v, a.y / v, a.z / v }; }
	friend vec3 operator/(const vec3& a, const int& v) { return{ a.x / v, a.y / v, a.z / v }; }


	bool all_zero()
	{
		return x == y == z == 0.0f;
	}
};

struct vec3i
{
	int x;
	int y;
	int z;

	vec3i() {}
	vec3i(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
	/*vec3i(Matrix m)
	{
	float inv_z = 1.0f / m[3][0];

	x = m[0][0] * inv_z;
	y = m[1][0] * inv_z;
	z = m[2][0] * inv_z;
	}*/
	int operator[](const int& i) const { return (&x)[i]; }

	friend vec3i __fastcall operator+(const vec3i& a, const vec3i& b)
	{
		return{ a.x + b.x, a.y + b.y, a.z + b.z };
	}
	friend vec3i __fastcall operator-(const vec3i& a, const vec3i& b)
	{
		return{ a.x - b.x, a.y - b.y, a.z - b.z };
	}
	friend vec3i __fastcall operator*(const vec3i& a, const float& v) { return vec3i(a.x * v, a.y * v, a.z * v); }
	friend vec3i __fastcall operator*(const float& v, const vec3i& a) { return vec3i(a.x * v, a.y * v, a.z * v); }
};

#endif // !_GEOMETRY_H_
