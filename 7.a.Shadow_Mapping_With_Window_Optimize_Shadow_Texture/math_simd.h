#pragma once
//#include "FastPix3D.h"
#include <fvec.h>
#include <stdio.h>
#include <smmintrin.h>

using sbyte = signed __int8;
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
			float X;
			float Y;
			float Z;
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

	vec3 Normalize() const
	{
		return _mm_mul_ps(MM, _mm_rsqrt_ps(_mm_dp_ps(MM, MM, 0x7f)));
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
	vec3 operator *(const vec3 &other) const
	{
		return _mm_mul_ps(MM, other.MM);
	}
	vec3 operator *(float scalar) const
	{
		return _mm_mul_ps(MM, _mm_set1_ps(scalar));
	}
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
		return X == other.X && Y == other.Y && Z == other.Z;
	}
	bool operator !=(const vec3 &other) const
	{
		return X != other.X || Y != other.Y || Z != other.Z;
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
};