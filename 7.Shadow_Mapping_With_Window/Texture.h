#ifndef _TEXTURE_H_
#define _TEXTURE_H_
#include "maths.h"
//#include "maths_linear.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define i255 1 / 255
#include <iostream>
#include <string.h>

using namespace std;

static int clamp_int_texture(const int& i, const int& a, const int& b)
{
	return min(max(a, i), b);
}

struct Texture
{
	int w;
	int h;
	int n;

	float* c;

	Texture() {}
	Texture(string file_name, bool use_gammar_correction = true, bool is_multiply_255 = true, bool is_normal_map = false)
	{
		c = stbi_loadf(file_name.c_str(), &w, &h, &n, 0);

		//cout << file_name.c_str() << " " << n << "\n";

		//invert texture y value
		//flip_vertical();
		//for (int i = 1; i < w*h*n; i += n)
		//	c[i] = 1.0f - c[i];

		if (use_gammar_correction)
			for (int i = 0; i < w * h * n; ++i)
				c[i] = powf(c[i], 1.0f / 2.2f);

		if (is_multiply_255)
			for (int i = 0; i < w * h * n; ++i)
				c[i] *= 255.0f;

		if (is_normal_map)
			for (int i = 0; i < w * h * n; ++i)
				c[i] = 2.0f * c[i] - 1.0f;
	}

	void flip_vertical()
	{
		int half = h >> 1;

		for (int i = 0; i < w; ++i)
		{
			for (int j = 0; j < half; ++j)
			{
				//for (int k = 0; k < n; ++k)
				{
					swap(c[n * (i + j * w)],	 c[n * (i + (h - 1 - j) * w)]);
					swap(c[n * (i + j * w) + 1], c[n * (i + (h - 1 - j) * w) + 1]);
					swap(c[n * (i + j * w) + 2], c[n * (i + (h - 1 - j) * w) + 1]);
				}
			}
		}
	}

	vec3 __fastcall ev(const vec2& t)
	{
		//float u = t.x - floorf(t.x), v = t.y - floorf(t.y);


		int x = clamp_int_texture((t.x * w), 0, w - 1),y = clamp_int_texture(((1.0f - t.y) * h), 0, h - 1);

		//int x = t.x * w, y = (1.0f - t.y) * h;

		int i = n * (x + y * w);

		return vec3(c[i], c[i + 1], c[i + 2]);
	}

	void clear()
	{
		delete(c);
	}
};

//answer 2
//https://stackoverflow.com/questions/55558241/opengl-cubemap-face-order-sampling-issue
//page 253
//https://registry.khronos.org/OpenGL/specs/gl/glspec46.core.pdf

int compute_cubemap_uv(vec3& direction, vec2& texcoord_st)
{
	int face_ind = -1;
	float sc = -1, tc = -1;

	float abs_x = abs(direction.x), abs_y = abs(direction.y), abs_z = abs(direction.z);

	float ma = max(abs_x, max(abs_y, abs_z));

	if (abs_x == ma)
	{
		if (direction.x > 0)
		{
			face_ind = 0;
			sc = direction.z;
			tc = direction.y;
		}
		else
		{
			face_ind = 1;
			sc = -direction.z;
			tc = direction.y;
		}
	}
	else if (abs_y == ma)
	{
		if (direction.y > 0)
		{
			face_ind = 2;
			sc = -direction.x;
			tc = -direction.z;
		}
		else
		{
			face_ind = 3;
			sc = -direction.x;
			tc = direction.z;
		}
	}
	else
	{
		if (direction.z > 0)
		{
			face_ind = 4;
			sc = -direction.x;
			tc = direction.y;
		}
		else
		{
			face_ind = 5;
			sc = direction.x;
			tc = direction.y;
		}
	}

	float inv_ma = abs(1.0f / ma);

	//sc = -sc;//*= -1.0f;
	//tc = -tc;//*= -1.0f;

	texcoord_st.x = 0.5f * (sc * inv_ma + 1.0f);
	texcoord_st.y = 0.5f * (tc * inv_ma + 1.0f);

	return face_ind;
}

/*
vec3 cube_map_sampling(vec3& direction, CubeMap* cube_map)
{
	vec2 uv;

	int face_ind = compute_cubemap_uv(direction, uv);

	return cube_map->ev(face_ind, uv);
}
*/

struct CubeMap
{
	Texture cube[6];

	CubeMap() {}
	CubeMap(string face_texture[6])
	{
		for (int i = 0; i < 6; ++i)
			cube[i] = Texture(face_texture[i]);
	}
	CubeMap(string face_texture0, string face_texture1, string face_texture2, string face_texture3, string face_texture4, string face_texture5)
	{
		cube[0] = Texture(face_texture0);
		cube[1] = Texture(face_texture1);
		cube[2] = Texture(face_texture2);
		cube[3] = Texture(face_texture3);
		cube[4] = Texture(face_texture4);
		cube[5] = Texture(face_texture5);
	}
	vec3 ev(vec3& direction)//(int& face_ind, vec2& tex_coord)
	{
		vec2 uv;
		int face_ind = compute_cubemap_uv(direction, uv);

		return cube[face_ind].ev(uv);

		//return cube[face_ind].ev(tex_coord);
	}
};

/*
struct Texture
{
	int w;
	int h;
	int n;

	//float* c;

	vector<vec3> c;

	Texture() {}
	Texture(string file_name, bool use_gammar_correction, bool is_multiply_255, bool is_normal_map)
	{

		float* d = stbi_loadf(file_name.c_str(), &w, &h, &n, 0);

		c.resize(w * h);

		//cout << file_name.c_str() << " " << n << "\n";

		//invert texture y value
		//flip_vertical();
		//for (int i = 1; i < w*h*n; i += n)
		//	c[i] = 1.0f - c[i];

		if (use_gammar_correction)
		{
			//for (int i = 0; i < w * h * n; ++i)
			//	c[i] = powf(c[i], 1.0f / 2.2f);

			if (n >= 3)
			{
				for (int i = 0; i < w * h; ++i)
				{
					c[i].x = powf(d[i * n], 1.0f / 2.2f);
					c[i].y = powf(d[i * n + 1], 1.0f / 2.2f);
					c[i].z = powf(d[i * n + 2], 1.0f / 2.2f);
				}
			}
			else
			{
				for (int i = 0; i < w * h; ++i)
				{
					c[i].x = powf(d[i * n], 1.0f / 2.2f);
					c[i].y = powf(d[i * n], 1.0f / 2.2f);
					c[i].z = powf(d[i * n], 1.0f / 2.2f);
				}
			}
		}
		if (is_multiply_255)
			for (int i = 0; i < w * h; ++i)
				c[i] *= 255.0f;

		if (is_normal_map)
			for (int i = 0; i < w * h; ++i)
				c[i] = 2.0f * c[i] - vec3(1.0f);

		delete(d);
	}

	void flip_vertical()
	{
		int half = h >> 1;

		for (int i = 0; i < w; ++i)
		{
			for (int j = 0; j < half; ++j)
			{
				//for (int k = 0; k < n; ++k)
				{
					swap(c[(i + j * w)], c[(i + (h - 1 - j) * w)]);
					swap(c[(i + j * w) + 1], c[(i + (h - 1 - j) * w) + 1]);
					swap(c[(i + j * w) + 2], c[(i + (h - 1 - j) * w) + 1]);
				}
			}
		}
	}

	vec3 __fastcall ev(const vec2& t)
	{
		//float u = t.x - floorf(t.x), v = t.y - floorf(t.y);


		//int x = clamp_int_texture((t.x * w), 0, w - 1),y = clamp_int_texture(((1.0f - t.y) * h), 0, h - 1);

		int x = t.x * w, y = (1.0f - t.y) * h;

		int i = (x + y * w);

		return c[i];
		//return vec3(c[i], c[i + 1], c[i + 2]);
	}

	void clear()
	{
		vector<vec3>().swap(c);
		//delete(c);
	}
};
*/

#endif // !_TEXTURE_H_

