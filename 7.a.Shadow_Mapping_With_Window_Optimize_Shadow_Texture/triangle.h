#include "line.h"
#include "mesh.h"
#include "Rnd.h"
#include "Texture.h"

vec3 red(255, 0, 0);
vec3 green(0, 255, 0);
vec3 white(255, 255, 255);

vec3 barycentric(vec2i points[3], vec2i P)
{
	vec3 bary = cross(vec3(points[1].x - points[0].x, points[2].x - points[0].x, points[0].x - P.x),
		vec3(points[1].y - points[0].y, points[2].y - points[0].y, points[0].y - P.y));

	if (abs(bary.z) < 1)
		return vec3(-1, 1, 1);

	float inv_z = 1.0f / bary.z;

	return vec3(bary.x * inv_z, bary.y * inv_z, 1.0f - (bary.x + bary.y) * inv_z);
}

vec3 barycentricf(vec3 points[3], vec3 P)
{
	vec3 u = cross(vec3(points[1][0] - points[0][0], points[2][0] - points[0][0], points[0][0] - P[0]), vec3(points[1][1] - points[0][1], points[2][1] - points[0][1], points[0][1] - P[1]));

	if (abs(u.z) < 1)
		return vec3(-1, 1, 1);

	float inv_z = 1.0f / u.z;

	return vec3(u.x * inv_z, u.y * inv_z, 1.0f - (u.x + u.y) * inv_z);
}

void triangle_bary_centric(vec2i points[3], image& img, vec3 color)
{
	int w1 = img.w - 1;
	int h1 = img.h - 1;

	vec2i bmin(w1, h1);
	vec2i bmax(0, 0);

	for (int i = 0; i < 3; ++i)
	{
		bmin.x = max(0, min(bmin.x, points[i].x));
		bmin.y = max(0, min(bmin.y, points[i].y));

		bmax.x = min(w1, max(bmax.x, points[i].x));
		bmax.y = min(h1, max(bmax.y, points[i].y));
	}

	vec2i P;

	for (P.x = bmin.x; P.x <= bmax.x; ++P.x)
	{
		for (P.y = bmin.y; P.y <= bmax.y; ++P.y)
		{
			vec3 b = barycentric(points, P);

			if (b.x < 0 || b.y < 0 || b.z < 0)
				continue;
			img.set(P.x, P.y, color);
		}
	}
}

void triangle_bary_centric_depth(vec3 points[3], vector<float>& z_buffer, image& img, vec3 color)
{
	float w = img.w - 1;
	float h = img.h - 1;

	vec2 bmin(w, h);
	vec2 bmax(0, 0);

	for (int i = 0; i < 3; ++i)
	{
		bmin.x = max(0.0f, min(bmin.x, points[i].x));
		bmin.y = max(0.0f, min(bmin.y, points[i].y));

		bmax.x = min(w, max(bmax.x, points[i].x));
		bmax.y = min(h, max(bmax.y, points[i].y));
	}

	vec3 P;

	for (P.x = bmin.x; P.x <= bmax.x; ++P.x)
	{
		for (P.y = bmin.y; P.y <= bmax.y; ++P.y)
		{
			vec3 coord = barycentricf(points, P);

			if (coord.x < 0.0f || coord.y < 0.0f || coord.z < 0.0f)
				continue;

			P.z = 0;
			for (int i = 0; i < 3; ++i)
				P.z += points[i].z * coord[i];

			if (z_buffer[int(P.x + P.y * w)] < P.z)
			{
				z_buffer[int(P.x + P.y * w)] = P.z;
				img.set(P.x, P.y, color);
			}
		}
	}
}

void triangle_bary_centric_depth_texture(vec3 points[3], vec2 uv[3], vector<float>& z_buffer, image& img, Texture& tex, float intensity)
{
	float w = img.w - 1;
	float h = img.h - 1;

	vec2 bmin(w, h);
	vec2 bmax(0, 0);

	for (int i = 0; i < 3; ++i)
	{
		bmin.x = max(0.0f, min(bmin.x, points[i].x));
		bmin.y = max(0.0f, min(bmin.y, points[i].y));

		bmax.x = min(w, max(bmax.x, points[i].x));
		bmax.y = min(h, max(bmax.y, points[i].y));
	}

	vec3 P;

	for (P.x = bmin.x; P.x <= bmax.x; ++P.x)
	{
		for (P.y = bmin.y; P.y <= bmax.y; ++P.y)
		{
			vec3 coord = barycentricf(points, P);

			if (coord.x < 0.0f || coord.y < 0.0f || coord.z < 0.0f)
				continue;

			P.z = 0;
			for (int i = 0; i < 3; ++i)
				P.z += points[i].z * coord[i];

			if (z_buffer[int(P.x + P.y * w)] < P.z)
			{
				vec2 texcoord = uv[1] * coord[0] + uv[2] * coord[1] + uv[0] * coord[2];

				vec3 color = tex.ev(texcoord) * intensity;

				z_buffer[int(P.x + P.y * w)] = P.z;
				img.set(P.x, P.y, color);
			}

		}
	}
}

