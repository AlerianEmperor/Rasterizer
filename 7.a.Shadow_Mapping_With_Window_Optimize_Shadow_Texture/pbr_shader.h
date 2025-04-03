#ifndef _PBR_H_
#define _PBR_H_
#include "shader.h"

float D_ggx(vec3& n, vec3& h, float& a)
{
	float a2 = a * a;
	float cos_nh = dot(n, h);

	float denom = cos_nh * cos_nh * (a2 - 1) + 1;

	return a2 / (pi * denom * denom);
}

float G_schlick(float& cos_nv, float& k)
{
	return cos_nv / (cos_nv * (1 - k) + k);
}

float G_Schlick_ibl(vec3& n, vec3& v, vec3& l, float& k)
{
	float cos_nv = dot(n, v);
	float cos_nl = dot(n, l);

	return G_schlick(cos_nv, k) * G_schlick(cos_nl, k);
}

vec3 Fresnel(vec3& F0, float& cos_theta)
{
	float p = 1.0f - cos_theta;
	float p2 = p * p;

	return F0 + (vec3(1.0f) - F0) * p * p2 * p2;
}

float float_aces(float& value)
{
	float a = 2.51f;
	float b = 0.03f;
	float c = 2.43f;
	float d = 0.59f;
	float e = 0.14f;
	value = (value * (a * value + b)) / (value * (c * value + d) + e);
	return clamp(value, 0, 1);
}


vec3 Reinhard_mapping(vec3& color)
{
	int i;
	for (i = 0; i < 3; i++)
	{
		color[i] = float_aces(color[i]);
		//color[i] = color[i] / (color[i] + 0.5);
		color[i] = pow(color[i], gammar_power);// 1.0f / 2.2f
	}
	return color;
}

struct PBR : Shader
{
	vec3 T;
	vec3 N;
	vec3 B;
	mat3 TBN;

	void vertex(int& face_ind)
	{
		for (int vertex_ind = 0; vertex_ind < 3; ++vertex_ind)
		{
			int face = mesh->faces[9 * face_ind + 3 * vertex_ind];
			int tex_ind = mesh->faces[9 * face_ind + 3 * vertex_ind + 1];
			int normal_ind = mesh->faces[9 * face_ind + 3 * vertex_ind + 2];

			vec3 v = mesh->vertices[face];

			vec4 homogeneous = mvp * vec4(v, 1.0f);

			homogeneous = view_port * homogeneous;

			screen_coord_w[vertex_ind] = homogeneous;

			float inv_w = 1.0f / homogeneous.w;

			inv_clip_coord_w[vertex_ind] = inv_w;

			homogeneous *= inv_w;

			world_pos[vertex_ind] = v;
			screen_coord[vertex_ind] = vec3(homogeneous.x, homogeneous.y, homogeneous.z);
			uv[vertex_ind] = mesh->texcoords[tex_ind];
			normal[vertex_ind] = mesh->normals[normal_ind];
		}

		T = mesh->tangents[face_ind];

		T = (normal_matrix * T);

		T.normalize();

		N = (normal_matrix * (normal[0] + normal[1] + normal[2])).norm();


		T = (T - dot(T, N) * N).norm();

		B = cross(N, T);

		TBN = mat3(T, B, N);

		TBN = TBN.transpose();
	}

	vec3 direct_lighting()
	{
		vec3 light_position(2, 1.5, 5);

		vec3 radiance(3);

		//vec2 texcoord = (uv[0] * barycentric_coord[0] * inv_clip_coord_w[0] +
		//	uv[1] * barycentric_coord[1] * inv_clip_coord_w[1] +
		//	uv[2] * barycentric_coord[2] * inv_clip_coord_w[2]) * equalize;

		vec2 texcoord = uv[0] * barycentric_coord[0] + uv[1] * barycentric_coord[1] + uv[2] * barycentric_coord[2];

		vec3 n = tex_normal->ev(texcoord);

		n = (TBN * n);

		n.normalize();

		//float equalize = 1.0f / (barycentric_coord[0] * inv_clip_coord_w[0] + barycentric_coord[1] * inv_clip_coord_w[1] + barycentric_coord[2] * inv_clip_coord_w[2]);

		//vec3 world_coord = (barycentric_coord[0] * world_pos[0] * inv_clip_coord_w[0] +
		//	barycentric_coord[0] * world_pos[0] * inv_clip_coord_w[0] +
		//	barycentric_coord[0] * world_pos[0] * inv_clip_coord_w[0]) * equalize;

		vec3 world_coord = barycentric_coord[0] * world_pos[0] + barycentric_coord[1] * world_pos[1] + barycentric_coord[2] * world_pos[2];

		vec3 l = (light_position - world_coord).norm();

		vec3 v = (view_position - world_coord).norm();

		vec3 h = (l + v).norm();

		float cos_ln = max(dot(l, n), 0.0f);

		if (cos_ln > 0.0f)
		{
			float cos_vn = max(dot(v, n), 0.0f);
			float cos_hn = max(dot(h, n), 0.0f);
			float cos_vh = max(dot(v, h), 0.0f);
			float roughness = tex_roughness->ev(texcoord).x;
			float metal = tex_metal->ev(texcoord).x;

			float D = D_ggx(n, h, roughness);
			float G = G_Schlick_ibl(n, v, l, roughness);

			vec3 albedo = tex_color->ev(texcoord);

			vec3 F0(0.04);

			F0 = lerp(F0, albedo, metal);

			vec3 F = Fresnel(F0, cos_vh);

			vec3 kD = (vec3(1.0f) - F) * (1.0f - metal);

			vec3 brdf = N * D * F / (4.0f * cos_ln * cos_vn + 0.0001f);

			vec3 L0 = (kD * albedo * ipi + brdf) * radiance * cos_ln;

			vec3 ambient = 0.05f * albedo;

			vec3 color = L0 + ambient;

			return Reinhard_mapping(color);
		}
	}

	vec3 fragment()
	{
		return vec3();
	}
};


#endif // !_PBR_H_
