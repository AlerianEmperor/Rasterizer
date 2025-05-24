#ifndef _SSAO_H_
#define _SSAO_H_

//#include "Gourad2.h"
//#include "Shadow_Mapping.h"
//#include "Gourad_Final_2.h"
//#include "Shadow_Mapping_Final.h"
#include "mesh.h"
#include "win32.h"
#include "camera.h"
#include <time.h>
#include <omp.h>
#include <iostream>

using namespace std;
bool write = false;

float DIRECTIONAL_LIGHT_DISTANCE = 1000000.0;
float SHADOW_MAP_BIAS = 7.5;

struct Scene
{
	Mesh mesh;

	//float* shadow_buffer;

	vector<float> shadow_buffer;

	int w;
	int h;

	int sw;
	int sh;

	//float inv_w;
	//float inv_h;

	float w1;
	float h1;

	vec3 eye = vec3(10.0, 10.0, 10.0);
	vec3 center = vec3(0.0, 0.0, 0.0);
	vec3 up = vec3(0.0, 1.0, 0.0);

	float angle = 0;

	float light_coeff = 20;

	//vec3 light_position = vec3(light_coeff * sinf(angle), light_coeff, light_coeff * cosf(angle));
	//vec3 light_direction = vec3(1.0f, 1.0f, -0.2f);
	//vec3 light_dir;// (0, 0, -1);

	vec3 light_position;// = vec3(1000.0, 1000.0, 0.0);
	vec3 close_light_position;
	//vec3 light_direction = vec3(1.0f, 0.4, -1.0f);
	//vec3 light_direction = vec3(1.0f, 0.9, -0.5f);
	vec3 light_direction = vec3(1.0f, 1.0f, -0.6f);

	float light_distance_scale = 0.00008f;

	float slope_factor = 0.001f;
	float base_bias = 0.0005f;

	vec3 l;
	const float DIRECTIONAL_LIGHT_DISTANCE = 1e6;

	mat4 model = mat4_identity();
	mat4 inv_model;
	
	Camera* perspective_camera = NULL;
	Camera* orthographic_camera = NULL;

	mat3 normal_matrix;

	
	Scene() {}
	Scene(string file_name, int w_, int h_)
	{
		mesh = load_obj(file_name);

		w = w_;
		h = h_;

		sw = 3 * w;
		sh = 3 * h;

		w1 = w - 1;
		h1 = h - 1;

		shadow_buffer.resize(sw * sh, 1e6);

		init_window(w, h, "render");

		light_direction.normalize();
		l = light_direction;

		light_position = vec3(light_direction.x * DIRECTIONAL_LIGHT_DISTANCE * cosf(angle), light_direction.y * DIRECTIONAL_LIGHT_DISTANCE, light_direction.z * DIRECTIONAL_LIGHT_DISTANCE * sinf(angle));

		close_light_position = light_direction * 10.0f;//(light_position - center) * light_distance_scale;//vec3(light_direction.x * 10.0f * cosf(angle), light_direction.y * 10.0f, light_direction.z * 10.0f * sinf(angle));//light_position * light_scale;


													   //model = translate(-0.3f, -0.4f, 0.0f) * scale(0.8, 0.8, 0.8) * translate(0.6, 0.7, 0.8) * rotate(12, 0.2, 0.4, 0.1);
		model = mat4_identity();

		inv_model = inverse(model);

		
		perspective_camera = new Perspective(eye, center, vec3(0, 1, 0), w, h, 0.01f, 1e2, 90.0f);

		orthographic_camera = new Orthographic(light_position, center, vec3(0, 1, 0), sw, sh, 2.0f, 2.0f * 1e6);

		perspective_camera->m_model = model;

		orthographic_camera->m_model = model;

		const float DIRECTIONAL_LIGHT_DISTANCE = 1e6;
	}

	void update_camera(Camera*& cam)
	{
		//mat4 lookat = lookAt(cam->eye, cam->center, cam->up);

		cam->m_lookat = lookAt(cam->eye, cam->center, cam->up);//lookat;
		cam->m_lookat_inv = inverse(cam->m_lookat);

		cam->m_model_inv = inv_model;

		//cam->m_model_view = cam->m_lookat * model;
	}

	void compute_barycentric(float& u, float& v, float& w, vec2 p[3], vec2& point) const
	{
		vec2 v0 = p[1] - p[0], v1 = p[2] - p[0], v2 = point - p[0];
		float d00 = dot(v0, v0);
		float d01 = dot(v0, v1);
		float d11 = dot(v1, v1);
		float d20 = dot(v2, v0);
		float d21 = dot(v2, v1);
		float inv_denom = 1.0f / (d00 * d11 - d01 * d01);
		v = (d11 * d20 - d01 * d21) * inv_denom;
		w = (d00 * d21 - d01 * d20) * inv_denom;
		u = 1.0f - v - w;
	}

	void update_light(Camera*& shadow_camera)
	{
		angle += 0.2f;

		//close_light_position.x = //light_direction.x * 10.0f * cosf(angle);
		//close_light_position.z = light_direction.z * 10.0f * sinf(angle);

		//l = close_light_position - center;

		light_position.x = light_direction.x * DIRECTIONAL_LIGHT_DISTANCE * cosf(angle);
		light_position.z = light_direction.z * DIRECTIONAL_LIGHT_DISTANCE * sinf(angle);

		vec3 l = (light_position - center).norm();

		close_light_position = l * 10.0f;

		shadow_camera->m_lookat = lookAt(light_position, shadow_camera->center, shadow_camera->up);
	}


	float calculate_depth(vec4 view_space[3], float& u, float& v)
	{
		float z = ((1.0f - u - v) * view_space[0].z + u * view_space[1].z + v * view_space[2].z);

		return 1.0f / z;
	}

	float calculate_depth(float*& camera_inv_z, float& u, float& v)
	{
		return 1.0f / (camera_inv_z[0] + u * camera_inv_z[1] + v * camera_inv_z[2]);
	}

	//vec2 rasterize(mat4& view_, mat4& projection_, vec4& world_point, vec4& camera_point)
	vec2 rasterize(Camera* camera, vec3& world_point, vec4& camera_point)
	{
		// model_point = model * vec4(world_point, 1.0f);	
		vec2 screen_point = camera->rasterize(world_point, camera_point);

		//camera_point.z = 1.0f / camera_point.z;

		return screen_point;
	}

	vec2 rasterize_2(Camera* camera, vec3& world_point, float& camera_inv_z)
	{
		vec4 model_point = model * vec4(world_point, 1.0f);
		vec4 camera_point = camera->view_transform(model_point);
		vec2 ndc_point = camera->project_transform(camera_point);
		vec2 screen_point = camera->viewport_transform(ndc_point);

		camera_inv_z = 1.0f / camera_point.z;

		return screen_point;
	}

	float in_shadow(Camera*& camera, vec3& world_point, vec3& normal)
	{
		vec4 camera_point;
		vec2 light_point = camera->rasterize(world_point, camera_point);

		uint32_t index = round(light_point.y) * sw + round(light_point.x) + 1;

		if (index < 0 || index >= sw * sh)
			return 1.0f;

		
		float shadow = 0.0f;

		float closest_depth = shadow_buffer[index];

		
		double current_depth = camera_point.z;

		
		double bias = maxf(0.05f * (1.0f - abs(dot(normal, light_direction))), 0.005f);

		
		float s = (current_depth > closest_depth + bias) ? 0.4f : 1.0f;

		
		return s;
	}

	vec3 get_light_position()
	{
		vec3 light_pos;// (light_dir.x * DIRECTIONAL_LIGHT_DISTANCE, light_dir.y * DIRECTIONAL_LIGHT_DISTANCE, light_dir.z * DIRECTIONAL_LIGHT_DISTANCE);

		return light_pos;
	}

	vec3 fragment(vec3& n)
	{
		float d = dot(n, light_direction);

		//float d = dot(n, l);

		return vec3(d);
	}

	//void __fastcall render_scene(Camera*& camera, image& img, float*& z_buffer, bool render_pass)
	void __fastcall render_scene(Camera*& camera, int w_, int h_, image& img, vector<float>& z_buffer, bool render_pass)
	{
		//int w = camera->w;
		//int h = camera->h;

		int w1 = w_ - 1;
		int h1 = h_ - 1;

		vec2i bmin(w1, h1);
		vec2i bmax(0, 0);

		for (int i = 0; i < mesh.num_triangles; ++i)
		{
			vec2* points = new vec2[3];

			//vec4 camera_point_inv_z[3];
			float* camera_point_inv_z = new float[3];

			vec3 normals[3];

			for (int j = 0; j < 3; ++j)
				//for(int j = 3; j--;)
			{
				int vertex_ind = mesh.faces[9 * i + 3 * j];
				int normal_ind = mesh.faces[9 * i + 3 * j + 2];

				vec3 v = mesh.vertices[vertex_ind];

				normals[j] = mesh.normals[normal_ind];

				
				points[j] = camera->rasterize(v, camera_point_inv_z[j]);

				//camera_point_inv_z[j].z = 1.0f / camera_point_inv_z[j].z;

				bmin.x = minf(bmin.x, points[j].x);
				bmax.x = maxf(bmax.x, points[j].x);

				bmin.y = minf(bmin.y, points[j].y);
				bmax.y = maxf(bmax.y, points[j].y);

			}



			float area = (points[0].x - points[1].x) * (points[0].y - points[2].y) + (points[2].x - points[0].x) * (points[0].y - points[1].y);

			//front face culling
			//use for shadow mapping
			if (!render_pass && area <= 0)
			{
				delete(camera_point_inv_z);
				delete(points);
				continue;
			}
			//back face culling
			//use for forward rendering
			if (render_pass && area >= 0)
			{
				delete(camera_point_inv_z);
				delete(points);
				continue;
			}

			normals[1] -= normals[0];
			normals[2] -= normals[0];

			camera_point_inv_z[1] -= camera_point_inv_z[0];
			camera_point_inv_z[2] -= camera_point_inv_z[0];

			bmin.x = maxf(0.0f, bmin.x);
			bmin.y = maxf(0.0f, bmin.y);

			bmax.x = minf(w1, bmax.x);
			bmax.y = minf(h1, bmax.y);

			float inv_area = 1.0f / area;

			const float p01x = (points[0].x - points[1].x) * inv_area;
			const float p01y = (points[0].y - points[1].y) * inv_area;

			const float p02x = (points[0].x - points[2].x) * inv_area;
			const float p02y = (points[2].y - points[0].y) * inv_area;

			
			#pragma omp parallel for schedule(guided)
			for (int y = bmin.y; y <= bmax.y; ++y)
			{
				
				float u = (p02x * (y - points[0].y) + (bmin.x - points[0].x) * p02y),
					v = ((bmin.x - points[0].x) * p01y + p01x * (points[0].y - y));

				
				bool in = false;

				for (uint32_t x = bmin.x; x <= bmax.x; ++x)
				{
					u += p02y;
					v += p01y;

					if (u < 0 || v < 0 || u + v > 1.0f)
					{
						if (in)
							break;
						continue;
					}

					in = true;
					float z = calculate_depth(camera_point_inv_z, u, v);

					int pixel_ind = y * w_ + x;
					if (z >= z_buffer[pixel_ind])
					{
						continue;
					}


					//vec2 screen_point(x, y); //= points[0] * a + points[1] * b + points[2] * c;

					//vec3 world_point = unrasterize(screen_point, z);

					//float shadow_value = point_in_shadow(world_point);

					z_buffer[pixel_ind] = z;

					if (render_pass)
					{
						
						vec3 n = normals[0] + u * normals[1] + v * normals[2];

						
						//pcf
						/*float shadow_factor = 0;// in_shadow(orthographic_camera, world_point, n);
						float d = 0;
						float ambient = 0.001f;

						for (int dx = -1; dx <= 1; ++dx)
						{
						for (int dy = -1; dy <= 1; ++dy)
						{
						vec2 screen_point(x + dx, y + dy);

						vec3 world_point = camera->unrasterize(screen_point, z);

						float shadow_value = in_shadow(orthographic_camera, world_point, n);

						shadow_factor += shadow_value;

						d += ambient + maxf(0.0f, dot(n, (close_light_position - world_point).norm()));
						}
						}

						shadow_factor *= 0.1111f;
						d *= 0.1111f;*/

						vec2 screen_point(x, y);
						float ambient = 0.001f;
						vec3 world_point = camera->unrasterize(screen_point, z);

						float d = ambient + maxf(0.0f, dot(n, (close_light_position - world_point).norm()));

						float shadow_factor = in_shadow(orthographic_camera, world_point, n);

						//img.data[pixel_ind] = 255.0f * shadow_factor * vec3(d);

						//vec3 d = fragment(n);

						img.data[pixel_ind] = 200.0f * shadow_factor * vec3(d);
					}

					//delete(camera_point_inv_z);
				}
			}
		}
	}

	void __fastcall render_scene_depth(Camera*& camera, int w_, int h_, vector<float>& z_buffer, bool render_pass)
	{
		//int w = camera->w;
		//int h = camera->h;

		int w1 = w_ - 1;
		int h1 = h_ - 1;

		vec2i bmin(w1, h1);
		vec2i bmax(0, 0);

		for (int i = 0; i < mesh.num_triangles; ++i)
		{
			vec2* points = new vec2[3];

			//vec4 camera_point_inv_z[3];
			float* camera_point_inv_z = new float[3];

			vec3 normals[3];

			for (int j = 0; j < 3; ++j)
				//for(int j = 3; j--;)
			{
				int vertex_ind = mesh.faces[9 * i + 3 * j];
				int normal_ind = mesh.faces[9 * i + 3 * j + 2];

				vec3 v = mesh.vertices[vertex_ind];

				normals[j] = mesh.normals[normal_ind];


				points[j] = camera->rasterize(v, camera_point_inv_z[j]);

				//camera_point_inv_z[j].z = 1.0f / camera_point_inv_z[j].z;

				bmin.x = minf(bmin.x, points[j].x);
				bmax.x = maxf(bmax.x, points[j].x);

				bmin.y = minf(bmin.y, points[j].y);
				bmax.y = maxf(bmax.y, points[j].y);

			}



			float area = (points[0].x - points[1].x) * (points[0].y - points[2].y) + (points[2].x - points[0].x) * (points[0].y - points[1].y);

			//front face culling
			//use for shadow mapping
			if (!render_pass && area <= 0)
			{
				delete(camera_point_inv_z);
				delete(points);
				continue;
			}
			//back face culling
			//use for forward rendering
			if (render_pass && area >= 0)
			{
				delete(camera_point_inv_z);
				delete(points);
				continue;
			}

			normals[1] -= normals[0];
			normals[2] -= normals[0];

			camera_point_inv_z[1] -= camera_point_inv_z[0];
			camera_point_inv_z[2] -= camera_point_inv_z[0];

			bmin.x = maxf(0.0f, bmin.x);
			bmin.y = maxf(0.0f, bmin.y);

			bmax.x = minf(w1, bmax.x);
			bmax.y = minf(h1, bmax.y);

			float inv_area = 1.0f / area;

			const float p01x = (points[0].x - points[1].x) * inv_area;
			const float p01y = (points[0].y - points[1].y) * inv_area;

			const float p02x = (points[0].x - points[2].x) * inv_area;
			const float p02y = (points[2].y - points[0].y) * inv_area;


			#pragma omp parallel for schedule(guided)
			for (int y = bmin.y; y <= bmax.y; ++y)
			{

				float u = (p02x * (y - points[0].y) + (bmin.x - points[0].x) * p02y),
					v = ((bmin.x - points[0].x) * p01y + p01x * (points[0].y - y));


				bool in = false;

				for (uint32_t x = bmin.x; x <= bmax.x; ++x)
				{
					u += p02y;
					v += p01y;

					if (u < 0 || v < 0 || u + v > 1.0f)
					{
						if (in)
							break;
						continue;
					}

					in = true;
					float z = calculate_depth(camera_point_inv_z, u, v);

					int pixel_ind = y * w_ + x;
					if (z >= z_buffer[pixel_ind])
					{
						continue;
					}

					z_buffer[pixel_ind] = z;		
				}
			}
		}
	}

	float max_elevation_angle(vector<float>& z_buffer, vec2& p, vec2& direction)
	{
		float max_angle = 0.0f;

		float depth = z_buffer[int(p.x) + int(p.y) * w];

		for (float t = 0.0f; t <= 1.0f; t += 1.0f)
		{
			vec2 cur = p + direction * t;

			if (cur.x >= w || cur.y >= h || cur.x < 0 || cur.y < 0)
				continue;

			float distance = (p - cur).length();

			if (distance < 1.0f)
				continue;

			float elevation = z_buffer[int(cur.y) * w + int(cur.x)] - depth;

			max_angle = maxf(max_angle, atanf(elevation / distance));
		}
		return max_angle;
	}

	void __fastcall render_scene_ssao(Camera*& camera, int w_, int h_, image& img, vector<float>& z_buffer, bool render_pass)
	{
		//int w = camera->w;
		//int h = camera->h;

		vector<float> depth_buffer(w * h, 1e6);

		render_scene_depth(camera, w_, h_, depth_buffer, true);

		int w1 = w_ - 1;
		int h1 = h_ - 1;

		vec2i bmin(w1, h1);
		vec2i bmax(0, 0);

		for (int i = 0; i < mesh.num_triangles; ++i)
		{
			vec2* points = new vec2[3];

			//vec4 camera_point_inv_z[3];
			float* camera_point_inv_z = new float[3];

			vec3 normals[3];

			for (int j = 0; j < 3; ++j)
				//for(int j = 3; j--;)
			{
				int vertex_ind = mesh.faces[9 * i + 3 * j];
				int normal_ind = mesh.faces[9 * i + 3 * j + 2];

				vec3 v = mesh.vertices[vertex_ind];

				normals[j] = mesh.normals[normal_ind];


				points[j] = camera->rasterize(v, camera_point_inv_z[j]);

				//camera_point_inv_z[j].z = 1.0f / camera_point_inv_z[j].z;

				bmin.x = minf(bmin.x, points[j].x);
				bmax.x = maxf(bmax.x, points[j].x);

				bmin.y = minf(bmin.y, points[j].y);
				bmax.y = maxf(bmax.y, points[j].y);

			}



			float area = (points[0].x - points[1].x) * (points[0].y - points[2].y) + (points[2].x - points[0].x) * (points[0].y - points[1].y);

			//front face culling
			//use for shadow mapping
			if (!render_pass && area <= 0)
			{
				delete(camera_point_inv_z);
				delete(points);
				continue;
			}
			//back face culling
			//use for forward rendering
			if (render_pass && area >= 0)
			{
				delete(camera_point_inv_z);
				delete(points);
				continue;
			}

			normals[1] -= normals[0];
			normals[2] -= normals[0];

			camera_point_inv_z[1] -= camera_point_inv_z[0];
			camera_point_inv_z[2] -= camera_point_inv_z[0];

			bmin.x = maxf(0.0f, bmin.x);
			bmin.y = maxf(0.0f, bmin.y);

			bmax.x = minf(w1, bmax.x);
			bmax.y = minf(h1, bmax.y);

			float inv_area = 1.0f / area;

			const float p01x = (points[0].x - points[1].x) * inv_area;
			const float p01y = (points[0].y - points[1].y) * inv_area;

			const float p02x = (points[0].x - points[2].x) * inv_area;
			const float p02y = (points[2].y - points[0].y) * inv_area;


			#pragma omp parallel for schedule(guided)
			for (int y = bmin.y; y <= bmax.y; ++y)
			{

				float u = (p02x * (y - points[0].y) + (bmin.x - points[0].x) * p02y),
					v = ((bmin.x - points[0].x) * p01y + p01x * (points[0].y - y));


				bool in = false;

				for (uint32_t x = bmin.x; x <= bmax.x; ++x)
				{
					u += p02y;
					v += p01y;

					if (u < 0 || v < 0 || u + v > 1.0f)
					{
						if (in)
							break;
						continue;
					}

					in = true;
					float z = calculate_depth(camera_point_inv_z, u, v);

					int pixel_ind = y * w_ + x;
					if (z >= z_buffer[pixel_ind])
					{
						continue;
					}

					float total_ssao = 0.0f;
					vec2 p(x, y);
					for (float a = 0.0f; a < 2.0f * pi - 1e-4; a += 0.25 * pi)
					{
						total_ssao += pi * 0.5f - max_elevation_angle(depth_buffer, p, vec2(cosf(a), sinf(a)));
					}
					total_ssao *= (ipi * 0.25f) * 0.1;
					//total_ssao = powf(total_ssao, 1.0f);
					//good
					//img.data[pixel_ind] = vec3(200 * total);

					//vec2 screen_point(x, y); //= points[0] * a + points[1] * b + points[2] * c;

					//vec3 world_point = unrasterize(screen_point, z);

					//float shadow_value = point_in_shadow(world_point);

					z_buffer[pixel_ind] = z;

					if (render_pass)
					{

						vec3 n = normals[0] + u * normals[1] + v * normals[2];
						

						//pcf
						/*float shadow_factor = 0;// in_shadow(orthographic_camera, world_point, n);
						float d = 0;
						float ambient = 0.001f;

						for (int dx = -1; dx <= 1; ++dx)
						{
						for (int dy = -1; dy <= 1; ++dy)
						{
						vec2 screen_point(x + dx, y + dy);

						vec3 world_point = camera->unrasterize(screen_point, z);

						float shadow_value = in_shadow(orthographic_camera, world_point, n);

						shadow_factor += shadow_value;

						d += ambient + maxf(0.0f, dot(n, (close_light_position - world_point).norm()));
						}
						}

						shadow_factor *= 0.1111f;
						d *= 0.1111f;*/

						vec2 screen_point(x, y);
						//float ambient = 0.001f;
						float ambient = total_ssao;
						vec3 world_point = camera->unrasterize(screen_point, z);

						float d = ambient + maxf(0.0f, dot(n, (close_light_position - world_point).norm()));

						float shadow_factor = in_shadow(orthographic_camera, world_point, n);

						//img.data[pixel_ind] = 255.0f * shadow_factor * vec3(d);

						//vec3 d = fragment(n);

						img.data[pixel_ind] = 200.0f * shadow_factor * vec3(d);
					}

					//delete(camera_point_inv_z);
				}
			}
		}
	}


	void render()
	{
		clock_t start = clock();

		//shadow_buffer = new float[sw * sh];
		//memset(shadow_buffer, inf, sizeof(shadow_buffer));

		image depth_img(1, 1);

		image img(w, h);

		//float* z_buffer = new float[w * h];
		//memset(z_buffer, inf, sizeof(z_buffer));

		vector<float> z_buffer(w * h, 1e6);

		float print_time = platform_get_time();
		int num_frame = 0;

		while (!window->is_close)
		{
			float curr_time = platform_get_time();

			handle_event(perspective_camera);

			update_camera(perspective_camera);

			update_light(orthographic_camera);

			++num_frame;

			render_scene(orthographic_camera, sw, sh, depth_img, shadow_buffer, false);

			//for (int k = 0; k < sw * sh; ++k)
			//	if (shadow_buffer[k] != 2 * 1e6)
			//		cout << shadow_buffer[k];


			render_scene(perspective_camera, w, h, img, z_buffer, true);

			//render_scene_fast(perspective_camera, img, z_buffer, true);

			if (curr_time - print_time >= 1)
			{
				int sum_millis = (int)((curr_time - print_time) * 1000);
				int avg_millis = sum_millis / num_frame;
				fprintf(stderr, "fps: %3d, avg: %3d ms\n", num_frame, avg_millis);

				num_frame = 0;
				print_time = curr_time;
			}

			img.flip_vertical();

			window_draw(img);

			msg_dispatch();

			clean_z_buffer(w, h, z_buffer);
			img.clean();

			clean_z_buffer(sw, sh, shadow_buffer);
			depth_img.clean();
		}

		//delete(z_buffer);
		//delete(shadow_buffer);
		z_buffer.swap(vector<float>());
		shadow_buffer.swap(vector<float>());

		//img.clear();
		//depth_img.clear();
		img.data.swap(vector<vec3>());
		depth_img.data.swap(vector<vec3>());
	}

	void render_ssao()
	{
		clock_t start = clock();

		//shadow_buffer = new float[sw * sh];
		//memset(shadow_buffer, inf, sizeof(shadow_buffer));

		image depth_img(1, 1);

		image img(w, h);

		//float* z_buffer = new float[w * h];
		//memset(z_buffer, inf, sizeof(z_buffer));

		vector<float> z_buffer(w * h, 1e6);

		float print_time = platform_get_time();
		int num_frame = 0;

		while (!window->is_close)
		{
			float curr_time = platform_get_time();

			handle_event(perspective_camera);

			update_camera(perspective_camera);

			update_light(orthographic_camera);

			++num_frame;

			render_scene(orthographic_camera, sw, sh, depth_img, shadow_buffer, false);

			//for (int k = 0; k < sw * sh; ++k)
			//	if (shadow_buffer[k] != 2 * 1e6)
			//		cout << shadow_buffer[k];


			//render_scene(perspective_camera, w, h, img, z_buffer, true);

			//render_scene_fast(perspective_camera, img, z_buffer, true);

			render_scene_ssao(perspective_camera, w, h, img, z_buffer, true);

			if (curr_time - print_time >= 1)
			{
				int sum_millis = (int)((curr_time - print_time) * 1000);
				int avg_millis = sum_millis / num_frame;
				fprintf(stderr, "fps: %3d, avg: %3d ms\n", num_frame, avg_millis);

				num_frame = 0;
				print_time = curr_time;
			}

			img.flip_vertical();

			window_draw(img);

			msg_dispatch();

			clean_z_buffer(w, h, z_buffer);
			img.clean();

			clean_z_buffer(sw, sh, shadow_buffer);
			depth_img.clean();
		}

		//delete(z_buffer);
		//delete(shadow_buffer);
		z_buffer.swap(vector<float>());
		shadow_buffer.swap(vector<float>());

		//img.clear();
		//depth_img.clear();
		img.data.swap(vector<vec3>());
		depth_img.data.swap(vector<vec3>());
	}


	void clean_z_buffer(int w_, int h_, vector<float>& z_buffer)
	{
		for (int i = 0; i < w_ * h_; ++i)
			z_buffer[i] = 2 * 1e6;
	}
	void clean_z_buffer(float* z_buffer)
	{
		for (int i = 0; i < w * h; ++i)
			z_buffer[i] = 2 * 1e6;//inf;
	}
	void clean_scene()
	{
		//tex_color.clear();
		//tex_normal.clear();
		//tex_specular.clear();

		mesh.clear();
	}
};

#endif // !_SCENE_H_