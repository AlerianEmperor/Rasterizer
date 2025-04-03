#ifndef _SCENE_2_FINAL_H_
#define _SCENE_2_FINAL_H_

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

	float inv_w;
	float inv_h;

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
	vec3 close_light_poition;
	//vec3 light_direction = vec3(1.0f, 0.4, -1.0f);
	//vec3 light_direction = vec3(1.0f, 0.9, -0.5f);
	vec3 light_direction = vec3(1.0f, 1.0f, -0.6f);

	const float DIRECTIONAL_LIGHT_DISTANCE = 1e6;


	mat4 model = mat4_identity();
	mat4 inv_model;
	//mat4 view;// = lookAt(vec3(0.5, 1, 1), vec3(0, 0.5, 0), vec3(0, 1, 0));
	//mat4 projection;// = simple_perpspective(1.2);
	//mat4 view_port;// = viewport(0, 0, w, h, 256);

	//mat4 inv_projection;
	//mat4 inv_view;

	//mat4 shadow_view;
	//mat4 shadow_projection;
	//mat4 shadow_view_port;

	Camera* perspective_camera = NULL;
	Camera* orthographic_camera = NULL;

	//Texture tex_color;
	//Texture tex_normal;
	//Texture tex_specular;

	//Texture depth_texture;

	mat3 normal_matrix;

	//mat4 mvp = projection * view * model;
	//mat4 shadow_mvp = shadow_projection * shadow_view * model;
	
	//reality morph exe

	//const float light_scale = 0.00002f;
	const float light_scale = 0.02f;

	Scene() {}
	Scene(string file_name, int w_, int h_)
	{
		mesh = load_obj(file_name);

		w = w_;
		h = h_;

		inv_w = 1.0f / (float)w;
		inv_h = 1.0f / (float)h;

		sw = 2.5 * w;
		sh = 2.5 * h;

		w1 = w - 1;
		h1 = h - 1;

		shadow_buffer.resize(sw * sh, 1e6);

		//shadow_buffer = new float[sw * sh];
		//memset(shadow_buffer, 1e6, sizeof(shadow_buffer));

		init_window(w, h, "render");

		light_direction.normalize();

		

		light_position = vec3(light_direction.x * DIRECTIONAL_LIGHT_DISTANCE * cosf(angle), light_direction.y * DIRECTIONAL_LIGHT_DISTANCE, light_direction.z * DIRECTIONAL_LIGHT_DISTANCE * sinf(angle));
		close_light_poition = light_position * light_scale;

		
		model = translate(-0.3f, -0.4f, 0.0f) * scale(0.8, 0.8, 0.8) * translate(0.6, 0.7, 0.8) * rotate(12, 0.2, 0.4, 0.1);
		inv_model = inverse(model);
		//model = mat4_identity();

		//getchar();

		perspective_camera = new Perspective(eye, center, vec3(0, 1, 0), w, h, 0.01f, 1e2, 90.0f);

		//getchar();
		
		orthographic_camera = new Orthographic(close_light_poition, center, vec3(0, 1, 0), sw, sh, 2.0f, 2.0f * 1e6);

		perspective_camera->m_model = model;

		orthographic_camera->m_model = model;

		//light_direction = (light_position - center).norm();

	

		const float DIRECTIONAL_LIGHT_DISTANCE = 1e6;
	}

	void update_camera(Camera*& cam)
	{
		cam->m_lookat = lookAt(cam->eye, cam->center, cam->up);
		cam->m_lookat_inv = inverse(cam->m_lookat);

		cam->m_model_inv = inv_model;
	}

	void update_light(Camera*& shadow_camera)
	{
		angle += 0.01f;
		light_position.x = light_direction.x * DIRECTIONAL_LIGHT_DISTANCE * cosf(angle);
		light_position.z = light_direction.z * DIRECTIONAL_LIGHT_DISTANCE * sinf(angle);

		//light_direction = (light_position - center).norm();
		//light_position = vec3(light_direction.x * DIRECTIONAL_LIGHT_DISTANCE, light_direction.y * DIRECTIONAL_LIGHT_DISTANCE, light_direction.z * DIRECTIONAL_LIGHT_DISTANCE);
		
		close_light_poition = light_position * light_scale;

		shadow_camera->m_lookat = lookAt(light_position, shadow_camera->center, shadow_camera->up);
		
	}


	float calculate_depth(vec4 view_space[3], float& u, float& v)
	{
		//float z = 1.0f / ((1.0f - u - v) / view_space[0].z + u / view_space[1].z + v / view_space[2].z);

		float z = ((1.0f - u - v) * view_space[0].z + u * view_space[1].z + v * view_space[2].z);

		return 1.0f / z;
	}

	float calculate_depth(float*& camera_inv_z, float& u, float& v)
	{
		//float z = 1.0f / ((1.0f - u - v) / view_space[0].z + u / view_space[1].z + v / view_space[2].z);

		//float z = ((1.0f - u - v) * camera_inv_z[0] + u * camera_inv_z[1] + v * camera_inv_z[2]);

		//return 1.0f / z;

		//return 1.0f / ((1.0f - u - v) * camera_inv_z[0] + u * camera_inv_z[1] + v * camera_inv_z[2]);

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

		//cout << index << "\n";

		float shadow = 0.0f;

		float closest_depth = shadow_buffer[index];
		
		//if (closest_depth > 0)
		//	cout << closest_depth << "\n";

		float current_depth = camera_point.z;

		//vec3 lp = light_position - (world_point + normal);

		//float current_depth = sqrtf(lp.x * lp.x + lp.y * lp.y + lp.z * lp.z);

		float bias = maxf(0.06f * (1.0f - abs(dot(normal, light_direction))), 0.006f);

		//cout << closest_depth << " " << current_depth << "\n";

		float s = current_depth >= closest_depth + bias ? 0.4f : 1.0f;

		//if (s <= 0.4)
		//	cout << "shadow\n";
		//else
		//	cout << "not shadow\n";
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

		return vec3(d);
	}

	//void __fastcall render_scene(Camera*& camera, image& img, float*& z_buffer, bool render_pass)
	void __fastcall render_scene(Camera*& camera, image& img, vector<float>& z_buffer, bool render_pass)
	{
		int w = camera->w;
		int h = camera->h;

		int w1 = w - 1;
		int h1 = h - 1;

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

				//points[j] = rasterize(camera, v, camera_point_inv_z[j]);

				//points[j] = rasterize_2(camera, v, camera_point_inv_z[j]);

				//vec4 model_point = model * vec4(v, 1.0f);

				points[j] = camera->rasterize(v, camera_point_inv_z[j]);

				//camera_point_inv_z[j].z = 1.0f / camera_point_inv_z[j].z;

				bmin.x = minf(bmin.x, points[j].x);
				bmax.x = maxf(bmax.x, points[j].x);

				bmin.y = minf(bmin.y, points[j].y);
				bmax.y = maxf(bmax.y, points[j].y);

				/*minf2(bmin.x, points[j].x);
				minf2(bmin.y, points[j].y);

				maxf2(bmax.x, points[j].x);
				maxf2(bmax.y, points[j].y);*/
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

			//points[0].x -= bmin.x;
			//points[0].x = -points[0].x;

			//#pragma omp parallel for schedule(guided)
			for (int y = bmin.y; y <= bmax.y; ++y)	
			{		
			
				float u = (p02x * (y - points[0].y) + (bmin.x - points[0].x) * p02y),
					  v = ((bmin.x - points[0].x) * p01y + p01x * (points[0].y - y));
				
				//float u = (p02x * (y - points[0].y) + p02y * points[0].x);
				//float v = (p01y * points[0].x + p01x * (points[0].y - y));

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

					int pixel_ind = y * w + x;
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
						
						vec2 screen_point(x, y);

						vec3 world_point = camera->unrasterize(screen_point, z);
						
						//float shadow_value = in_shadow(orthographic_camera, world_point, n);
						
						//if (z > 0)
						//	cout << z << "\n";
						//img.data[pixel_ind] = z * 25500;
						//img.data[pixel_ind] = 255 * fragment(n);

						float shadow_factor = in_shadow(orthographic_camera, world_point, n);


						float d = dot(n, (close_light_poition - world_point).norm());

						img.data[pixel_ind] = 255.0f * shadow_factor * vec3(d);
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

			render_scene(orthographic_camera, depth_img, shadow_buffer, false);

			//for (int k = 0; k < sw * sh; ++k)
			//	if (shadow_buffer[k] != 2 * 1e6)
			//		cout << shadow_buffer[k];


			render_scene(perspective_camera, img, z_buffer, true);
		
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