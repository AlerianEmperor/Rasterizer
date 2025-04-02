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

	float* shadow_buffer;

	int w;
	int h;

	float inv_w;
	float inv_h;

	int sw;
	int sh;

	float w1;
	float h1;

	vec3 eye = vec3(10.0, 10.0, 10.0);
	vec3 center = vec3(0.0, 0.0, 0.0);
	vec3 up = vec3(0.0, 1.0, 0.0);

	float angle = 0;

	float light_coeff = 20;

	vec3 light_position = vec3(light_coeff * sinf(angle), light_coeff, light_coeff * cosf(angle));
	vec3 light_dir;// (0, 0, -1);



	mat4 model = mat4_identity();
	//mat4 view;// = lookAt(vec3(0.5, 1, 1), vec3(0, 0.5, 0), vec3(0, 1, 0));
	//mat4 projection;// = simple_perpspective(1.2);
	//mat4 view_port;// = viewport(0, 0, w, h, 256);

	//mat4 inv_projection;
	//mat4 inv_view;

	//mat4 shadow_view;
	//mat4 shadow_projection;
	//mat4 shadow_view_port;

	Camera* perspective_camera;

	//Texture tex_color;
	//Texture tex_normal;
	//Texture tex_specular;

	//Texture depth_texture;

	mat3 normal_matrix;

	//mat4 mvp = projection * view * model;
	//mat4 shadow_mvp = shadow_projection * shadow_view * model;

	Scene() {}
	Scene(string file_name, int w_, int h_)
	{
		mesh = load_obj(file_name);

		w = w_;
		h = h_;

		inv_w = 1.0f / (float)w;
		inv_h = 1.0f / (float)h;

		sw = w;
		sh = h;

		w1 = w - 1;
		h1 = h - 1;

		shadow_buffer = new float[w * h];
		memset(shadow_buffer, inf, sizeof(shadow_buffer));

		//shadow_buffer.resize(w * h);

		init_window(w, h, "render");

		model = translate(-0.3f, -0.4f, 0.0f) * scale(0.8, 0.8, 0.8) * translate(0.6, 0.7, 0.8) * rotate(12, 0.2, 0.4, 0.1);

		//view = lookAt(eye, center, up);

		//projection = persepective_camera(eye, center, w, h, 2.0f, 5000.0f, 40.0f, inv_projection);

		//projection = perspective_horizontal_fov(40, (float)w / (float)h, 0.1, 100);

		//projection = PerspectiveCamera(eye, center, w, h, 1.0f, 100.0f, 40.0f);

		//view_port = viewport(0, 0, w, h, 0);

		//cam = Camera(eye, center, up);

		perspective_camera = new Perspective(eye, center, vec3(0, 1, 0), w, h, 2.0f, 5000.0f, 50.0f);

		light_dir = (light_position - center).norm();

		//tex_color = Texture("D:\\a_c++Rasterizer\\Models\\african_head_diffuse.tga", true, true, false);
		//tex_normal = Texture("D:\\a_c++Rasterizer\\Models\\tinyrenderer_master\\obj\\african_head\\african_head_nm.tga", false, false, true);
		//tex_specular = Texture("D:\\a_c++Rasterizer\\Models\\tinyrenderer_master\\obj\\african_head\\african_head_spec.tga", false, false, false);

		//init gourad


		float a = 40;

		//mat4 shadow_model = model;
		//shadow_view = lookAt(light_position, center, up);
		//shadow_projection = simple_perpspective(0);
		//shadow_projection = perspective_vertical_fov(40, 1, 0.5, 10.0);//0.1 10

		//shadow_projection = ortho(-a, a, a, -a, 0, 40);

		//shadow_view_port = viewport(0, 0, sw, sh, 0);

		//update_matrix(model, projection, cam);
	}

	//void update_matrix(mat4& model_, mat4& projection_, Camera& cam)
	void update_camera(Camera*& cam)
	{
		//model = model_;
		//view = lookAt(cam.eye, cam.center, cam.up);
		//projection = projection_;

		//mat3 n_matrix = mat4_to_mat3(model);

		//normal_matrix = mat4_to_mat3(inverse_transpose(model));

		cam->m_lookat = lookAt(cam->eye, cam->center, cam->up);

		angle += 0.01f;
		light_position.x = light_coeff * sinf(angle);
		light_position.z = light_coeff * cosf(angle);

		light_dir = (light_position - center).norm();

		//shadow_view = lookAt(light_position, center, up);

		//mat4 shadow_mvp = shadow_projection * shadow_view * model;

		//mat4 shadow_mvpv = shadow_view_port * shadow_projection * shadow_view * model;


		//mat4 shadow_mvp = shadow_projection * shadow_view * model;
	}

	/*float calculate_depth(vec4 view_space[3], float& u, float& v, float& w)
	{
	float z = 1.0f/(u / view_space[0].z + v / view_space[1].z + w / view_space[2].z);

	return z;
	}*/

	float calculate_depth(vec4 view_space[3], float& u, float& v)
	{
		//float z = 1.0f / ((1.0f - u - v) / view_space[0].z + u / view_space[1].z + v / view_space[2].z);

		float z = 1.0f / ((1.0f - u - v) * view_space[0].z + u * view_space[1].z + v * view_space[2].z);

		return z;
	}

	float calculate_depth_2(float camera_inv_z[3], float& u, float& v)
	{
		//float z = 1.0f / ((1.0f - u - v) / view_space[0].z + u / view_space[1].z + v / view_space[2].z);

		float z = 1.0f / ((1.0f - u - v) * camera_inv_z[0] + u * camera_inv_z[1] + v * camera_inv_z[2]);


		return z;
	}


	vec4 view_transform(mat4& view_, vec4& world_point)
	{
		//vec4 p(world_point, 1.0f);

		vec4 r = view_ * world_point;//p;

		return r;//vec3(r.x, r.y, r.z);
	}

	vec3 project_transform(mat4& projection_, vec4& camera_point)
	{
		//vec4 p(camera_point.x, camera_point.y, camera_point.z, 1.0f);

		vec4 r = projection_ * camera_point;//p;

		float inv_w = 1.0f / r.w;

		return vec3(r.x * inv_w, r.y * inv_w, r.z * inv_w);
	}

	vec2 viewport_transform(vec3& ndc)
	{
		return vec2((ndc.x + 1.0f) * w * 0.5f, (ndc.y + 1.0f) * h * 0.5f);
	}

	//vec2 rasterize(mat4& view_, mat4& projection_, vec4& world_point, vec4& camera_point)
	vec2 rasterize(Camera* camera, vec3& world_point, vec4& camera_point)
	{
		//vec4 model_point = model * vec4(world_point, 1.0f);
		//camera_point = view_transform(view_, model_point);
		//vec3 ndc_point = project_transform(projection_, camera_point);
		//vec2 screen_point = viewport_transform(ndc_point);

		vec4 model_point = model * vec4(world_point, 1.0f);
		camera_point = camera->view_transform(model_point);
		vec2 ndc_point = camera->project_transform(camera_point);
		vec2 screen_point = camera->viewport_transform(ndc_point);

		camera_point.z = 1.0f / camera_point.z;

		return screen_point;
	}

	vec2 rasterize_2(Camera* camera, vec3& world_point, float& camera_inv_z)
	{
		//vec4 model_point = model * vec4(world_point, 1.0f);
		//camera_point = view_transform(view_, model_point);
		//vec3 ndc_point = project_transform(projection_, camera_point);
		//vec2 screen_point = viewport_transform(ndc_point);

		vec4 model_point = model * vec4(world_point, 1.0f);
		vec4 camera_point = camera->view_transform(model_point);
		vec2 ndc_point = camera->project_transform(camera_point);
		vec2 screen_point = camera->viewport_transform(ndc_point);

		camera_inv_z = 1.0f / camera_point.z;

		return screen_point;
	}

	//unrastersize
	//viewport to ndc
	/*vec2 viewport_transform_inv(vec2& screen_point)
	{
	return vec2(2.0f * screen_point.x * inv_w - 1.0f, 2.0f * screen_point.y * inv_h - 1.0f);
	}

	//ndc to camera
	vec3 project_transform_inv(vec2& ndc_point, float& depth)
	{
	vec4 p(ndc_point.x * depth, ndc_point.y * depth, depth, 1.0f);
	vec4 r = inv_projection * p;
	return vec3(r.x, r.y, r.z * depth);
	}

	//camera to world
	vec3 view_transform_inv(vec3& camera_point)
	{
	vec4 p(camera_point.x, camera_point.y, camera_point.z, 1.0f);

	vec4 r = inv_view * p;

	return vec3(r.x, r.y, r.z);
	}

	vec3 unrasterize(vec2& screen_point, float& camera_depth)
	{
	vec2 ndc_point = viewport_transform_inv(screen_point);
	vec3 camera_point = project_transform_inv(ndc_point, camera_depth);
	vec3 world_point = view_transform_inv(camera_point);

	return world_point;
	}
	*/

	vec3 get_light_position()
	{
		vec3 light_pos(light_dir.x * DIRECTIONAL_LIGHT_DISTANCE, light_dir.y * DIRECTIONAL_LIGHT_DISTANCE, light_dir.z * DIRECTIONAL_LIGHT_DISTANCE);

		return light_pos;
	}

	vec3 fragment(vec3& n)
	{
		float d = dot(n, light_dir);

		return 200 * d;
	}

	void render_scene(Camera*& camera, image& img, vector<float>& z_buffer, bool render_pass)
	{
		vec2i bmin(w1, h1);
		vec2i bmax(0, 0);


		for (int i = 0; i < mesh.num_triangles; ++i)
		{
			vec2 points[3];
			vec4 camera_point[3];
			vec3 normals[3];

			for (int j = 0; j < 3; ++j)
			{
				int vertex_ind = mesh.faces[9 * i + 3 * j];
				int normal_ind = mesh.faces[9 * i + 3 * j + 2];

				vec3 v = mesh.vertices[vertex_ind];

				normals[j] = mesh.normals[normal_ind];

				//vec4 world_point = model * vec4(v, 1.0f);

				//camera_coord[j] = view_transform(view, v);//world_point);

				//vec4 world_point(v, 1.0f);

				//points[j] = rasterize(view, projection, world_point, camera_coord[j]);

				points[j] = rasterize(camera, v, camera_point[j]);

				bmin.x = min(bmin.x, points[j].x);
				bmin.y = min(bmin.y, points[j].y);

				bmax.x = max(bmax.x, points[j].x);
				bmax.y = max(bmax.y, points[j].y);
			}

			bmin.x = max(0.0f, bmin.x);
			bmin.y = max(0.0f, bmin.y);

			bmax.x = min(w1, bmax.x);
			bmax.y = min(h1, bmax.y);

			float area = (points[0].x - points[1].x) * (points[0].y - points[2].y) + (points[2].x - points[0].x) * (points[0].y - points[1].y);

			if (area < 1e-4 && area > -1e-4)
				continue;

			float inv_area = 1.0f / area;

			//#pragma omp parallel for schedule(guided)
			for (int x = bmin.x; x <= bmax.x; ++x)
			{
				//float u = (points[0].x - x) * (points[0].y - points[2].y) * inv_area
				//float u = ((points[0].x - points[2].x) * (bmin.y - points[0].y) + (points[0].x - x) * (points[0].y - points[2].y)) * inv_area;
				//float v = ((x - points[0].x) * (points[0].y - points[1].y) + (points[0].x - points[1].x) * (points[0].y - bmin.y)) * inv_area;

				//float diff_x = (points[0].x - points[2].x) * inv_area;
				//float diff_y = (points[0].x - points[1].x) * inv_area;
#pragma omp parallel for schedule(guided)
				for (int y = bmin.y; y <= bmax.y; ++y)
				{
					float u = ((points[0].x - points[2].x) * (y - points[0].y) + (points[0].x - x) * (points[0].y - points[2].y)) * inv_area;
					float v = ((x - points[0].x) * (points[0].y - points[1].y) + (points[0].x - points[1].x) * (points[0].y - y)) * inv_area;

					//u += diff_x;
					//v -= diff_y;

					//if (ux < 0.0f || uy < 0.0f || ux + uy > 1.0f)
					if (u < 0 || v < 0 || u + v > 1.0f)
						continue;

					//float a = 1.0f - ux - uy;
					//float b = ux;
					//float c = uy;

					//float z = calculate_depth(camera_coord, a, b, c);

					float z = calculate_depth(camera_point, u, v);

					int pixel_ind = y * w + x;
					if (z >= z_buffer[pixel_ind])
						continue;

					vec3 n = normals[0] * (1.0f - u - v) + normals[1] * u + normals[2] * v;

					vec2 screen_point(x, y); //= points[0] * a + points[1] * b + points[2] * c;

											 //vec3 world_point = unrasterize(screen_point, z);

											 //float shadow_value = point_in_shadow(world_point);

					z_buffer[pixel_ind] = z;

					if (render_pass)
						img.data[pixel_ind] = fragment(n);// -1000 * vec3(shadow_value);
				}
			}
		}
	}

	void render_scene_fast(Camera*& camera, image& img, vector<float>& z_buffer, bool render_pass)
	{
		vec2i bmin(w1, h1);
		vec2i bmax(0, 0);


		for (int i = 0; i < mesh.num_triangles; ++i)
		{
			vec2 points[3];
			vec4 camera_point[3];
			vec3 normals[3];

			for (int j = 0; j < 3; ++j)
			{
				int vertex_ind = mesh.faces[9 * i + 3 * j];
				int normal_ind = mesh.faces[9 * i + 3 * j + 2];

				vec3 v = mesh.vertices[vertex_ind];

				normals[j] = mesh.normals[normal_ind];

				//vec4 world_point = model * vec4(v, 1.0f);

				//camera_coord[j] = view_transform(view, v);//world_point);

				//vec4 world_point(v, 1.0f);

				//points[j] = rasterize(view, projection, world_point, camera_coord[j]);

				points[j] = rasterize(camera, v, camera_point[j]);

				bmin.x = min(bmin.x, points[j].x);
				bmin.y = min(bmin.y, points[j].y);

				bmax.x = max(bmax.x, points[j].x);
				bmax.y = max(bmax.y, points[j].y);
			}

			bmin.x = max(0.0f, bmin.x);
			bmin.y = max(0.0f, bmin.y);

			bmax.x = min(w1, bmax.x);
			bmax.y = min(h1, bmax.y);

			float area = (points[0].x - points[1].x) * (points[0].y - points[2].y) + (points[2].x - points[0].x) * (points[0].y - points[1].y);

			if (area < 1e-4 && area > -1e-4)
				continue;

			float inv_area = 1.0f / area;

#pragma omp parallel for schedule(guided)
			for (int x = bmin.x; x <= bmax.x; ++x)
			{
				//float u = (points[0].x - x) * (points[0].y - points[2].y) * inv_area
				float u = ((points[0].x - points[2].x) * (bmin.y - points[0].y) + (points[0].x - x) * (points[0].y - points[2].y)) * inv_area;
				float v = ((x - points[0].x) * (points[0].y - points[1].y) + (points[0].x - points[1].x) * (points[0].y - bmin.y)) * inv_area;

				float diff_x = (points[0].x - points[2].x) * inv_area;
				float diff_y = (points[0].x - points[1].x) * inv_area;

				for (int y = bmin.y; y <= bmax.y; ++y)
				{
					//float ux = ((points[0].x - points[2].x) * (y - points[0].y) + (points[0].x - x) * (points[0].y - points[2].y)) * inv_area;
					//float uy = ((x - points[0].x) * (points[0].y - points[1].y) + (points[0].x - points[1].x) * (points[0].y - y)) * inv_area;

					u += diff_x;
					v -= diff_y;

					//if (ux < 0.0f || uy < 0.0f || ux + uy > 1.0f)
					if (u < 0 || v < 0 || u + v > 1.0f)
						continue;

					//float a = 1.0f - ux - uy;
					//float b = ux;
					//float c = uy;

					//float z = calculate_depth(camera_coord, a, b, c);

					float z = calculate_depth(camera_point, u, v);

					int pixel_ind = y * w + x;
					if (z >= z_buffer[pixel_ind])
						continue;

					vec3 n = normals[0] * (1.0f - u - v) + normals[1] * u + normals[2] * v;

					vec2 screen_point(x, y); //= points[0] * a + points[1] * b + points[2] * c;

											 //vec3 world_point = unrasterize(screen_point, z);

											 //float shadow_value = point_in_shadow(world_point);

					z_buffer[pixel_ind] = z;

					if (render_pass)
						img.data[pixel_ind] = fragment(n);// -1000 * vec3(shadow_value);
				}
			}
		}
	}

	void render_scene_fast_2(Camera*& camera, image& img, vector<float>& z_buffer, bool render_pass)
	{
		vec2i bmin(w1, h1);
		vec2i bmax(0, 0);


		for (int i = 0; i < mesh.num_triangles; ++i)
		{
			vec2 points[3];
			vec4 camera_point[3];
			vec3 normals[3];

			for (int j = 0; j < 3; ++j)
			{
				int vertex_ind = mesh.faces[9 * i + 3 * j];
				int normal_ind = mesh.faces[9 * i + 3 * j + 2];

				vec3 v = mesh.vertices[vertex_ind];

				normals[j] = mesh.normals[normal_ind];

				//vec4 world_point = model * vec4(v, 1.0f);

				//camera_coord[j] = view_transform(view, v);//world_point);

				//vec4 world_point(v, 1.0f);

				//points[j] = rasterize(view, projection, world_point, camera_coord[j]);

				points[j] = rasterize(camera, v, camera_point[j]);

				bmin.x = min(bmin.x, points[j].x);
				bmin.y = min(bmin.y, points[j].y);

				bmax.x = max(bmax.x, points[j].x);
				bmax.y = max(bmax.y, points[j].y);
			}

			float area = (points[0].x - points[1].x) * (points[0].y - points[2].y) + (points[2].x - points[0].x) * (points[0].y - points[1].y);

			//front face culling
			//if(area < 0)
			//	continue

			//back face culling
			if (area >= 0)
				continue;

			bmin.x = max(0.0f, bmin.x);
			bmin.y = max(0.0f, bmin.y);

			bmax.x = min(w1, bmax.x);
			bmax.y = min(h1, bmax.y);


			//if (area < 1e-4 && area > -1e-4)



			//bmin.x = (int)bmin.x;
			//bmax.x = (int)bmax.x;

			//bmin.y = (int)bmin.y;
			//bmax.y = (int)bmax.y;

			float inv_area = 1.0f / area;

			//float u = ((points[0].x - points[2].x) * (bmin.y - points[0].y) + (points[0].x - bmin.x) * (points[0].y - points[2].y)) * inv_area;
			//float v = ((bmin.x - points[0].x) * (points[0].y - points[1].y) + (points[0].x - points[1].x) * (points[0].y - bmin.y)) * inv_area;

			//float dx = (points[0].y - points[2].y) * inv_area;
			//float dy = (points[0].y - points[1].y) * inv_area;


			//points[1] -= points[0];
			//points[2] -= points[0];

			//points[1] = -points[1];
			//points[2] = -points[2];

#pragma omp parallel for schedule(guided)
			for (int x = bmin.x; x <= bmax.x; ++x)
			{
				//float u = (points[0].x - x) * (points[0].y - points[2].y) * inv_area

				float u = ((points[0].x - points[2].x) * (bmin.y - points[0].y) + (points[0].x - x) * (points[0].y - points[2].y)) * inv_area;
				float v = ((x - points[0].x) * (points[0].y - points[1].y) + (points[0].x - points[1].x) * (points[0].y - bmin.y)) * inv_area;

				//float u = (-points[2].x * (bmin.y - points[0].y) + (points[0].x - x) * -points[2].y) * inv_area;
				//float v = ((x - points[0].x) * -points[1].y + -points[1].x * (points[0].y - bmin.y)) * inv_area;



				//u -= (points[0].y - points[2].y) * inv_area;
				//v += (points[0].y - points[1].y) * inv_area;


				//u -= dx;
				//v += dy;

				const float diff_x = (points[0].x - points[2].x) * inv_area;
				const float diff_y = (points[0].x - points[1].x) * inv_area;

				//float diff_x = -points[2].x * inv_area;
				//float diff_y = -points[1].x * inv_area;

				bool first_cross = false;

				for (int y = bmin.y; y <= bmax.y; ++y)
				{
					//float ux = ((points[0].x - points[2].x) * (y - points[0].y) + (points[0].x - x) * (points[0].y - points[2].y)) * inv_area;
					//float uy = ((x - points[0].x) * (points[0].y - points[1].y) + (points[0].x - points[1].x) * (points[0].y - y)) * inv_area;

					u += diff_x;
					v -= diff_y;

					//if (ux < 0.0f || uy < 0.0f || ux + uy > 1.0f)
					//if (u < 0 || v < 0 || u + v > 1.0f || first_cross)

					if (u < 0 || v < 0 || u + v > 1.0f)
						continue;

					first_cross = true;

					//float a = 1.0f - ux - uy;
					//float b = ux;
					//float c = uy;

					//float z = calculate_depth(camera_coord, a, b, c);

					float z = calculate_depth(camera_point, u, v);

					int pixel_ind = y * w + x;
					if (z >= z_buffer[pixel_ind])
						continue;

					vec3 n = normals[0] * (1.0f - u - v) + normals[1] * u + normals[2] * v;

					vec2 screen_point(x, y); //= points[0] * a + points[1] * b + points[2] * c;

											 //vec3 world_point = unrasterize(screen_point, z);

											 //float shadow_value = point_in_shadow(world_point);

					z_buffer[pixel_ind] = z;

					if (render_pass)
						img.data[pixel_ind] = fragment(n);// -1000 * vec3(shadow_value);
				}
			}
		}
	}

	//failed!!
	void __fastcall render_scene_fast(Camera*& camera, image& img, float* z_buffer, bool render_pass)
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
			//if (!render_pass && area <= 0)
			//	continue;

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

			bmin.x = maxf(0, bmin.x);
			bmin.y = maxf(0, bmin.y);

			bmax.x = minf(w1, bmax.x);
			bmax.y = minf(h1, bmax.y);

			float inv_area = 1.0f / area;

			//const float p01x = (points[0].x - points[1].x) * inv_area;
			//const float p01y = (points[0].y - points[1].y) * inv_area;

			//const float p02x = (points[0].x - points[2].x) * inv_area;
			//const float p02y = (points[2].y - points[0].y) * inv_area;

			//points[0].x -= bmin.x;
			//points[0].x = -points[0].x;

			/*const int p02x = points[0].x - points[2].x;
			const int p10x = points[1].x - points[0].x;
			const int p21x = points[2].x - points[1].x;

			const int p02y = points[0].y - points[2].y;
			const int p10y = points[1].y - points[0].y;
			const int p21y = points[2].y - points[1].y;
			*/

			const int E0_const = (bmin.x - points[2].x) * (points[0].y - points[2].y) + points[2].y * (points[0].x - points[2].x);
			const int E1_const = (bmin.x - points[0].x) * (points[1].y - points[0].y) + points[0].y * (points[1].x - points[0].x);
			const int E2_const = (bmin.x - points[1].x) * (points[2].y - points[1].y) + points[1].y * (points[2].x - points[1].x);

#pragma omp parallel for schedule(guided)
			for (int y = bmin.y; y <= bmax.y; ++y)
			{

				//float u = (p02x * (y - points[0].y) + (bmin.x - points[0].x) * p02y),
				//	v = ((bmin.x - points[0].x) * p01y + p01x * (points[0].y - y));

				//float u = (p02x * (y - points[0].y) + p02y * points[0].x);
				//float v = (p01y * points[0].x + p01x * (points[0].y - y));

				int E0 = E0_const - y * (points[0].x - points[2].x);
				int E1 = E1_const - y * (points[1].x - points[0].x);
				int E2 = E2_const - y * (points[2].x - points[1].x);


				bool in = false;

				for (uint32_t x = bmin.x; x <= bmax.x; ++x)
				{
					//u += p02y;
					//v += p01y;

					//int E0 = (x - points[2].x) * (points[0].y - points[2].y) - (y - points[2].y) * (points[0].x - points[2].x);
					//int E1 = (x - points[0].x) * (points[1].y - points[0].y) - (y - points[0].y) * (points[1].x - points[0].x);
					//int E2 = (x - points[1].x) * (points[2].y - points[1].y) - (y - points[1].y) * (points[2].x - points[1].x);

					E0 += (points[0].y - points[2].y);
					E1 += (points[1].y - points[0].y);
					E2 += (points[2].y - points[1].y);

					bool b = ((E0 >= 0) && (E1 >= 0) && (E2 >= 0)) || ((E0 <= 0) && (E1 <= 0) && (E2 <= 0));

					if (!b)
					{
						if (in)
							break;
						continue;
					}

					in = true;

					float u = E0 * inv_area;
					float v = E1 * inv_area;

					/*if (u < 0 || v < 0 || u + v > 1.0f)
					{
					if (in)
					break;
					continue;
					}*/

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
						img.data[pixel_ind] = fragment(n);
					}

					//delete(camera_point_inv_z);
				}
			}
		}
	}


	int oriented_2d(vec2i& v0, vec2i& v1, vec2i& p)
	{
		return (v0.y - v1.y) * p.x + (v1.x - v0.x) * p.y + (v0.x * v1.y - v0.y * v1.x);
	}

	float oriented_2d(vec2& v0, vec2& v1, vec2& p)
	{
		return (v0.y - v1.y) * p.x + (v1.x - v0.x) * p.y + (v0.x * v1.y - v0.y * v1.x);
	}


	//https://zielon.github.io/rasterizer/
	//https://fgiesen.wordpress.com/2013/02/10/optimizing-the-basic-rasterizer/
	void render_scene_fast_3(Camera*& camera, image& img, vector<float>& z_buffer, bool render_pass)
	{
		vec2i bmin(w1, h1);
		vec2i bmax(0, 0);


		for (int i = 0; i < mesh.num_triangles; ++i)
		{
			vec2 points[3];
			vec4 camera_point[3];
			vec3 normals[3];

			for (int j = 0; j < 3; ++j)
			{
				int vertex_ind = mesh.faces[9 * i + 3 * j];
				int normal_ind = mesh.faces[9 * i + 3 * j + 2];

				vec3 v = mesh.vertices[vertex_ind];

				normals[j] = mesh.normals[normal_ind];

				points[j] = rasterize(camera, v, camera_point[j]);

				bmin.x = min(bmin.x, points[j].x);
				bmin.y = min(bmin.y, points[j].y);

				bmax.x = max(bmax.x, points[j].x);
				bmax.y = max(bmax.y, points[j].y);
			}

			bmin.x = max(0.0f, bmin.x);
			bmin.y = max(0.0f, bmin.y);

			bmax.x = min(w1, bmax.x);
			bmax.y = min(h1, bmax.y);

			// Triangle setup
			//int A01 = points[0].y - points[1].y, B01 = points[1].x - points[0].x;
			//int A12 = points[1].y - points[2].y, B12 = points[2].x - points[1].x;
			//int A20 = points[2].y - points[0].y, B20 = points[0].x - points[2].x;

			/*
			vec2i points_i[3];

			points_i[0] = vec2i(points[0].x, points[0].y);
			points_i[1] = vec2i(points[1].x, points[1].y);
			points_i[2] = vec2i(points[2].x, points[2].y);

			vec2i point_i(bmin.x, bmin.y);

			int w0_row = oriented_2d(points_i[1], points_i[2], point_i);
			int w1_row = oriented_2d(points_i[2], points_i[0], point_i);
			int w2_row = oriented_2d(points_i[0], points_i[1], point_i);

			int w0_row = oriented_2d(points_i[1], points_i[2], point_i);
			int w1_row = oriented_2d(points_i[2], points_i[0], point_i);
			int w2_row = oriented_2d(points_i[0], points_i[1], point_i);
			*/

			float A01 = points[0].y - points[1].y, B01 = points[1].x - points[0].x;
			float A12 = points[1].y - points[2].y, B12 = points[2].x - points[1].x;
			float A20 = points[2].y - points[0].y, B20 = points[0].x - points[2].x;

			vec2 p(bmin.x, bmin.y);

			float w0_row = oriented_2d(points[1], points[2], p);
			float w1_row = oriented_2d(points[2], points[0], p);
			float w2_row = oriented_2d(points[0], points[1], p);

			//#pragma omp parallel for schedule(guided)
			for (int y = bmin.y; y <= bmax.y; ++y)
			{
				float w0 = w0_row;
				float w1 = w1_row;
				float w2 = w2_row;

				for (int x = bmin.x; x <= bmax.x; ++x)
				{

					if (w0 < 0 || w1 < 0 || w2 < 0)
						continue;


					float z = calculate_depth(camera_point, w0, w1);



					int pixel_ind = y * w + x;
					if (z >= z_buffer[pixel_ind])
						continue;

					vec3 n = normals[0] * w0 + normals[1] * w1 + normals[2] * w2;

					w0 += A12;
					w1 += A20;
					w2 += A01;

					z_buffer[pixel_ind] = z;

					if (render_pass)
						img.data[pixel_ind] = fragment(n);// -1000 * vec3(shadow_value);



				}
				w0_row += B12;
				w1_row += B20;
				w2_row += B01;
			}
		}
	}

	void render_scene_fast_4(Camera*& camera, image& img, vector<float>& z_buffer, bool render_pass)
	{
		vec2i bmin(w1, h1);
		vec2i bmax(0, 0);


		for (int i = 0; i < mesh.num_triangles; ++i)
		{
			vec2 points[3];
			//vec4 camera_point[3];

			vec4 camera_point_inv_z[3];
			vec3 normals[3];

			for (int j = 0; j < 3; ++j)
			{
				int vertex_ind = mesh.faces[9 * i + 3 * j];
				int normal_ind = mesh.faces[9 * i + 3 * j + 2];

				vec3 v = mesh.vertices[vertex_ind];

				normals[j] = mesh.normals[normal_ind];

				//vec4 world_point = model * vec4(v, 1.0f);

				//camera_coord[j] = view_transform(view, v);//world_point);

				//vec4 world_point(v, 1.0f);

				//points[j] = rasterize(view, projection, world_point, camera_coord[j]);

				//points[j] = rasterize(camera, v, camera_point[j]);

				points[j] = rasterize(camera, v, camera_point_inv_z[j]);

				bmin.x = min(bmin.x, points[j].x);
				bmin.y = min(bmin.y, points[j].y);

				bmax.x = max(bmax.x, points[j].x);
				bmax.y = max(bmax.y, points[j].y);
			}

			float area = (points[0].x - points[1].x) * (points[0].y - points[2].y) + (points[2].x - points[0].x) * (points[0].y - points[1].y);

			//front face culling
			//if(area < 0)
			//	continue

			//back face culling
			if (area >= 0)
				continue;

			bmin.x = max(0, bmin.x);
			bmin.y = max(0, bmin.y);

			bmax.x = min(w1, bmax.x);
			bmax.y = min(h1, bmax.y);


			//if (area < 1e-4 && area > -1e-4)



			//bmin.x = (int)bmin.x;
			//bmax.x = (int)bmax.x;

			//bmin.y = (int)bmin.y;
			//bmax.y = (int)bmax.y;

			float inv_area = 1.0f / area;

			//float u = ((points[0].x - points[2].x) * (bmin.y - points[0].y) + (points[0].x - bmin.x) * (points[0].y - points[2].y)) * inv_area;
			//float v = ((bmin.x - points[0].x) * (points[0].y - points[1].y) + (points[0].x - points[1].x) * (points[0].y - bmin.y)) * inv_area;

			//float dx = (points[0].y - points[2].y) * inv_area;
			//float dy = (points[0].y - points[1].y) * inv_area;


			//points[1] -= points[0];
			//points[2] -= points[0];

			//points[1] = -points[1];
			//points[2] = -points[2];

			const float p02x = (points[0].x - points[2].x);
			const float p02y = (points[0].y - points[2].y);

			const float p01x = (points[0].x - points[1].x);
			const float p01y = (points[0].y - points[1].y);


#pragma omp parallel for schedule(guided)
			for (int y = bmin.y; y <= bmax.y; ++y)
			{
				//float u = (points[0].x - x) * (points[0].y - points[2].y) * inv_area

				//float u = ((points[0].x - points[2].x) * (y - points[0].y) + (points[0].x - bmin.x) * (points[0].y - points[2].y)) * inv_area;
				//float v = ((bmin.x - points[0].x) * (points[0].y - points[1].y) + (points[0].x - points[1].x) * (points[0].y - y)) * inv_area;

				float u = (p02x * (y - points[0].y) + (points[0].x - bmin.x) * p02y) * inv_area;
				float v = ((bmin.x - points[0].x) * p01y + p01x * (points[0].y - y)) * inv_area;


				//float u = (-points[2].x * (bmin.y - points[0].y) + (points[0].x - x) * -points[2].y) * inv_area;
				//float v = ((x - points[0].x) * -points[1].y + -points[1].x * (points[0].y - bmin.y)) * inv_area;



				//u -= (points[0].y - points[2].y) * inv_area;
				//v += (points[0].y - points[1].y) * inv_area;


				//u -= dx;
				//v += dy;

				const float diff_x = p02y * inv_area;
				const float diff_y = p01y * inv_area;

				//const float diff_x = (points[0].y - points[2].y) * inv_area;
				//const float diff_y = (points[0].y - points[1].y) * inv_area;

				//const float diff_x = (points[0].y - points[2].y) * inv_area;
				//const float diff_y = (points[0].y - points[1].y) * inv_area;

				//float diff_x = -points[2].x * inv_area;
				//float diff_y = -points[1].x * inv_area;

				//bool first_cross = false;

				for (int x = bmin.x; x <= bmax.x; ++x)
				{
					//float ux = ((points[0].x - points[2].x) * (y - points[0].y) + (points[0].x - x) * (points[0].y - points[2].y)) * inv_area;
					//float uy = ((x - points[0].x) * (points[0].y - points[1].y) + (points[0].x - points[1].x) * (points[0].y - y)) * inv_area;

					u -= diff_x;
					v += diff_y;

					//if (ux < 0.0f || uy < 0.0f || ux + uy > 1.0f)
					//if (u < 0 || v < 0 || u + v > 1.0f || first_cross)

					if (u < 0 || v < 0 || u + v > 1.0f)
						continue;

					//first_cross = true;

					//float a = 1.0f - ux - uy;
					//float b = ux;
					//float c = uy;

					//float z = calculate_depth(camera_coord, a, b, c);

					float z = calculate_depth(camera_point_inv_z, u, v);

					//cout << z << " " << camera->m_far << "\n";

					int pixel_ind = y * w + x;
					if (z >= z_buffer[pixel_ind])// || z < camera->m_near || z > camera->m_far)
						continue;

					vec3 n = normals[0] * (1.0f - u - v) + normals[1] * u + normals[2] * v;

					vec2 screen_point(x, y); //= points[0] * a + points[1] * b + points[2] * c;

											 //vec3 world_point = unrasterize(screen_point, z);

											 //float shadow_value = point_in_shadow(world_point);

					z_buffer[pixel_ind] = z;

					if (render_pass)
						img.data[pixel_ind] = fragment(n);// -1000 * vec3(shadow_value);
				}
			}
		}
	}

	void render_scene_fast_5(Camera*& camera, image& img, vector<float>& z_buffer, bool render_pass)
	{
		vec2i bmin(w1, h1);
		vec2i bmax(0, 0);


		for (int i = 0; i < mesh.num_triangles; ++i)
		{
			vec2 points[3];
			//vec4 camera_point[3];

			vec4 camera_point_inv_z[3];
			vec3 normals[3];

			for (int j = 0; j < 3; ++j)
			{
				int vertex_ind = mesh.faces[9 * i + 3 * j];
				int normal_ind = mesh.faces[9 * i + 3 * j + 2];

				vec3 v = mesh.vertices[vertex_ind];

				normals[j] = mesh.normals[normal_ind];

				//vec4 world_point = model * vec4(v, 1.0f);

				//camera_coord[j] = view_transform(view, v);//world_point);

				//vec4 world_point(v, 1.0f);

				//points[j] = rasterize(view, projection, world_point, camera_coord[j]);

				//points[j] = rasterize(camera, v, camera_point[j]);

				points[j] = rasterize(camera, v, camera_point_inv_z[j]);

				bmin.x = min(bmin.x, points[j].x);
				bmin.y = min(bmin.y, points[j].y);

				bmax.x = max(bmax.x, points[j].x);
				bmax.y = max(bmax.y, points[j].y);
			}

			float area = (points[0].x - points[1].x) * (points[0].y - points[2].y) + (points[2].x - points[0].x) * (points[0].y - points[1].y);

			//front face culling
			//if(area < 0)
			//	continue

			//back face culling
			if (area >= 0)
				continue;

			bmin.x = max(0, bmin.x);
			bmin.y = max(0, bmin.y);

			bmax.x = min(w1, bmax.x);
			bmax.y = min(h1, bmax.y);


			//if (area < 1e-4 && area > -1e-4)



			//bmin.x = (int)bmin.x;
			//bmax.x = (int)bmax.x;

			//bmin.y = (int)bmin.y;
			//bmax.y = (int)bmax.y;

			float inv_area = 1.0f / area;

			//float u = ((points[0].x - points[2].x) * (bmin.y - points[0].y) + (points[0].x - bmin.x) * (points[0].y - points[2].y)) * inv_area;
			//float v = ((bmin.x - points[0].x) * (points[0].y - points[1].y) + (points[0].x - points[1].x) * (points[0].y - bmin.y)) * inv_area;

			//float dx = (points[0].y - points[2].y) * inv_area;
			//float dy = (points[0].y - points[1].y) * inv_area;


			//points[1] -= points[0];
			//points[2] -= points[0];

			//points[1] = -points[1];
			//points[2] = -points[2];

			const float p02x = (points[0].x - points[2].x);
			const float p02y = (points[0].y - points[2].y);

			const float p01x = (points[0].x - points[1].x);
			const float p01y = (points[0].y - points[1].y);

			const float const_u = (p02x * -points[0].y + (points[0].x - bmin.x) * p02y) * inv_area;
			const float const_v = ((bmin.x - points[0].x) * p01y + p01x * points[0].y) * inv_area;

			//const float p02x_inv_area = p02x * inv_area;
			//const float p01x_inv_area = p01x * inv_area;

#pragma omp parallel for schedule(guided)
			for (int y = bmin.y; y <= bmax.y; ++y)
			{

				//good
				//float u = ((points[0].x - points[2].x) * (y - points[0].y) + (points[0].x - bmin.x) * (points[0].y - points[2].y)) * inv_area;
				//float v = ((bmin.x - points[0].x) * (points[0].y - points[1].y) + (points[0].x - points[1].x) * (points[0].y - y)) * inv_area;

				//good
				//float u = (p02x * (y - points[0].y) + (points[0].x - bmin.x) * p02y) * inv_area;
				//float v = ((bmin.x - points[0].x) * p01y + p01x * (points[0].y - y)) * inv_area;

				//good
				float u = const_u + y * p02x * inv_area;
				float v = const_v - y * p01x * inv_area;

				//float u = const_u + y * p02x_inv_area;
				//float v = const_v - y * p01x_inv_area;

				//float u = (-points[2].x * (bmin.y - points[0].y) + (points[0].x - x) * -points[2].y) * inv_area;
				//float v = ((x - points[0].x) * -points[1].y + -points[1].x * (points[0].y - bmin.y)) * inv_area;



				//u -= (points[0].y - points[2].y) * inv_area;
				//v += (points[0].y - points[1].y) * inv_area;


				//u -= dx;
				//v += dy;

				const float diff_x = p02y * inv_area;
				const float diff_y = p01y * inv_area;

				//const float diff_x = (points[0].y - points[2].y) * inv_area;
				//const float diff_y = (points[0].y - points[1].y) * inv_area;

				//const float diff_x = (points[0].y - points[2].y) * inv_area;
				//const float diff_y = (points[0].y - points[1].y) * inv_area;

				//float diff_x = -points[2].x * inv_area;
				//float diff_y = -points[1].x * inv_area;

				//bool first_cross = false;

				for (int x = bmin.x; x <= bmax.x; ++x)
				{
					//float ux = ((points[0].x - points[2].x) * (y - points[0].y) + (points[0].x - x) * (points[0].y - points[2].y)) * inv_area;
					//float uy = ((x - points[0].x) * (points[0].y - points[1].y) + (points[0].x - points[1].x) * (points[0].y - y)) * inv_area;

					u -= diff_x;
					v += diff_y;

					//if (ux < 0.0f || uy < 0.0f || ux + uy > 1.0f)
					//if (u < 0 || v < 0 || u + v > 1.0f || first_cross)

					if (u < 0 || v < 0 || u + v > 1.0f)
						continue;

					//first_cross = true;

					//float a = 1.0f - ux - uy;
					//float b = ux;
					//float c = uy;

					//float z = calculate_depth(camera_coord, a, b, c);

					float z = calculate_depth(camera_point_inv_z, u, v);

					//cout << z << " " << camera->m_far << "\n";

					int pixel_ind = y * w + x;
					if (z >= z_buffer[pixel_ind])// || z < camera->m_near || z > camera->m_far)
						continue;

					vec3 n = normals[0] * (1.0f - u - v) + normals[1] * u + normals[2] * v;

					vec2 screen_point(x, y); //= points[0] * a + points[1] * b + points[2] * c;

											 //vec3 world_point = unrasterize(screen_point, z);

											 //float shadow_value = point_in_shadow(world_point);

					z_buffer[pixel_ind] = z;

					if (render_pass)
						img.data[pixel_ind] = fragment(n);// -1000 * vec3(shadow_value);
				}
			}
		}
	}

	void render_scene_fast_final_version(Camera*& camera, image& img, vector<float>& z_buffer, bool shadow_pass, bool render_pass)
	{
		vec2i bmin(w1, h1);
		vec2i bmax(0, 0);


		for (int i = 0; i < mesh.num_triangles; ++i)
		{
			vec2 points[3];

			//vec4 camera_point_inv_z[3];
			float camera_inv_z[3];
			vec3 normals[3];

			for (int j = 0; j < 3; ++j)
			{
				int vertex_ind = mesh.faces[9 * i + 3 * j];
				int normal_ind = mesh.faces[9 * i + 3 * j + 2];

				vec3 v = mesh.vertices[vertex_ind];

				normals[j] = mesh.normals[normal_ind];

				points[j] = rasterize_2(camera, v, camera_inv_z[j]);

				bmin.x = min(bmin.x, points[j].x);
				bmin.y = min(bmin.y, points[j].y);

				bmax.x = max(bmax.x, points[j].x);
				bmax.y = max(bmax.y, points[j].y);
			}

			float area = (points[0].x - points[1].x) * (points[0].y - points[2].y) + (points[2].x - points[0].x) * (points[0].y - points[1].y);

			//front face culling
			if (shadow_pass && area < 0)
				continue;

			//back face culling
			if (render_pass && area >= 0)
				continue;

			/*if (points[0].y > points[1].y)
			{
			swap(points[0], points[1]);
			swap(camera_inv_z[0], camera_inv_z[1]);
			}
			if (points[1].y > points[2].y)
			{
			swap(points[1], points[2]);
			swap(camera_inv_z[1], camera_inv_z[2]);
			if (points[0].y > points[1].y)
			{
			swap(points[0], points[1]);
			swap(camera_inv_z[0], camera_inv_z[1]);
			}
			}*/

			bmin.x = max(0, bmin.x);
			bmin.y = max(0, bmin.y);

			bmax.x = min(w1, bmax.x);
			bmax.y = min(h1, bmax.y);

			float inv_area = 1.0f / area;

			const float p02x = (points[0].x - points[2].x);
			const float p02y = (points[0].y - points[2].y);

			const float p01x = (points[0].x - points[1].x);
			const float p01y = (points[0].y - points[1].y);

#pragma omp parallel for schedule(guided)
			for (int y = bmin.y; y <= bmax.y; ++y)
			{

				float u = (p02x * (y - points[0].y) + (points[0].x - bmin.x) * p02y) * inv_area;
				float v = ((bmin.x - points[0].x) * p01y + p01x * (points[0].y - y)) * inv_area;

				const float diff_x = p02y * inv_area;
				const float diff_y = p01y * inv_area;

				for (int x = bmin.x; x <= bmax.x; ++x)
				{
					u -= diff_x;
					v += diff_y;

					if (u < 0 || v < 0 || u + v > 1.0f)
						continue;

					float z = calculate_depth_2(camera_inv_z, u, v);


					int pixel_ind = y * w + x;
					if (z >= z_buffer[pixel_ind])
						continue;

					vec3 n = normals[0] * (1.0f - u - v) + normals[1] * u + normals[2] * v;

					vec2 screen_point(x, y); //= points[0] * a + points[1] * b + points[2] * c;

											 //vec3 world_point = unrasterize(screen_point, z);

											 //float shadow_value = point_in_shadow(world_point);

					z_buffer[pixel_ind] = z;

					if (render_pass)
						img.data[pixel_ind] = fragment(n);// -1000 * vec3(shadow_value);
				}
			}
		}
	}

	void render()
	{
		clock_t start = clock();

		//shadow_buffer = new float[sw * sh];
		//memset(shadow_buffer, inf, sizeof(shadow_buffer));

		image depth_img(sw, sh);

		//float* z_buffer = new float[w * h];
		//memset(z_buffer, inf, sizeof(z_buffer));
		image img(w, h);

		vector<float> z_buffer(w * h, inf);

		vector<float> shadow_buffer(w * h, inf);

		float print_time = platform_get_time();
		int num_frame = 0;

		while (!window->is_close)
		{
			float curr_time = platform_get_time();

			handle_event(perspective_camera);

			//update_matrix(model, projection, cam);

			update_camera(perspective_camera);

			++num_frame;

			//66 fps
			//render_scene(perspective_camera, img, z_buffer, true);

			//90 fps
			//116 parallel x

			//render_scene_fast(perspective_camera, img, z_buffer, true);

			//116 fps omp
			//154 fps back face culling
			//render_scene_fast_2(perspective_camera, img, z_buffer, true);

			//failed!
			//render_scene_fast_3(perspective_camera, img, z_buffer, true);

			//158 fps, y before x is faster
			//160 fps, inv_z
			//161 fps, add constant, peak 161 fps
			//render_scene_fast_4(perspective_camera, img, z_buffer, true);

			//160 fps
			//render_scene_fast_5(perspective_camera, img, z_buffer, true);

			render_scene_fast_final_version(perspective_camera, img, z_buffer, false, true);

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

			clean_z_buffer(z_buffer);
			img.clean();

			clean_z_buffer(shadow_buffer);
			depth_img.clean();
		}

		z_buffer.swap(vector<float>());
		shadow_buffer.swap(vector<float>());

		img.data.swap(vector<vec3>());
		depth_img.data.swap(vector<vec3>());
	}

	void clean_z_buffer(vector<float>& z_buffer)
	{
		for (int i = 0; i < w * h; ++i)
			z_buffer[i] = inf;
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