#ifndef _SCENE_H_
#define _SCENE_H_
#include "Mesh.h"
#include "S_Camera.h"
#include <vector>
#include <fstream>
#include <iostream>

struct Scene
{
	Mesh mesh;

	int w;
	int h;

	int sw;
	int sh;

	int w1;
	int h1;

	glm::vec3 eye = glm::vec3(0.0, 10.0, 10.0);//glm::vec3(10.0, 10.0, 10.0);
	glm::vec3 center = glm::vec3(0.0, 0.0, 0.0);
	glm::vec3 up = glm::vec3(0.0, 1.0, 0.0);
	glm::vec3 light_position;// = glm::vec3(1000.0, 1000.0, 0.0);
	glm::vec3 close_light_poition;
							 //glm::vec3 light_direction = glm::vec3(1.0f, 0.4, -1.0f);
							 //glm::vec3 light_direction = glm::vec3(1.0f, 0.9, -0.5f);
	glm::vec3 light_direction = glm::vec3(1.0f, 1.0f, -0.2f);

	const double DIRECTIONAL_LIGHT_DISTANCE = 1e6;

	Camera * perspective_camera;// = new Perspective(eye, center, 400, 400, 1, 100, 40.0f);
	Camera * shadow_camera;// = new Orthographic(get_light_position(), glm::vec3(0), w, h, 2, 1000000.0 * 2);

	vector<double> shadow_buffer;

	Scene() {}
	Scene(string file_name, int w_, int h_)
	{
		mesh = load_obj(file_name);

		w = w_;
		h = h_;

		w1 = w - 1;
		h1 = h - 1;

		sw = 4 * w;
		sh = 4 * h;

		light_direction = glm::normalize(light_direction);

		light_position = glm::vec3(light_direction.x * DIRECTIONAL_LIGHT_DISTANCE, light_direction.y * DIRECTIONAL_LIGHT_DISTANCE, light_direction.z * DIRECTIONAL_LIGHT_DISTANCE);
		close_light_poition = light_direction * 10.0f;


		perspective_camera = new Perspective(eye, center, w, h, 0.01f, 1e2, 90.0f);
		shadow_camera = new Orthographic(light_position, center, sw, sh, 2.0f, 2.0f * 1e6);

		//shadow_camera = new Orthographic2(light_position, center, sw, sh, -0.5, 0.5, 0.5, -0.5, 0.01, 2 * 1e6);

		//shadow_camera = new Orthographic2(light_position, center, sw, sh, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0);


		shadow_buffer.resize(sw * sh, 1e6);
	}

	void clean_after_frame()
	{
		for (int i = 0; i < w * h; ++i)
			shadow_buffer[i] = 1e6;
	}

	glm::vec3 get_light_position()
	{
		return glm::vec3(light_direction.x * DIRECTIONAL_LIGHT_DISTANCE, light_direction.y * DIRECTIONAL_LIGHT_DISTANCE, light_direction.z * DIRECTIONAL_LIGHT_DISTANCE);
	}

	void compute_barycentric(float& u, float& v, float& w, glm::vec2 p[3], glm::vec2& point) const
	{
		const glm::vec2 v0 = p[1] - p[0], v1 = p[2] - p[0], v2 = point - p[0];
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
	float calculate_depth(float& u, float& v, float& w, glm::vec3 camera_point[3])
	{
		return 1.0f / (1.0f / camera_point[0].z * u + 1.0f / camera_point[1].z * v + 1.0f / camera_point[2].z * w);
	}

	glm::vec2 rasterize(Camera*& camera, glm::vec3& point_world, glm::vec3& camera_point)
	{
		camera_point = camera->viewTransform(point_world);
		glm::vec2 ndc_point = camera->projectTransform(camera_point);
		glm::vec2 raster_point = camera->viewportTransform(ndc_point);
		return raster_point;
	}

	glm::vec3 unrasterize(Camera*& camera, glm::vec2& point_screen, double& depth_camera)
	{
		glm::vec2 ndc_point = camera->viewportTransformInv(point_screen);
		glm::vec3 camera_point = camera->projectTransformInv(ndc_point, depth_camera);
		glm::vec3 world_point = camera->viewTransformInv(camera_point);

		return world_point;
	}

	void triangle_rasterize(Camera*& camera, int& triangle_ind, glm::vec2 screen_point[3], glm::vec3 camera_point[3])
	{
		for (int i = 0; i < 3; ++i)
		{
			int ind = mesh.faces[9 * triangle_ind + 3 * i];

			glm::vec3 world_point = mesh.vertices[ind];

			screen_point[i] = rasterize(camera, world_point, camera_point[i]);
		}
	}

	double in_shadow(Camera*& camera, glm::vec3& world_point, glm::vec3& normal)
	{
		glm::vec3 camera_point;
		glm::vec2 light_point = rasterize(camera, world_point, camera_point);

		double shadow = 0;

		//light_point = 0.5f * light_point + glm::vec2(0.5f);

		uint32_t index = round(light_point.y) * sw + round(light_point.x) + 1;

		if (index < 0 || index >= sw * sh)
			return 0.0f;

		// << index << "\n";

		double closest_depth = shadow_buffer[index];

		//double current_depth = camera_point.z;//(camera_point.z - camera->get_near_plane()) / (camera->get_far_plane() - camera->get_near_plane());

		//glm::vec3 lp = get_light_position() - (world_point + normal * 0.0f);

		glm::vec3 lp = light_position - (world_point + normal );

		double current_depth = sqrt(lp.x * lp.x + lp.y * lp.y + lp.z * lp.z);//(lp - world_point).length();

																			 //double current_depth = camera_point.z;

																			 //cout << closest_depth <<"  "<< current_depth << "\n";

		//lp = glm::normalize(lp);

		//double d = abs(dot(lp, light_direction));

		//current_depth *= d;


		//cout << lp.x << " " << lp.y << " " << lp.z << "\n";

		//cout << world_point.x << " " << world_point.y << " " << world_point.z << "\n";

		//cout << current_depth << " " << closest_depth << "\n";

		//float a = 5.5;

		double bias = maxf(0.05 * (1.0 - abs(dot(normal, light_direction))), 0.005);

		//double bias = maxf(0.5 * (1.0 - d), 0.05);

		shadow += (current_depth > closest_depth + bias) ? 1.0 : 0.0;

		//if (shadow > 0)
		//	cout << "shadow\n";
		//else
		//	cout << "not shadow\n";
		return shadow;
	}

	glm::vec3 compute_fragment(float& u, float& v, float& w, glm::vec3 normal[3])
	{
		return u * normal[0] + v * normal[1] + w * normal[2];
	}

	void write_image(string file_name, vector<glm::vec3>& color)
	{
		ofstream ofs(file_name + ".ppm");

		ofs << "P3\n" << w << " " << h << "\n255\n";

		for (int i = w * h - 1; i >= 0; --i)
			//for (int i = 0; i < w * h; ++i)
			//ofs <<  shadow_depth_buffer[i] << " " <<   shadow_depth_buffer[i] << " " <<  shadow_depth_buffer[i] << "\n";
			ofs << color[i].x << " " << color[i].y << " " << color[i].z << "\n";
	}

	void render_scene(Camera*& camera, int& w_, int h_, bool shadow_pass)
	{
		vector<float> depth_buffer(w_ * h_, camera->get_far_plane());
		vector<glm::vec3>color(w_ * h_, glm::vec3(0));

		for (int i = 0; i < mesh.num_triangles; ++i)
		{
			glm::vec2 screen_point[3];
			glm::vec3 camera_point[3];
			glm::vec3 normal[3];

			triangle_rasterize(camera, i, screen_point, camera_point);

			glm::vec2 bmin(w_ - 1, h_ - 1);
			glm::vec2 bmax(0, 0);

			for (int j = 0; j < 3; ++j)
			{
				bmin.x = minf(bmin.x, screen_point[j].x);
				bmin.y = minf(bmin.y, screen_point[j].y);

				bmax.x = maxf(bmax.x, screen_point[j].x);
				bmax.y = maxf(bmax.y, screen_point[j].y);

				int normal_ind = mesh.faces[9 * i + 3 * j + 2];

				normal[j] = mesh.normals[normal_ind];
			}

			bmin.x = maxf(0.0f, bmin.x);
			bmin.y = maxf(0.0f, bmin.y);

			bmax.x = minf(w_ - 1, bmax.x);
			bmax.y = minf(h_ - 1, bmax.y);

			for (uint32_t x = bmin.x; x <= bmax.x; ++x)
			{
				for (uint32_t y = bmin.y; y <= bmax.y; ++y)
				{
					glm::vec2 p(x, y);

					float a, b, c;

					compute_barycentric(a, b, c, screen_point, p);

					if (a < 0.0f || b < 0.0f || c < 0.0f)
						continue;

					double depth = calculate_depth(a, b, c, camera_point);

					//depth = (depth + 1.0f) * 0.5f;
					int index = y * w_ + x;

					//cout << index << "\n";

					if (depth < depth_buffer[index] && depth >= camera->get_near_plane() && depth <= camera->get_far_plane() && camera->insideFrustrum(p, depth))
					{
						//depth = (depth - camera->get_near_plane()) / (camera->get_far_plane() - camera->get_near_plane());

						depth_buffer[index] = depth;

						if (shadow_pass)
						{
							shadow_buffer[index] = depth;

						}
						else
						{
							//cout << depth << "\n";
							glm::vec3 world_point = unrasterize(perspective_camera, p, depth);

							glm::vec3 n = (a * normal[0] + b * normal[1] + c * normal[2]);

							float shadow_factor = in_shadow(shadow_camera, world_point, n);

						

							float d = glm::dot(n, glm::normalize(close_light_poition - world_point));

							//color[index] = 255.0f * (1.0f - shadow_factor) * compute_fragment(a, b, c, normal);

							color[index] = 255.0f * (1.0f - shadow_factor) * glm::vec3(d);
						}
						//color[index] = 255.0f * compute_fragment(a, b, c, normal);//glm::vec3(0, 255, 0);

						//float d = (depth - camera->get_near_plane()) / (camera->get_far_plane() - camera->get_near_plane());
						//color[index] = 255.0f * glm::vec3(d);
					}
				}
			}
		}
		//getchar();
		if (!shadow_pass)
			write_image("final", color);
	}



	void render()
	{
		render_scene(shadow_camera, sw, sh, true);
		render_scene(perspective_camera, w, h, false);
	}

};

#endif