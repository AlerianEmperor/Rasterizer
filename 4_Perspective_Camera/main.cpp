#include "tgaimage.h"
#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include "model.h"
#include "Matrix.h"

using namespace std;

TGAColor white(255, 255, 255, 255);
TGAColor red(255, 0, 0, 255);
TGAColor green(0, 255, 0, 255);
TGAColor bl(0, 0, 255, 255);

const int width = 800;
const int height = 800;
const int depth = 255;

vec3 eye(1, 2, 3);
vec3 center(0, 0, 0);
vec3 up(0, 1, 0);

Model* model = NULL;

//P = A + AB * u + AC * v
//PA + AB * u + AC * v = 0

//            [AB]
//[u, v, 1] * [AC] = 0
//            [PA]
//cross(PAx, ABx, ACx) and (PAy, ABy, ACy)

int min(int a, int b)
{
	return a < b ? a : b;
}

int max(int a, int b)
{
	return a > b ? a : b;
}

/*float maxf(float a, float b)
{
	return a > b ? a : b;
}

float minf(float a, float b)
{
	return a > b ? a : b;
}*/

void line(int x0, int y0, int x1, int y1, TGAImage& image, TGAColor color)
{
	bool steep = false;

	if (abs(x0 - x1) < abs(y0 - y1))
	{
		swap(x0, y0);
		swap(x1, y1);
		steep = true;
	}
	if (x0 > x1)
	{
		swap(x0, x1);
		swap(y0, y1);
	}
	int dx = x1 - x0;
	int dy = y1 - y0;

	int d_error2 = 2 * abs(dy);
	int error2 = 0;

	int y = y0;

	int direction_y = y1 > y0 ? 1 : -1;

	for (int x = x0; x <= x1; ++x)
	{
		if (steep)
			image.set(y, x, color);
		else
			image.set(x, y, color);

		error2 += d_error2;
		if (error2 > dx)
		{
			y += direction_y;
			error2 -= 2 * dx;
		}
	}
}

void line(vec2i v0, vec2i v1, TGAImage& image, TGAColor color)
{
	bool steep = false;

	if (abs(v0.x - v1.x) < abs(v0.y - v1.y))
	{
		swap(v0.x, v0.y);
		swap(v1.x, v1.y);
		steep = true;
	}
	if (v0.x > v1.x)
	{
		swap(v0.x, v1.x);
		swap(v0.y, v1.y);
	}
	int dx = v1.x - v0.x;
	int dy = v1.y - v0.y;

	int d_error2 = 2 * abs(dy);
	int error2 = 0;

	int y = v0.y;

	//POSIX deperecate
	//int direction_y = y1 > y0 ? 1 : -1;

	int direction_y = v1.y > v0.y ? 1 : -1;

	for (int x = v0.x; x <= v1.x; ++x)
	{
		if (steep)
			image.set(y, x, color);
		else
			image.set(x, y, color);

		error2 += d_error2;
		if (error2 > dx)
		{
			y += direction_y;
			error2 -= 2 * dx;
		}
	}
}

vec3 bary_centric_coordinate(vec2i* pts, vec2i P)
{
	vec3 u = vec3(pts[1][0] - pts[0][0], pts[2][0] - pts[0][0], pts[0][0] - P[0]) ^ vec3(pts[1][1] - pts[0][1], pts[2][1] - pts[0][1], pts[0][1] - P[1]);

	if (abs(u.z) < 1)
		return vec3(-1, 1, 1);
	return vec3(u.x / u.z, u.y / u.z, 1.0f - (u.x + u.y) / u.z);
}

vec3 bary_centric_coordinate(vec3* pts, vec3 P)
{
	vec3 u = vec3(pts[1][0] - pts[0][0], pts[2][0] - pts[0][0], pts[0][0] - P[0]) ^ vec3(pts[1][1] - pts[0][1], pts[2][1] - pts[0][1], pts[0][1] - P[1]);

	if (abs(u.z) < 1)
		return vec3(-1, 1, 1);
	return vec3(u.x / u.z, u.y / u.z, 1.0f - (u.x + u.y) / u.z);
}

void triangle(vec2i* pts, TGAImage& image, TGAColor color)
{
	vec2i box_min(image.get_width() - 1, image.get_height() - 1);
	vec2i box_max(0, 0);

	vec2i clamp(image.get_width() - 1, image.get_height() - 1);

	for (int i = 0; i < 3; ++i)
	{
		box_min.x = max(0, min(box_min.x, pts[i][0]));
		box_min.y = max(0, min(box_min.y, pts[i][1]));

		box_max.x = min(clamp.x, max(box_max.x, pts[i][0]));
		box_max.y = min(clamp.y, max(box_max.y, pts[i][1]));
	}
	vec2i P;

	for(P.x = box_min.x; P.x <= box_max.x; ++P.x)
	{
		for(P.y = box_min.y; P.y <= box_max.y; ++P.y)
		{
			vec3 coord = bary_centric_coordinate(pts, P);

			if (coord.x < 0.0f || coord.y < 0.0f || coord.z < 0.0f)
				continue;
			image.set(P.x, P.y, color);
		}
	}
}

void rasterize(vec2i p0, vec2i p1, TGAImage& image, TGAImage& ybuffer_image, TGAColor color, int ybuffer[])
{
	if (p0.x > p1.y)
		swap(p0, p1);
	for (int x = p0.x; x <= p1.x; ++x)
	{
		float t = (x - p0.x) / (float)(p1.x - p0.x);
		int y = p0.y * (1.0f - t) + p1.y * t;

		if (ybuffer[x] < y)
		{
			ybuffer[x] = y;
			image.set(x, 0, color);
			//TGAColor buffer_color( y,  y, y, 255);
			//ybuffer_image.set(x, 0, buffer_color);
		}
	}
}

void triangle_rasterize(vec3* pts, float* zbuffer, TGAImage& image, TGAColor color)
{
	vec2i box_min(image.get_width() - 1, image.get_height() - 1);
	vec2i box_max(0, 0);

	vec2i clamp(image.get_width() - 1, image.get_height() - 1);

	for (int i = 0; i < 3; ++i)
	{
		box_min.x = max(0, min(box_min.x, pts[i][0]));
		box_min.y = max(0, min(box_min.y, pts[i][1]));

		box_max.x = min(clamp.x, max(box_max.x, pts[i][0]));
		box_max.y = min(clamp.y, max(box_max.y, pts[i][1]));
	}
	vec3 P;

	for (P.x = box_min.x; P.x <= box_max.x; ++P.x)
	{
		for (P.y = box_min.y; P.y <= box_max.y; ++P.y)
		{
			vec3 coord = bary_centric_coordinate(pts, P);

			if (coord.x < 0.0f || coord.y < 0.0f || coord.z < 0.0f)
				continue;
			P.z = 0;

			for (int i = 0; i < 3; ++i)
			{
				P.z += pts[i].z * coord[i];
				if (zbuffer[int(P.x + P.y * width)] < P.z)
				{
					zbuffer[int(P.x + P.y * width)] = P.z;
					image.set(P.x, P.y, color);
				}
			}
		}
	}
}

void triangle_rasterize_texture(vec3* pts, vec2* texs, vec3* texture_values, float* zbuffer, TGAImage& image, int tex_width, int tex_height)
{
	vec2i box_min(image.get_width() - 1, image.get_height() - 1);
	vec2i box_max(0, 0);

	vec2i clamp(image.get_width() - 1, image.get_height() - 1);

	for (int i = 0; i < 3; ++i)
	{
		box_min.x = max(0, min(box_min.x, pts[i][0]));
		box_min.y = max(0, min(box_min.y, pts[i][1]));

		box_max.x = min(clamp.x, max(box_max.x, pts[i][0]));
		box_max.y = min(clamp.y, max(box_max.y, pts[i][1]));
	}
	vec3 P;
	vec2 texcoord;

	for (P.x = box_min.x; P.x <= box_max.x; ++P.x)
	{
		for (P.y = box_min.y; P.y <= box_max.y; ++P.y)
		{
			vec3 coord = bary_centric_coordinate(pts, P);

			if (coord.x < 0.0f || coord.y < 0.0f || coord.z < 0.0f)
				continue;
			P.z = 0;

			texcoord.x = 0;
			texcoord.y = 0;

			for (int i = 0; i < 3; ++i)
			{
				P.z += pts[i].z * coord[i];
				//if(i < 2)
				//texcoord = texcoord + texs[i] * coord[(i + 1) % 3];
			}

			//texcoord = texs[1] * (1.0f - coord[1] - coord[2]) + texs[2] * coord[1] + texs[0] * coord[2];

			texcoord = texs[1] * coord[0] + texs[2] * coord[1] + texs[0] * coord[2];

			
			if (zbuffer[int(P.x + P.y * width)] < P.z)
			{
				zbuffer[int(P.x + P.y * width)] = P.z;

				int tex_x = texcoord.x * tex_width;
				int tex_y = texcoord.y * tex_height;

				int tex_location = tex_x + tex_y * tex_width;
				vec3 c = texture_values[tex_location];

				

				TGAColor color(c.x, c.y, c.z, 255);
				image.set(P.x, P.y, color);
			}
		}
	}
}

void triangle_rasterize_texture_with_lighting(vec3* pts, vec2* texs, vec3* texture_values, float* zbuffer, float intensity[3], TGAImage& image, int tex_width, int tex_height)
{
	vec2i box_min(image.get_width() - 1, image.get_height() - 1);
	vec2i box_max(0, 0);

	vec2i clamp(image.get_width() - 1, image.get_height() - 1);

	for (int i = 0; i < 3; ++i)
	{
		box_min.x = max(0, min(box_min.x, pts[i][0]));
		box_min.y = max(0, min(box_min.y, pts[i][1]));

		box_max.x = min(clamp.x, max(box_max.x, pts[i][0]));
		box_max.y = min(clamp.y, max(box_max.y, pts[i][1]));
	}
	vec3 P;
	vec2 texcoord;

	for (P.x = box_min.x; P.x <= box_max.x; ++P.x)
	{
		for (P.y = box_min.y; P.y <= box_max.y; ++P.y)
		{
			vec3 coord = bary_centric_coordinate(pts, P);

			if (coord.x < 0.0f || coord.y < 0.0f || coord.z < 0.0f)
				continue;
			P.z = 0;

			texcoord.x = 0;
			texcoord.y = 0;

			for (int i = 0; i < 3; ++i)
			{
				P.z += pts[i].z * coord[i];
				//if(i < 2)
				//texcoord = texcoord + texs[i] * coord[(i + 1) % 3];
			}

			//texcoord = texs[1] * (1.0f - coord[1] - coord[2]) + texs[2] * coord[1] + texs[0] * coord[2];

			texcoord = texs[1] * coord[0] + texs[2] * coord[1] + texs[0] * coord[2];
			float intense = intensity[1] * coord[0] + intensity[2] * coord[1] + intensity[0] * coord[2];

			if (zbuffer[int(P.x + P.y * width)] < P.z)
			{
				zbuffer[int(P.x + P.y * width)] = P.z;

				int tex_x = texcoord.x * tex_width;
				int tex_y = texcoord.y * tex_height;

				int tex_location = tex_x + tex_y * tex_width;
				vec3 c = texture_values[tex_location];

				c *= intense;
				/*c.x = c.x * intensity[0];// *255;
				c.y = c.y * intensity[1];// *255;
				c.z = c.z * intensity[2];// *255;*/

				//cout << c.x *intensity[0] << " " << c.y *intensity[1] << " " << c.z *intensity[2] << "\n";
				//*/

				TGAColor color(c.x, c.y, c.z, 255);
				image.set(P.x, P.y, TGAColor(c.x, c.y, c.z, 255) );
			}
		}
	}
}


/*vec3 world_to_screen(vec3 v)
{
	return vec3((v.x + 1.0f) * width / 2 + 0.5f, (v.y + 1.0f) * height / 2 + 0.5f, v.z);
}*/

int main_texture_only()
{
	Model* model = new Model("D:/a_c++Rasterizer/Models/african_head.obj");
	TGAImage image(width, height, TGAImage::RGB);
	
	//unsigned char* tex_data = TGAImage::read_tga_file("D:/a_c++Rasterizer/Models/african_head_diffuse.tga");

	TGAImage tex_data;
	tex_data.read_tga_file("D:/a_c++Rasterizer/Models/african_head_diffuse.tga");

	int tex_w = tex_data.get_width();
	int tex_h = tex_data.get_height();
	vec3* tex_values = new vec3[tex_w * tex_h];

	int k = 0;
	for (int y = 0; y < tex_h; ++y)
	{
		for (int x = 0; x < tex_w; ++x)
		{
			TGAColor color = tex_data.get(x, y);
			//cout << (int)color.r << " " << (int)color.g << " " << (int)color.b << "\n";
			tex_values[k++] = vec3((int)color.r , (int)color.g , (int)color.b);
		}
	}

	float* zbuffer = new float[width * height];

	for (int i = width * height; i >= 0; --i)
		zbuffer[i] = -std::numeric_limits<float>::max();

	Matrix model_view = lookat(eye, center, up);
	Matrix projection = perspective_projection(eye, center);
	Matrix view_port = viewport(width / 8, height / 8, 3 * width / 4, 3 * height / 4, depth);


	vec3 light_dir(0, 0, -1);
	for (int i = 0; i < model->nfaces(); ++i)
	{
		vector<int> face = model->face(i);
		vector<int> vt_index = model->vt_index(i);

		vec3 world_coord[3];
		vec3 pts[3];
		vec2 vts[3];

		for (int j = 0; j < 3; ++j)
		{
			vec3 v = model->vert(face[j]);
			vts[j] = model->tex(vt_index[j]);

			//cout << vts[j].x << " " << vts[j].y << " " << vts[j].z << "\n";

			//screen_coord[j] = vec2i((v.x + 1) * width / 2.0f, (v.y + 1) * height / 2.0f);
			world_coord[j] = v;
		
			//pts[j] = world_to_screen(v);
			pts[j] = m2v(view_port * projection * model_view * v2m(v));

			
		}
		vec3 n = (world_coord[2] - world_coord[0]) ^ (world_coord[1] - world_coord[0]);

		n.normalize();
		//float intensity = n.dot(light_dir);
		//if (intensity > 0)
			//triangle_rasterize(pts, zbuffer, image, TGAColor(intensity * 255, intensity * 255, intensity * 255, 255));
		triangle_rasterize_texture(pts, vts, tex_values, zbuffer, image, tex_w, tex_h);
	}

	image.flip_vertically();
	image.write_tga_file("african_head_light_rasterize_texture_circular_permutation_perspective_camera.tga");
	return 0;
}

int main_shading()
{
	Model* model = new Model("D:/a_c++Rasterizer/Models/african_head.obj");
	TGAImage image(width, height, TGAImage::RGB);

	//unsigned char* tex_data = TGAImage::read_tga_file("D:/a_c++Rasterizer/Models/african_head_diffuse.tga");

	TGAImage tex_data;
	tex_data.read_tga_file("D:/a_c++Rasterizer/Models/african_head_diffuse.tga");

	int tex_w = tex_data.get_width();
	int tex_h = tex_data.get_height();
	vec3* tex_values = new vec3[tex_w * tex_h];

	int k = 0;
	for (int y = 0; y < tex_h; ++y)
	{
		for (int x = 0; x < tex_w; ++x)
		{
			TGAColor color = tex_data.get(x, y);
			//cout << (int)color.r << " " << (int)color.g << " " << (int)color.b << "\n";
			tex_values[k++] = vec3((int)color.r, (int)color.g, (int)color.b);
		}
	}

	float* zbuffer = new float[width * height];

	for (int i = width * height; i >= 0; --i)
		zbuffer[i] = -std::numeric_limits<float>::max();

	Matrix model_view = lookat(eye, center, up);
	Matrix projection = perspective_projection(eye, center);
	Matrix view_port = viewport(width / 8, height / 8, 3 * width / 4, 3 * height / 4, depth);

	float intensity[3];

	vec3 light_dir(0, 0, 1);
	for (int i = 0; i < model->nfaces(); ++i)
	{
		vector<int> face = model->face(i);
		vector<int> vt_index = model->vt_index(i);

		vec3 world_coord[3];
		vec3 pts[3];
		vec2 vts[3];

		float intense[3];
		for (int j = 0; j < 3; ++j)
		{
			vec3 v = model->vert(face[j]);
			vts[j] = model->tex(vt_index[j]);	

			world_coord[j] = v;
		
			pts[j] = m2v(view_port * projection * model_view * v2m(v));

			intense[j] = maxf(0, model->normal_value(i, j).dot(light_dir));
			//cout << intense[j] << "\n";
			//cout << model->normal_value(i, j).dot(light_dir) << "\n";
		}

		//cout << intense[0] << " " << intense[1] << " " << intense[2] << "\n";
		//vec3 n = (world_coord[2] - world_coord[0]) ^ (world_coord[1] - world_coord[0]);

		//n.normalize();
		
		//float intensity = n.dot(light_dir);//max(0, n.dot(light_dir));
		//cout << intensity << "\n";

		/*intensity[0] = n.dot(light_dir) * 255;
		intensity[1] = n.dot(light_dir) * 255;
		intensity[2] = n.dot(light_dir) * 255;*/
		//if (intensity > 0)
		//triangle_rasterize(pts, zbuffer, image, TGAColor(intensity * 255, intensity * 255, intensity * 255, 255));
		//vec3 intensity(intense[0], intense[1], intense[2]);
		triangle_rasterize_texture_with_lighting(pts, vts, tex_values, zbuffer, intense, image, tex_w, tex_h);
	}
	//getchar();
	image.flip_vertically();
	image.write_tga_file("african_head_light_rasterize_texture_circular_permutation_perspective_camera_with_lighting.tga");

	return 0;
}

void main()
{
	//main_texture_only();
	main_shading();
}