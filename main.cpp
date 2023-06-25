#include "tgaimage.h"
#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include "model.h"
#include "geometry.h"
using namespace std;

TGAColor white(255, 255, 255, 255);
TGAColor red(255, 0, 0, 255);
TGAColor green(0, 255, 0, 255);
TGAColor bl(0, 0, 255, 255);

const int width = 800;
const int height = 800;

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

void line(Vec2i v0, Vec2i v1, TGAImage& image, TGAColor color)
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

Vec3f bary_centric_coordinate(Vec2i* pts, Vec2i P)
{
	Vec3f u = Vec3f(pts[1][0] - pts[0][0], pts[2][0] - pts[0][0], pts[0][0] - P[0]) ^ Vec3f(pts[1][1] - pts[0][1], pts[2][1] - pts[0][1], pts[0][1] - P[1]);

	if (abs(u.z) < 1)
		return Vec3f(-1, 1, 1);
	return Vec3f(u.x / u.z, u.y / u.z, 1.0f - (u.x + u.y) / u.z);
}

Vec3f bary_centric_coordinate(Vec3f* pts, Vec3f P)
{
	Vec3f u = Vec3f(pts[1][0] - pts[0][0], pts[2][0] - pts[0][0], pts[0][0] - P[0]) ^ Vec3f(pts[1][1] - pts[0][1], pts[2][1] - pts[0][1], pts[0][1] - P[1]);

	if (abs(u.z) < 1)
		return Vec3f(-1, 1, 1);
	return Vec3f(u.x / u.z, u.y / u.z, 1.0f - (u.x + u.y) / u.z);
}

void triangle(Vec2i* pts, TGAImage& image, TGAColor color)
{
	Vec2i box_min(image.get_width() - 1, image.get_height() - 1);
	Vec2i box_max(0, 0);

	Vec2i clamp(image.get_width() - 1, image.get_height() - 1);

	for (int i = 0; i < 3; ++i)
	{
		box_min.x = max(0, min(box_min.x, pts[i][0]));
		box_min.y = max(0, min(box_min.y, pts[i][1]));

		box_max.x = min(clamp.x, max(box_max.x, pts[i][0]));
		box_max.y = min(clamp.y, max(box_max.y, pts[i][1]));
	}
	Vec2i P;

	for(P.x = box_min.x; P.x <= box_max.x; ++P.x)
	{
		for(P.y = box_min.y; P.y <= box_max.y; ++P.y)
		{
			Vec3f coord = bary_centric_coordinate(pts, P);

			if (coord.x < 0.0f || coord.y < 0.0f || coord.z < 0.0f)
				continue;
			image.set(P.x, P.y, color);
		}
	}
}

void rasterize(Vec2i p0, Vec2i p1, TGAImage& image, TGAImage& ybuffer_image, TGAColor color, int ybuffer[])
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

void triangle_rasterize(Vec3f* pts, float* zbuffer, TGAImage& image, TGAColor color)
{
	Vec2i box_min(image.get_width() - 1, image.get_height() - 1);
	Vec2i box_max(0, 0);

	Vec2i clamp(image.get_width() - 1, image.get_height() - 1);

	for (int i = 0; i < 3; ++i)
	{
		box_min.x = max(0, min(box_min.x, pts[i][0]));
		box_min.y = max(0, min(box_min.y, pts[i][1]));

		box_max.x = min(clamp.x, max(box_max.x, pts[i][0]));
		box_max.y = min(clamp.y, max(box_max.y, pts[i][1]));
	}
	Vec3f P;

	for (P.x = box_min.x; P.x <= box_max.x; ++P.x)
	{
		for (P.y = box_min.y; P.y <= box_max.y; ++P.y)
		{
			Vec3f coord = bary_centric_coordinate(pts, P);

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

void triangle_rasterize_texture(Vec3f* pts, Vec3f* texs, Vec3f* texture_values, float* zbuffer, TGAImage& image, int tex_width, int tex_height)
{
	Vec2i box_min(image.get_width() - 1, image.get_height() - 1);
	Vec2i box_max(0, 0);

	Vec2i clamp(image.get_width() - 1, image.get_height() - 1);

	for (int i = 0; i < 3; ++i)
	{
		box_min.x = max(0, min(box_min.x, pts[i][0]));
		box_min.y = max(0, min(box_min.y, pts[i][1]));

		box_max.x = min(clamp.x, max(box_max.x, pts[i][0]));
		box_max.y = min(clamp.y, max(box_max.y, pts[i][1]));
	}
	Vec3f P;
	Vec3f texcoord;

	for (P.x = box_min.x; P.x <= box_max.x; ++P.x)
	{
		for (P.y = box_min.y; P.y <= box_max.y; ++P.y)
		{
			Vec3f coord = bary_centric_coordinate(pts, P);

			if (coord.x < 0.0f || coord.y < 0.0f || coord.z < 0.0f)
				continue;
			P.z = 0;

			texcoord.x = 0;
			texcoord.y = 0;

			for (int i = 0; i < 3; ++i)
			{
				P.z += pts[i].z * coord[i];
				//if(i < 2)

			        // Error 1
				//This afican head need circular rotation to have the right texture
				//texcoord = texcoord + texs[i] * coord[i];
			}
			//Fix Error 1
			texcoord = texs[1] * coord[0] + texs[2] * coord[1] + texs[0] * coord[2];
			
			
			if (zbuffer[int(P.x + P.y * width)] < P.z)
			{
				zbuffer[int(P.x + P.y * width)] = P.z;

				int tex_x = texcoord.x * tex_width;
				int tex_y = texcoord.y * tex_height;

				int tex_location = tex_x + tex_y * tex_width;
				Vec3f c = texture_values[tex_location];
				TGAColor color(c.x, c.y, c.z, 255);
				image.set(P.x, P.y, color);
			}
		}
	}
}

int main1()
{
	//1
	//Scene
	
	TGAImage scene(width, height, TGAImage::RGB);

	// scene "2d mesh"
	line(Vec2i(20, 34), Vec2i(744, 400), scene, red);
	line(Vec2i(120, 434), Vec2i(444, 400), scene, green);
	line(Vec2i(330, 493), Vec2i(594, 200), scene, bl);

	// screen line
	//line(Vec2i(10, 10), Vec2i(790, 10), scene, white);

	scene.flip_vertically(); // i want to have the origin at the left bottom corner of the image
	scene.write_tga_file("scene3.tga");
	
	//2 Renderer
	TGAImage renderer(width, 16, TGAImage::RGB);
	TGAImage ybuffer_value(width, 16, TGAImage::RGB);

	int ybuffer[width];

	for (int y = 0; y < width; ++y)
		ybuffer[y] = std::numeric_limits<int>::min();//INT_MIN;

	//rasterize(Vec2i(20, 34), Vec2i(744, 400), renderer, ybuffer_value, red, ybuffer);
	//rasterize(Vec2i(120, 434), Vec2i(444, 400), renderer, ybuffer_value, green, ybuffer);
	rasterize(Vec2i(330, 463), Vec2i(594, 200), renderer, ybuffer_value, bl, ybuffer);

	renderer.flip_vertically();
	renderer.write_tga_file("render3.tga");

	//ybuffer_value.flip_vertically();
	//ybuffer_value.write_tga_file("ybuffer3.tga");

	/*Model* model = new Model("african_head.obj");
	TGAImage image(width, height, TGAImage::RGB);

	Vec3f light_dir(0, 0, -1);
	for (int i = 0; i < model->nfaces(); ++i)
	{
		vector<int> face = model->face(i);
		Vec2i screen_coord[3];
		Vec3f world_coord[3];

		for (int j = 0; j < 3; ++j)
		{
			Vec3f v = model->vert(face[j]);
			screen_coord[j] = Vec2i((v.x + 1) * width / 2.0f, (v.y + 1) * height / 2.0f);
			world_coord[j] = v;
		}
		Vec3f n = (world_coord[2] - world_coord[0]) ^ (world_coord[1] - world_coord[0]);

		n.normalize();
		float intensity = n*light_dir;
		if(intensity > 0)
		triangle(screen_coord, image, TGAColor(intensity * 255, intensity * 255, intensity * 255, 255));
	}

	image.flip_vertically();
	image.write_tga_file("african_head_light.tga");*/
}

Vec3f world_to_screen(Vec3f v)
{
	return Vec3f((v.x + 1.0f) * width / 2 + 0.5f, (v.y + 1.0f) * height / 2 + 0.5f, v.z);
}

int main()
{
	Model* model = new Model("D:/a_c++Rasterizer/Models/african_head.obj");
	TGAImage image(width, height, TGAImage::RGB);
	
	//unsigned char* tex_data = TGAImage::read_tga_file("D:/a_c++Rasterizer/Models/african_head_diffuse.tga");

	TGAImage tex_data;
	tex_data.read_tga_file("D:/a_c++Rasterizer/Models/african_head_diffuse.tga");

	int tex_w = tex_data.get_width();
	int tex_h = tex_data.get_height();
	Vec3f* tex_values = new Vec3f[tex_w * tex_h];

	int k = 0;
	for (int y = 0; y < tex_h; ++y)
	{
		for (int x = 0; x < tex_w; ++x)
		{
			TGAColor color = tex_data.get(x, y);
			//cout << (int)color.r << " " << (int)color.g << " " << (int)color.b << "\n";
			tex_values[k++] = Vec3f((int)color.r , (int)color.g , (int)color.b);
		}
	}

	float* zbuffer = new float[width * height];

	for (int i = width * height; i >= 0; --i)
		zbuffer[i] = -std::numeric_limits<float>::max();

	Vec3f light_dir(0, 0, -1);
	for (int i = 0; i < model->nfaces(); ++i)
	{
		vector<int> face = model->face(i);
		vector<int> vt_index = model->vt_index(i);

		Vec3f world_coord[3];
		Vec3f pts[3];
		Vec3f vts[3];

		for (int j = 0; j < 3; ++j)
		{
			Vec3f v = model->vert(face[j]);
			vts[j] = model->tex(vt_index[j]);

			//cout << vts[j].x << " " << vts[j].y << " " << vts[j].z << "\n";

			//screen_coord[j] = Vec2i((v.x + 1) * width / 2.0f, (v.y + 1) * height / 2.0f);
			world_coord[j] = v;
		
			pts[j] = world_to_screen(v);
		}
		Vec3f n = (world_coord[2] - world_coord[0]) ^ (world_coord[1] - world_coord[0]);

		n.normalize();
		float intensity = n*light_dir;
		if (intensity > 0)
			//triangle_rasterize(pts, zbuffer, image, TGAColor(intensity * 255, intensity * 255, intensity * 255, 255));
			triangle_rasterize_texture(pts, vts, tex_values, zbuffer, image, tex_w, tex_h);
	}

	image.flip_vertically();
	image.write_tga_file("african_head_light_rasterize_texture.tga");
}
