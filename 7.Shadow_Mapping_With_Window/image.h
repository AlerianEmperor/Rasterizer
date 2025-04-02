#ifndef _IMAGE_H_
#define _IMAGE_H_

#include "maths.h"
//#include "maths_linear.h"
#include <vector>
#include <fstream>
//#define STB_IMAGE_IMPLEMENTATION
//#include "stb_image.h"

/*#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
*/
using namespace std;

struct image
{
	int w, h;
	vector<vec3> data;
	//vec3* data;

	image() {}
	image(int w_, int h_) : w(w_), h(h_) 
	{
		data.resize(w * h);

		//data = new vec3[w * h];

		for (int i = 0; i < w * h; ++i)
			data[i] = vec3(0, 0, 0);
	}
	void set(int x, int y, vec3 color)
	{
		data[y * w + x] = color;
	}
	void flip_vertical()
	{
		int half = h >> 1;

		for (int i = 0; i < w; ++i)
		{
			for (int j = 0; j < half; ++j)
				swap(data[i + j * w], data[i + (h - 1 - j) * w]);
		}
	}

	void flip_horizontal()
	{
		int half = w >> 1;

		for (int i = 0; i < half; ++i)
		{
			for (int j = 0; j < h; ++j)
				swap(data[i + j * w], data[w - 1 - i + j * w]);
		}
	}

	void write(string file_name)
	{
		//stbi_write_jpg(file_name.c_str(), w, h, 4, data, 4 * w * h);

		ofstream ofs(file_name + ".ppm");

		ofs << "P3\n" << w << " " << h << "\n255\n";

		//for (auto& v : data)
		for(int i = 0; i < w * h; ++i)
		{
			//if (v.x > 0)
			//	cout << "good\n";
			ofs << data[i].x << " " << data[i].y << " " << data[i].z << "\n";
		}
	}

	void clean()
	{
		for (int i = 0; i < w; ++i)
			for (int j = 0; j < h; ++j)
				data[j * w + i] = vec3(0);
	}
	void clear()
	{
		//delete(data);
		vector<vec3>().swap(data);
	}
};

#endif // !_IMAGE_H_