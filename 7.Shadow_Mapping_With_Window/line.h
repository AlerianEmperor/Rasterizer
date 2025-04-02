#ifndef _LINE_H_
#define _LINE_H_

#include <iostream>
#include <omp.h>
#include "image.h"

using namespace std;

void line(int x0, int y0, int x1, int y1, image& img, vec3 color)
{
	for (float t = 0; t <= 1.0f; t += 0.01)
	{
		int x = x0 + (x1 - x0) * t;
		int y = y0 + (y1 - y0) * t;

		img.set(x, y, color);
	}
}

void line2(int x0, int y0, int x1, int y1, image& img, vec3 color)
{
	for (int x = x0; x <= x1; ++x)
	{
		float t = (x - x0) / (float)(x1 - x0);
		int y = y0 * (1 - t) + y1 * t;

		img.set(x, y, color);
	}
}

void line3(int x0, int y0, int x1, int y1, image& img, vec3 color)
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
	for (int x = x0; x <= x1; ++x)
	{
		float t = (x - x0) / (float)(x1 - x0);

		int y = (1 - t) * y0 + t * y1;

		if (steep)
			img.set(y, x, color);
		else
			img.set(x, y, color);
	}
}

void line4(int x0, int y0, int x1, int y1, image& img, vec3 color)
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

	float derror = abs(dy / (float)dx);
	float error = 0;

	int y = y0;

	for (int x = x0; x <= x1; ++x)
	{
		//float t = (x - x0) / (float)(x1 - x0);

		//int y = (1 - t) * y0 + t * y1;

		//error += derror;

		//cout << x << " " << y << "\n";
		if (steep)
			img.set(y, x, color);
		else
			img.set(x, y, color);

		error += derror;

		if (error > 0.5)
		{
			y += y1 > y0 ? 1 : -1;
			error -= 1;
		}
	}
}

void line5(int x0, int y0, int x1, int y1, image& img, vec3 color)
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

	int derror = abs(dy) * 2;
	int error = 0;

	int y = y0;

	const int incr = y1 > y0 ? 1 : -1;
	/*for (int x = x0; x <= x1; ++x)
	{
	//float t = (x - x0) / (float)(x1 - x0);

	//int y = (1 - t) * y0 + t * y1;

	//error += derror;

	if (steep)
	img.set(y, x, color);
	else
	img.set(x, y, color);

	error += derror;

	if (error > dx)
	{
	//y += y1 > y0 ? 1 : -1;
	y += incr;
	error -= dx * 2;
	}
	}*/

	if (steep)
	{
#pragma parallel for schedule(guided)
		for (int x = x0; x <= x1; ++x)
		{
			img.set(y, x, color);

			error += derror;

			if (error > dx)
			{
				y += incr;
				error -= dx * 2;
			}
		}
	}
	else
	{
#pragma parallel for schedule(guided)
		for (int x = x0; x <= x1; ++x)
		{
			img.set(x, y, color);

			error += derror;

			if (error > dx)
			{
				y += incr;
				error -= dx * 2;
			}
		}
	}
}



#endif // !_LINE_H_
