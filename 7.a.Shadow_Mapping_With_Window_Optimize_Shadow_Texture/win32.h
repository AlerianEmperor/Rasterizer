#ifndef _WIN_32_H_
#define _WIN_32_H_

#include "maths.h"
//#include "maths_linear.h"
#include "image.h"
#include <Windows.h>
#include <assert.h>
#include <string>

using namespace std;

struct Mouse
{
	vec2 orbit_pos;
	vec2 orbit_delta;

	//first person
	vec2 fv_pos;
	vec2 fv_delta;

	float wheel_delta;
};

struct Window
{
	//handle window
	HWND hwnd;
	//handle context
	HDC hdc;
	//handle bitmap
	HBITMAP old_bm;
	HBITMAP new_bm;

	unsigned char* frame_buffer;

	int w;
	int h;
	char keys[512];
	char button[2];
	bool is_close;
	Mouse mouse;
};

Window* window = NULL;

vec2 get_mouse_pos()
{
	POINT point;
	GetCursorPos(&point);
	//convert point from screen space to application screen space
	ScreenToClient(window->hwnd, &point);

	return vec2(point.x, point.y);
}

//Long Result
//Call Back is function that will be sent to window for processing
//word parameter
//long parameter
LRESULT CALLBACK msg_callback(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	switch (msg)
	{
	case WM_CLOSE:
		window->is_close = true;
		break;
	case WM_KEYDOWN:
		window->keys[wParam & 511] = 1;
		break;
	case WM_KEYUP:
		window->keys[wParam & 511] = 0;
		break;
	case WM_LBUTTONDOWN:
		window->mouse.orbit_pos = get_mouse_pos();
		window->button[0] = 1;
		break;
	case WM_LBUTTONUP:
		window->button[0] = 0;
		break;
	case WM_MOUSEWHEEL:
		window->mouse.wheel_delta = GET_WHEEL_DELTA_WPARAM(wParam) / (float)WHEEL_DELTA;
		break;

	default: return DefWindowProc(hwnd, msg, wParam, lParam);
	}
	return 0;
}

//https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-wndclassexa
//style: window display style
//lpfWndProc: long pointer function window procedure
//cbClsExtra: count byte window class extra byte for allocating, default = 0
//cbWndExtra: count byte window instance extra byte for allocating, default = 0
//hbrBackGround: back goround color, clear_color
//lpszMenuName: resource name
//lpszClassName: application name

void register_window_class()
{
	WNDCLASS wnd;

	wnd.style = CS_BYTEALIGNCLIENT;
	wnd.lpfnWndProc = (WNDPROC)msg_callback;
	wnd.cbClsExtra = 0;
	wnd.cbWndExtra = 0;
	wnd.hInstance = GetModuleHandle(NULL);
	wnd.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wnd.hCursor = LoadCursor(NULL, IDC_ARROW);//LoadIcon(NULL, IDC_ARROW);
	wnd.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wnd.lpszMenuName = NULL;
	wnd.lpszClassName = "Window Renderer";

	ATOM atom = RegisterClass(&wnd);

	assert(atom != 0);
}

void msg_dispatch()
{
	MSG msg;

	while (1)
	{
		if (!PeekMessage(&msg, NULL, 0, 0, PM_NOREMOVE)) break;
		if (!GetMessage(&msg, NULL, 0, 0)) break;

		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
}


//origin at upper left
//biPlanes: number of planes for target device, musr be 1
//biBitCount: number of bit per pixel
//bicompression: display format
//biSizeImgae: size of image, set to 0 for uncompress bitmap
void init_bm_header(BITMAPINFOHEADER& bi, int w, int h)
{
	memset(&bi, 0, sizeof(BITMAPINFOHEADER));

	bi.biSize = sizeof(BITMAPINFOHEADER);
	bi.biWidth = w;
	bi.biHeight = -h;
	bi.biPlanes = 1;
	bi.biBitCount = 32;
	bi.biCompression = BI_RGB;
	bi.biSizeImage = 4 * w * h;
}

int init_window(int w, int h, string name)
{
	window = (Window*)malloc(sizeof(Window));
	memset(window, 0, sizeof(Window));

	window->is_close = 0;

	RECT rect = { 0, 0, w, h };

	int wx, wy, sx, sy;

	LPVOID ptr;
	HDC hdc;
	BITMAPINFOHEADER bi;

	register_window_class();

	window->hwnd = CreateWindow("window renderer", name.c_str(), WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX, 0, 0, w, h, NULL, NULL, GetModuleHandle(NULL), NULL);

	assert(window->hwnd != NULL);

	init_bm_header(bi, w, h);

	hdc = GetDC(window->hwnd);
	window->hdc = CreateCompatibleDC(hdc);
	ReleaseDC(window->hwnd, hdc);

	//dib: device independent bitmap
	window->new_bm = CreateDIBSection(window->hdc, (BITMAPINFO*)&bi, DIB_RGB_COLORS, &ptr, 0, 0);
	assert(window->new_bm != NULL);

	//Hbitmap: handle bit map
	//handle to gdi bitmap: graphic device bimap

	window->old_bm = (HBITMAP)SelectObject(window->hdc, window->new_bm);
	window->frame_buffer = (unsigned char*) ptr;

	window->w = w;
	window->h = h;

	AdjustWindowRect(&rect, GetWindowLong(window->hwnd, GWL_STYLE), 0);

	wx = rect.right - rect.left;
	wy = rect.bottom - rect.top;

	//SM_CXSCREEN main display width
	sx = (GetSystemMetrics(SM_CXSCREEN) - wx) / 2;
	sy = (GetSystemMetrics(SM_CYSCREEN) - wy) / 2;


	if (sy < 0)
		sy = 0;

	SetWindowPos(window->hwnd, NULL, sx, sy, wx, wy, (SWP_NOCOPYBITS | SWP_NOZORDER | SWP_SHOWWINDOW));
	SetForegroundWindow(window->hwnd);
	ShowWindow(window->hwnd, SW_NORMAL);

	msg_dispatch();

	memset(window->frame_buffer, 0, 4 * w * h);
	memset(window->keys, 0, 512 * sizeof(char));

	return 0;
}

int window_destroy()
{
	if (window->hdc)
	{
		if (window->old_bm)
		{
			SelectObject(window->hdc, window->old_bm);
			window->old_bm = NULL;
		}

		DeleteDC(window->hdc);
		window->hdc = NULL;
	}
	if (window->new_bm)
	{
		DeleteObject(window->new_bm);
		window->new_bm = NULL;
	}
	if (window->hwnd)
	{
		CloseWindow(window->hwnd);
		window->hwnd = NULL;
	}
	return 0;
}

void window_display()
{
	LOGFONT logfont;

	ZeroMemory(&logfont, sizeof(LOGFONT));

	logfont.lfCharSet = ANSI_CHARSET;
	logfont.lfHeight = 20;
	HFONT hFont = CreateFontIndirect(&logfont);

	HDC hdc = GetDC(window->hwnd);

	SelectObject(window->hdc, hFont);
	SetTextColor(window->hdc, RGB(190, 190, 190));
	SetBkColor(window->hdc, RGB(80, 80, 80));

	//TextOut(window->hdc, 20, 20, "render", strlen("render"));

	//bit block, work similar to atomic
	BitBlt(hdc, 0, 0, window->w, window->h, window->hdc, 0, 0, SRCCOPY);

	ReleaseDC(window->hwnd, hdc);
}

void window_draw(image& img)//(unsigned char* frame_buffer)
{
	for (int i = 0; i < window->h; ++i)
	{
		for (int j = 0; j < window->w; ++j)
		{
			int index = (i * window->w + j);

			int index_4 = 4 * index;

			//window store color as BGR: bigera
			window->frame_buffer[index_4] = (unsigned char)img.data[index].z;//frame_buffer[index + 2];
			window->frame_buffer[index_4 + 1] =  (unsigned char)img.data[index].y;
			window->frame_buffer[index_4 + 2] =  (unsigned char)img.data[index].x;
		}
	}

	window_display();
}

double get_native_time()
{
	static double period = -1;
	LARGE_INTEGER counter;

	if (period < 0)
	{
		LARGE_INTEGER frequency;

		QueryPerformanceFrequency(&frequency);

		period = 1.0f / (double)frequency.QuadPart;
	}

	QueryPerformanceCounter(&counter);

	return counter.QuadPart * period;
}

float platform_get_time()
{
	static double start = -1;
	
	if (start < 0)
	{
		start =	get_native_time();
	}
	return (float)(get_native_time() - start);
}



#endif // !_WIN_32_H_

