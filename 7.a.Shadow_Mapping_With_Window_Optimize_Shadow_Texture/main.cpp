#include "Scene_Cube_Windows.h"

#include "win32.h"
#include "camera.h"

void main()
{
	int w = 400, h = 400;
	
	Scene scn("D:\\a_c++Rasterizer\\Models\\shadow_cube_3.obj", w, h);

	scn.render();
}