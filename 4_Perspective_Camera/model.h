#ifndef __MODEL_H__
#define __MODEL_H__

#include <vector>
#include "geometry.h"
using namespace std;
class Model {
private:
	std::vector<vec3> verts;
	vector<vec3> normals;
	std::vector<vec2> texs;
	std::vector<std::vector<int>> faces_;
	std::vector<std::vector<int>> vt_indices;
	std::vector<std::vector<int>> vn_indices;

public:
	Model(const char *filename);
	~Model();
	int nverts();
	int nfaces();
	vec3 vert(int i);
	vec2 tex(int i);
	vec3 normal_value(int face_index, int vertex_index);
	std::vector<int> face(int idx);
	std::vector<int> vt_index(int idx);
};

#endif //__MODEL_H__