#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include "model.h"

Model::Model(const char *filename) : verts(), faces_() {
	std::ifstream in;
	in.open(filename, std::ifstream::in);
	if (in.fail()) return;
	std::string line;
	while (!in.eof()) {
		std::getline(in, line);
		std::istringstream iss(line.c_str());
		char trash;
		if (!line.compare(0, 2, "v ")) {
			iss >> trash;

			float x[3];
			for (int i = 0; i<3; i++) iss >> x[i];

			vec3 v(x[0], x[1], x[2]);
			verts.push_back(v);
		}
		else if (!line.compare(0, 2, "vt"))
		{
			iss >> trash >> trash;
			vec2 vt;
			float x[2];
			for (int i = 0; i < 2; i++)
				iss >> x[i];
			//std::cout << vt.x << " " << vt.y << " " << vt.z << " " << "\n";
			//flip y
			vt.x = x[0];
			vt.y = x[1];

			vt.y = 1.0f - vt.y;

			texs.push_back(vt);
		}
		else if (!line.compare(0, 2, "vn"))
		{
			iss >> trash >> trash;
			
			float x[3];

			for (int i = 0; i < 3; ++i)
				iss >> x[i];

			vec3 n(x[0], x[1], x[2]);

			normals.push_back(n);
		}
		else if (!line.compare(0, 2, "f ")) {
			std::vector<int> f;
			std::vector<int> vt;
			vector<int> vn;
			int idx;
			int itex;
			int inormal;
			iss >> trash;
			while (iss >> idx >> trash >> itex >> trash >> inormal) {
				--idx; // in wavefront obj all indices start at 1, not zero
				--itex;
				--inormal;
				//std::cout << itex << " \n";
				f.push_back(idx);
				vt.push_back(itex);
				vn.push_back(inormal);
			}
			faces_.push_back(f);
			vt_indices.push_back(vt);
			vn_indices.push_back(vn);
		}
	}
	std::cerr << "# v# " << verts.size() << " f# " << faces_.size() << std::endl;
}

Model::~Model() {
}

int Model::nverts() {
	return (int)verts.size();
}

int Model::nfaces() {
	return (int)faces_.size();
}

std::vector<int> Model::face(int idx) {
	return faces_[idx];
}

std::vector<int> Model::vt_index(int idx)
{
	return vt_indices[idx];
}

vec3 Model::vert(int i) {
	return verts[i];
}

vec2 Model::tex(int i)
{
	return texs[i];
}

vec3 Model::normal_value(int face_index, int vertex_index)
{
	int idx = vn_indices[face_index][vertex_index];

	return normals[idx];
}
