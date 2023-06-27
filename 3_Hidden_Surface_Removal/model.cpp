#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include "model.h"

Model::Model(const char *filename) : verts_(), faces_() {
	std::ifstream in;
	in.open(filename, std::ifstream::in);
	if (in.fail()) return;
	std::string line;
	while (!in.eof()) {
		std::getline(in, line);
		std::istringstream iss(line.c_str());
		char trash;
		if (!line.compare(0, 2, "vt"))
		{
			iss >> trash;
			iss >> trash;
			Vec3f vt;
			for (int i = 0; i < 3; i++)
				iss >> vt.raw[i];
			//std::cout << vt.x << " " << vt.y << " " << vt.z << " " << "\n";
			//flip y
			vt[1] = 1 - vt[1];
			texs.push_back(vt);
		}
		if (!line.compare(0, 2, "v ")) {
			iss >> trash;
			Vec3f v;
			for (int i = 0; i<3; i++) iss >> v.raw[i];
			verts_.push_back(v);
		}
		
		else if (!line.compare(0, 2, "f ")) {
			std::vector<int> f;
			std::vector<int> vt;
			int itrash, idx;
			int itex;
			iss >> trash;
			while (iss >> idx >> trash >> itex >> trash >> itrash) {
				--idx; // in wavefront obj all indices start at 1, not zero
				--itex;
				//std::cout << itex << " \n";
				f.push_back(idx);
				vt.push_back(itex);
			}
			faces_.push_back(f);
			vt_indices.push_back(vt);
		}
	}
	std::cerr << "# v# " << verts_.size() << " f# " << faces_.size() << std::endl;
}

Model::~Model() {
}

int Model::nverts() {
	return (int)verts_.size();
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

Vec3f Model::vert(int i) {
	return verts_[i];
}

Vec3f Model::tex(int i)
{
	return texs[i];
}
