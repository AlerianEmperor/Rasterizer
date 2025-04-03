#ifndef _MESH_H_
#define _MESH_H_
#include "maths.h"
//#include "maths_linear.h"
#include <vector>
#include <fstream>

using namespace std;

struct Mesh
{
	/*int num_faces;
	vector<vec3> vertices;
	vec3 center;*/

	vector<vec3> vertices;
	
	vector<vec2> texcoords;
	vector<vec3> normals;

	vector<vec3> tangents;
	vector<vec3> biTangents;

	//vector<vec3i> vertex_indices;
	//vector<vec3i> texcoord_indices;
	//vector<vec3i> normal_indices;

	vector<int> faces;

	int num_triangles;

	void clear()
	{
		vector<vec3>().swap(vertices);
		vector<vec2>().swap(texcoords);
		vector<vec3>().swap(normals);

		vector<vec3>().swap(tangents);
		vector<vec3>().swap(biTangents);

		//vector<vec3i>().swap(vertex_indices);
		//vector<vec3i>().swap(texcoord_indices);
		//vector<vec3i>().swap(normal_indices);
	}
};


Mesh load_obj(string file_name)
{
	Mesh mesh;

	ifstream ifs(file_name);

	char line[256];
	int num_face = 0;
	while (ifs.getline(line, 256))
	{
		if (strncmp(line, "v ", 2) == 0)
		{
			//vec4 vertex(1);
			vec3 vertex;
			sscanf_s(line, "v %f %f %f", &vertex.x, &vertex.y, &vertex.z);
			mesh.vertices.emplace_back(vertex);
		}
		else if (strncmp(line, "vt ", 3) == 0)
		{
			vec2 texcoord;
			sscanf_s(line, "vt  %f %f", &texcoord.x, &texcoord.y);
			mesh.texcoords.emplace_back(texcoord);
		}
		else if (strncmp(line, "vn ", 3) == 0)
		{
			vec3 normal;
			sscanf_s(line, "vn  %f %f %f", &normal.x, &normal.y, &normal.z);

			normal.normalize();

			mesh.normals.emplace_back(normal);
		}
		else if (strncmp(line, "f ", 2) == 0)
		{
			vec3i v;
			vec3i vt;
			vec3i vn;

	
			sscanf_s(line, "f %d/%d/%d %d/%d/%d %d/%d/%d", &v.x, &vt.x, &vn.x,
														   &v.y, &vt.y, &vn.y,
														   &v.z, &vt.z, &vn.z);

			v.x--; v.y--; v.z--; vt.x--, vt.y--; vt.z--; vn.x--; vn.y--; vn.z--;

			mesh.faces.emplace_back(v.x);
			mesh.faces.emplace_back(vt.x);
			mesh.faces.emplace_back(vn.x);

			mesh.faces.emplace_back(v.y);
			mesh.faces.emplace_back(vt.y);
			mesh.faces.emplace_back(vn.y);

			mesh.faces.emplace_back(v.z);
			mesh.faces.emplace_back(vt.z);
			mesh.faces.emplace_back(vn.z);

			//mesh.vertex_indices.emplace_back(v);
			//mesh.texcoord_indices.emplace_back(vt);
			//mesh.normal_indices.emplace_back(vn);
			++num_face;
		}
	}

	mesh.num_triangles = num_face;//mesh.vertex_indices.size();

	mesh.tangents.resize(num_face, vec3(0));//(mesh.vertex_indices.size(), 0);
	mesh.biTangents.resize(num_face, vec3(0));//(mesh.vertex_indices.size(), 0);

	for (int i = 0; i < num_face; ++i)//mesh.vertex_indices.size(); ++i)
	{
		/*vec3i index = mesh.vertex_indices[i];

		vec3 p0 = mesh.vertices[index.x];
		vec3 p1 = mesh.vertices[index.y];
		vec3 p2 = mesh.vertices[index.z];

		vec3i tex_index = mesh.texcoord_indices[i];

		vec2 uv0 = mesh.texcoords[tex_index.x];
		vec2 uv1 = mesh.texcoords[tex_index.y];
		vec2 uv2 = mesh.texcoords[tex_index.z];
		*/
		
		int index0 = mesh.faces[9 * i];
		int index1 = mesh.faces[9 * i + 3];
		int index2 = mesh.faces[9 * i + 6];


		vec3 p0 = mesh.vertices[index0];
		vec3 p1 = mesh.vertices[index1];
		vec3 p2 = mesh.vertices[index2];

		int tex_index0 = mesh.faces[9 * i + 1];
		int tex_index1 = mesh.faces[9 * i + 4];
		int tex_index2 = mesh.faces[9 * i + 7];


		vec2 uv0 = mesh.texcoords[tex_index0];
		vec2 uv1 = mesh.texcoords[tex_index1];
		vec2 uv2 = mesh.texcoords[tex_index2];


		vec3 dp1 = p1 - p0;
		vec3 dp2 = p2 - p0;
		vec2 dt1 = uv1 - uv0;
		vec2 dt2 = uv2 - uv0;

		vec3 t, bt;

		t.x = dt2.y * dp1.x - dt1.y * dp2.x;
		t.y = dt2.y * dp1.y - dt1.y * dp2.y;
		t.z = dt2.y * dp1.z - dt1.y * dp2.z;

		bt.x = -dt2.x * dp1.x + dt1.x * dp2.x;
		bt.y = -dt2.x * dp1.y + dt1.x * dp2.y;
		bt.z = -dt2.x * dp1.z + dt1.x * dp2.z;
		float f = 1.0f / (dt1.x * dt2.y - dt1.y * dt2.x);

		t = (t * f);// .norm();
		bt = (bt * f);// .norm();

		mesh.tangents[i] += t;
		mesh.biTangents[i] += bt;
		/*mesh.tangents[index.x] += t;
		mesh.tangents[index.y] += t;
		mesh.tangents[index.z] += t;

		mesh.biTangents[index.x] += bt;
		mesh.biTangents[index.y] += bt;
		mesh.biTangents[index.z] += bt;*/
	}

	for (int i = 0; i < mesh.tangents.size(); ++i)
	{
		//vec3 n = mesh.normals[i];
		//vec3 t = mesh.tangents[i].norm();
		
		//mesh.tangents[i] = (t - n * dot(n, t)).norm();
		mesh.tangents[i].normalize();
		mesh.biTangents[i].normalize();
	}

	return mesh;
}



#endif // !_MESH_H_

