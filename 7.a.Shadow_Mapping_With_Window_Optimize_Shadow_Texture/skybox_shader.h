#ifndef _SKY_BOX_H_
#define _SKY_BOX_H_

#include "shader.h"
#include "sample.h"

struct SkyBox : Shader
{
	void vertex(int& face_ind)
	{
		
		for (int vertex_ind = 0; vertex_ind < 3; ++vertex_ind)
		{
			int face = mesh->faces[9 * face_ind + 3 * vertex_ind];
			int tex_ind = mesh->faces[9 * face_ind + 3 * vertex_ind + 1];
			int normal_ind = mesh->faces[9 * face_ind + 3 * vertex_ind + 2];

			vec3 v = mesh->vertices[face];

			vec4 homogeneous = mvp * vec4(v, 1.0f);

			homogeneous = view_port * homogeneous;

			homogeneous.z = homogeneous.w;

			//cout << homogeneous.x << " " << homogeneous.y << " " << homogeneous.z << "\n";

			//chia se lam bno nho lai
			float inv_w = 1.0f / homogeneous.w;

			//homogeneous *= inv_w;


			screen_coord_w[vertex_ind] = homogeneous *inv_w;//vec3(homogeneous.x, homogeneous.y, homogeneous.z);
			screen_coord[vertex_ind] = vec3(homogeneous.x, homogeneous.y, homogeneous.z);

			world_pos[vertex_ind] = v;
			clip_coord[vertex_ind] = homogeneous;

			uv[vertex_ind] = mesh->texcoords[tex_ind];
			normal[vertex_ind] = mesh->normals[normal_ind];
		}
		
	}
	vec3 fragment()
	{
		vec3 inv_clip_coord_w(1.0f / screen_coord_w[0].w, 1.0f / screen_coord_w[1].w, 1.0f / screen_coord_w[2].w);

		vec3 bc_clip(barycentric_coord[0] * inv_clip_coord_w.x, barycentric_coord[1] * inv_clip_coord_w.y, barycentric_coord[2] * inv_clip_coord_w.z);

		bc_clip = bc_clip / (bc_clip.x + bc_clip.y + bc_clip.z);

		vec3 position = bc_clip.x * world_pos[0] + bc_clip.y * world_pos[1] + bc_clip.z * world_pos[2];

		//cout << position.x << " " << position.y << " " << position.z << "\n";
		vec3 c = cube_map->ev(position);

		//cout << c.x << " " << c.y << " " << c.z << "\n";

		return c;
	}
};
#endif // !_SKY_BOX_H_

