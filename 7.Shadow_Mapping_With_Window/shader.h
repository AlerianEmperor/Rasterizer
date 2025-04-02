#ifndef _SHADER_BIKINI_H_
#define _SHADER_BIKINI_H_
#include "mesh.h"
#include "Texture.h"
#include "image.h"
#include <iostream>

using namespace std;

struct defered_info
{
	int face_ind = -1;
	vec3 bary_centric;

	defered_info() {}
	defered_info(int face_ind_, vec3 bary_centric_) : face_ind(face_ind_), bary_centric(bary_centric_) {}
};


struct Shader
{
	//-------uniform matrix variables, most of the time, it remain uniform(not change)------
	int w;//screen width
	int h;//screen height
	mat4 view_port;
	mat4 model;
	mat4 view;
	mat4 projection;
	mat4 mvp;
	
	mat4 shadow_vp;
	mat4 shadow_mvp;
	mat4 shadow_view_port;

	//mat4 mvp_view_port;
	mat3 normal_matrix;
	//mat4 screen_to_shadow;
	mat4 shadow_matrix;
	//-------------------------------------

	//------uniform mesh variables----------
	Mesh* mesh = NULL;
	
	Texture* tex_color = NULL;
	Texture* tex_normal = NULL;
	CubeMap* cube_map = NULL;
	Texture* tex_roughness = NULL;
	Texture* tex_metal = NULL;
	//--------------------------------------

	//------varying variables, often change with vertex, however newer opengl dont use varying keyword anymore----
	//vec3 projection_coord[3]; //gl_vertex = mvp * vertex
	//vec3 ndc_coord[3];		  //gl_vertex /= gl_vertex.w;
	//vec3 original_vertex[3];

	vec3 world_pos[3];

	vec4 model_coord[3];
	vec4 camera_view_coord[3];

	vec4 clip_coord[3];
	vec3 ndc_coord[3];
	vec4 screen_coord_w[3];
	vec3 screen_coord[3];	  //view_port * gl_vertex
	vec2 uv[3];
	vec3 normal[3];
	
	float inv_clip_coord_z[3];

	float inv_screen_coord_w[3];

	//no divide by w yet
	float inv_clip_coord_w[3];
	float clip_coord_w[3];
	
	float depth[3];
	//vec4 homogeneous4[3];//to compute corrective depth interpolation

	vec3 barycentric_coord;

	vec3 tangent;
	//vec3 BiTangent;
	//vec3 T;//tangent
	//---------------------------------------------
	
	vec3 view_position;

	vec3 light_direction;

	//---------------------------------------------
	//shadow related
	int sw;//shadow width
	int sh;//shadow_height
	//mat4 shadow_mvp;
	//mat4 shadow_viewport;
	//mat4 scene_mvp_inv;//(model view projection viewport) inverse
	//mat4 shadow_mvpv;	
	float* depth_buffer;

	//Texture* depth_texture;

	//---------------------------------------------
	Shader() {}
	virtual ~Shader() {}
	//face_ind = [0, num_triangle - 1] : index of each face
	//vertex_ind = [0, 2]			   : index of each vertex index for this face

	virtual void vertex(int& face_ind) = 0;//, int& vertex_ind) = 0;
	virtual vec3 fragment() = 0;//(int& face_ind) = 0;
	
	//virtual vec3 fragment(const float& alpha, const float& beta, const float& gammar) = 0;
	//virtual vec3 fragment(defered_info& defered);
	virtual vec3 fragment_shadow(int& x, int& y)
	{
		return vec3(1.0f);
	}
	virtual vec3 fragment_shadow(vec3& world_point)
	{
		return vec3(1.0f);
	}
};





#endif // !_SHADER_H_

