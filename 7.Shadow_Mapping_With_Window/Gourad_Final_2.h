#ifndef _GOURAD_Final_2_H_
#define _GOURAD_Final_2_H_
#include "shader.h"

const float directional_light_distance = 200;

vec3 light_direction_position(vec3& direction)
{
	return direction * directional_light_distance;
}

struct Gourad2 : Shader
{
	vec3 T;
	vec3 N;
	vec3 B;
	mat3 TBN;

	vec3 l;//TBN light direction;

	float calculate_depth(vec3& barycentric_coord, vec4 camera_view_coord[3])
	{
		return
			(1.0f / camera_view_coord[0].z * barycentric_coord.x +
				1.0f / camera_view_coord[1].z * barycentric_coord.y +
				1.0f / camera_view_coord[2].z * barycentric_coord.z);
	}

	vec2 screenspace_to_ndc(vec3& screen_point)
	{
		float x = screen_point.x * 2.0f / (float)w - 1.0f;
		float y = screen_point.y * 2.0f / (float)h - 1.0f;

		return vec2(x, y);
	}

	/*vec3 ndc_to_viewspace(vec2& ndc, float& depth_from_camera)
	{
		vec4 point(ndc.x * depth_from_camera, ndc.y * depth_from_camera, depth_from_camera, 1.0f);

		point = invert_projection * point;

		return vec3(point.x, point.y, point.z * depth_from_camera);
	}

	vec3 viewspace_to_world(vec3& view_space)
	{
		vec4 p = invert_view * vec4(view_space, 1.0f);

		return vec3(p.x, p.y, p.z);
	}

	vec3 unrasterize(vec3& screen_point, float& depth)
	{
		vec2 ndc = screenspace_to_ndc(screen_point);
		vec3 view_space = ndc_to_viewspace(ndc, depth);
		vec3 world_space = viewspace_to_world(view_space);

		return world_space;
	}*/

	void vertex(int& face_ind)//, int& vertex_ind)
	{
		//for (int vertex_ind = 3; vertex_ind--;)
		for (int vertex_ind = 0; vertex_ind < 3; ++vertex_ind)
		{
			//vec3i face = mesh->vertex_indices[face_ind];
			//vec3i tex_ind = mesh->texcoord_indices[face_ind];
			//vec3i normal_ind = mesh->normal_indices[face_ind];

			int face = mesh->faces[9 * face_ind + 3 * vertex_ind];
			int tex_ind = mesh->faces[9 * face_ind + 3 * vertex_ind + 1];
			int normal_ind = mesh->faces[9 * face_ind + 3 * vertex_ind + 2];

			vec3 v = mesh->vertices[face];

			//original_vertex[vertex_ind] = v;

			//vec4 homogeneous = projection * view * model * vec4(v, 1.0f);

			model_coord[vertex_ind] = model * vec4(v, 1.0f);

			camera_view_coord[vertex_ind] = view * model_coord[vertex_ind];

			vec4 homogeneous = projection * camera_view_coord[vertex_ind];


			//vec4 homogeneous = mvp * vec4(v, 1.0f);

			//vec4 homogeneous = mvp * v;

			//clip_coord_w[vertex_ind] = homogeneous.w;

			depth[vertex_ind] = homogeneous.z;




			//vec4 homogeneous = mvp_view_port * vec4(v, 1.0f);

			//screen_coord_w[vertex_ind] = homogeneous;//vec3(int(homogeneous.x), int(homogeneous.y), homogeneous.z);

			clip_coord[vertex_ind] = homogeneous;

			float inv_w = 1.0f / homogeneous.w;

			inv_clip_coord_w[vertex_ind] = inv_w;

			inv_clip_coord_z[vertex_ind] = 1.0f / homogeneous.z;

			homogeneous *= inv_w;

			ndc_coord[vertex_ind] = vec3(homogeneous.x, homogeneous.y, homogeneous.z);//proj3(homogeneous);

			homogeneous = view_port * homogeneous;

			world_pos[vertex_ind] = v;
			screen_coord[vertex_ind] = vec3(homogeneous.x, homogeneous.y, homogeneous.z);
			uv[vertex_ind] = mesh->texcoords[tex_ind];
			normal[vertex_ind] = mesh->normals[normal_ind];
		}

		//vec3i normal_ind = mesh->normal_indices[face_ind];
		//vec3 vertex_normal = mesh->normals[normal_ind[0]];

		T = mesh->tangents[face_ind];

		T = (normal_matrix * T);// .norm();

		T.normalize();

		N = (normal_matrix * (normal[0] + normal[1] + normal[2])).norm();

		//N = (normal_matrix * vertex_normal).norm();

		T = (T - dot(T, N) * N).norm();

		B = cross(N, T);

		TBN = mat3(T, B, N);

		TBN = TBN.transpose();

		l = light_direction;

		//l = (TBN * light_direction).norm();
	}
	

	vec3 fragment()
	{
		float depth = calculate_depth(barycentric_coord, camera_view_coord);

		//vec3 screen_point = barycentric_coord[0] * screen_coord[0] + barycentric_coord[1] * screen_coord[1] + barycentric_coord[2] * screen_coord[2];

		/*vec4 ndc(2.0f * screen_point.x / w - 1.0f, 2.0f * screen_point.y / h - 1.0f, 2.0f * screen_point.z - 1.0f, 1.0f);

		vec4 world_point = invert_projection_view * ndc;

		world_point.w /= world_point.w;

		world_point.x /= world_point.w;
		world_point.y /= world_point.w;
		world_point.z /= world_point.w;
		*/

		//vec3 world_point = unrasterize(vec2(screen_point.x, screen_point.y), depth);

		//vec3 light_position = light_direction_position(l);

		//float current_depth = 20.0f;//(light_position - world_point).length();

		//vec4 shadow_projection = shadow_mvp * vec4(world_point, 1.0f);

		//float current_depth = (shadow_projection.z + 1.0f) * 0.5f;

		//vec3 world_point = unrasterize(screen_point, depth);

		//vec4 shadow_projection = shadow_mvp * vec4(world_point, 1.0f);

		//float current_depth = (shadow_projection.z + 1.0f) * 0.5f;

		//float inv_w = 1.0f / shadow_projection.w;

		//shadow_projection *= inv_w;

		//shadow_projection = shadow_view_port * shadow_projection;
		//cout << "x\n";

		//shadow_projection.x = (shadow_projection.x + 1.0f) * 0.5f * w;
		//shadow_projection.y = (shadow_projection.y + 1.0f) * 0.5f * h;


		//int shadow_buffer_ind = shadow_projection.y * w + shadow_projection.x;

		/*if (shadow_buffer_ind >= w * h - 1)
		shadow_buffer_ind = w * h - 1;
		if (shadow_buffer_ind < 0)
		shadow_buffer_ind = 0;
		*/

		vec3 normal_coord = (normal[0] * barycentric_coord[0] + normal[1] * barycentric_coord[1] + normal[2] * barycentric_coord[2]);// .norm();

		float real_intensity = (dot(normal_coord, l));

		vec3 c(255 * 0.6);

		vec3 color;

		for (int i = 0; i < 3; ++i)
			color[i] = (c[i] * max(0.01f, (0.04f + real_intensity)));

		//if (shadow_buffer_ind > w * h - 1 || shadow_buffer_ind < 0)
		//	return vec3(255, 0, 0);

		float bias = max(0.05 * (1.0f - real_intensity), 0.005f);


		//float closest_depth = depth_buffer[shadow_buffer_ind];

		//cout << closest_depth << " " << current_depth << "\n";

		vec3 light_position = light_direction_position(light_direction);

		float current_depth = 1;// (light_position - world_point).length();

		//cout << closest_depth << "\n";

		/*if (real_intensity > 0.0f && closest_depth <= current_depth - bias)
		{
			//cout << "itz shadow\n";
			return 0.1f * color;
		}*/
		return color;
	}

	vec3 fragment_x()
	{
		float equalizer = 1.0f / (barycentric_coord[0] * inv_clip_coord_w[0] +
			barycentric_coord[1] * inv_clip_coord_w[1] +
			barycentric_coord[2] * inv_clip_coord_w[2]);

		vec2 texcoord = (barycentric_coord[0] * uv[0] * inv_clip_coord_w[0] +
			barycentric_coord[1] * uv[1] * inv_clip_coord_w[1] +
			barycentric_coord[2] * uv[2] * inv_clip_coord_w[2]) * equalizer;

		vec4 real_clip_coord = (barycentric_coord[0] * inv_clip_coord_w[0] * clip_coord[0] +
			barycentric_coord[1] * inv_clip_coord_w[1] * clip_coord[1] +
			barycentric_coord[2] * inv_clip_coord_w[2] * clip_coord[2]) * equalizer;


		/*vec3 interpolate_screen_coord = barycentric_coord[0] * screen_coord[0] + barycentric_coord[1] * screen_coord[1] + barycentric_coord[2] * screen_coord[2];

		vec4 interpolate_ndc_coord = (interpolate_screen_coord.x / w * 2.0f - 1.0f, interpolate_screen_coord.y / h * 2.0f - 1.0f,
		interpolate_screen_coord.z * 2.0f - 1.0f, 1.0f);

		float equalizer = 1.0f / (barycentric_coord[0] * inv_clip_coord_w[0] +
		barycentric_coord[1] * inv_clip_coord_w[1] +
		barycentric_coord[2] * inv_clip_coord_w[2]);

		float real_w = (barycentric_coord[0] * clip_coord[0].w * inv_clip_coord_w[0] +
		barycentric_coord[1] * clip_coord[1].w * inv_clip_coord_w[1] +
		barycentric_coord[2] * clip_coord[2].w * inv_clip_coord_w[2]) * equalizer;


		real_w = equalizer;

		vec4 real_clip_coord = interpolate_ndc_coord * real_w;
		*/

		//ko ton tai mvp inver nua
		//vec4 shadow_projection = shadow_mvp * mvp_invert * real_clip_coord;

		vec4 shadow_projection = shadow_mvp * real_clip_coord;

		float current_depth = (shadow_projection.z + 1) * 0.5f;


		float inv_w = 1.0f / shadow_projection.w;

		shadow_projection *= inv_w;

		//shadow_projection = shadow_view_port * shadow_projection;

		shadow_projection.x = (shadow_projection.x + 1) * 0.5f * w;
		shadow_projection.y = (shadow_projection.y + 1) * 0.5f * h;

		int shadow_buffer_ind = shadow_projection.y * w + shadow_projection.x;

		if (shadow_buffer_ind >= w * h - 1)
			shadow_buffer_ind = w * h - 1;
		if (shadow_buffer_ind < 0)
			shadow_buffer_ind = 0;

		vec3 normal_coord = (normal[0] * barycentric_coord[0] + normal[1] * barycentric_coord[1] + normal[2] * barycentric_coord[2]).norm();

		float real_intensity = abs(dot(normal_coord, l));

		vec3 c = vec3(255 * 0.6);

		vec3 color;

		for (int i = 0; i < 3; ++i)
			color[i] = (c[i] * max(0.01f, (0.4f + real_intensity)));

		float bias = max(0.5 * (1.0f - real_intensity), 0.05f);

		float closest_depth = depth_buffer[shadow_buffer_ind];

		cout << closest_depth << " " << current_depth << "\n";

		if (closest_depth >= current_depth - bias)
		{
			cout << "shadow\n";

			return 0.1f * color;
		}
		return color;
	}

};

#endif // !_GOURAD_H_

