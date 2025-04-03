#ifndef _CAMERA_H_
#define _CAMERA_H_

#include "win32.h"

struct Camera
{
	vec3 eye;
	vec3 center;
	vec3 up;
	vec3 x;
	vec3 y;
	vec3 z;

	float m_near;
	float m_far;

	mat4 m_model;
	mat4 m_model_inv;
	mat4 m_lookat;
	mat4 m_lookat_inv;
	mat4 m_project;
	mat4 m_project_inv;

	int w;
	int h;

	float w_2, h_2;
	float inv_w_2, inv_h_2;

	float aspect;

	
	Camera() {}
	Camera(vec3 eye_, vec3 center_, vec3 up_,  uint32_t w_, uint32_t h_, float near_, float far_) : eye(eye_), center(center_), up(up_), w(w_), h(h_), m_near(near_), m_far(far_)
	{
		w_2 = w * 0.5f;
		h_2 = h * 0.5f;

		inv_w_2 = 2.0f / w;
		inv_h_2 = 2.0f / h;

		aspect = (float)w / (float)h;
		
		//getchar();
		m_lookat = lookAt(eye, center, vec3(0, 1, 0));

		//getchar();
		m_lookat_inv = inverse(m_lookat);
		//getchar();
	}

	
	vec4 view_transform(vec4& model_point)
	{
		//vec4 p(model_point.x, model_point.y, model_point.z, 1.0f);
		//vec4 r = m_lookat * p;

		//vec4 r = m_lookat * model_point;
		//return r;

		return m_lookat * model_point;

		//return vec3(r.x, r.y, r.z);
	}
	vec2 project_transform(vec4& camera_point)
	{
		//vec4 p(camera_point.x, camera_point.y, camera_point.z, 1.0f);
		//vec4 r = m_project * p;

		vec4 r = m_project * camera_point;

		float inv_w = 1.0f / r.w;

		return vec2(r.x * inv_w, r.y * inv_w);
	}
	vec2 viewport_transform(vec2& ndc_point)
	{
		return vec2((ndc_point.x + 1.0f) * w_2, (ndc_point.y + 1.0f) * h_2);
	}
	vec2 rasterize(vec3& world_point, vec4& camera_point)
	{
		vec4 model_point = m_model * vec4(world_point, 1.0f);
		camera_point = m_lookat * model_point;
		vec4 r = m_project * camera_point;
		float inv_w = 1.0f / r.w;
		vec2 ndc_point(r.x * inv_w, r.y * inv_w);

		//camera_point.z = 1.0f / camera_point.z;
		return vec2((ndc_point.x + 1.0f) * w_2, (ndc_point.y + 1.0f) * h_2);
	}
	vec2 rasterize(vec3& world_point, float& camera_point_depth)
	{
		vec4 model_point = m_model * vec4(world_point, 1.0f);

		vec4 camera_point = m_lookat * model_point;
		vec4 r = m_project * camera_point;
		float inv_w = 1.0f / r.w;
		vec2 ndc_point(r.x * inv_w, r.y * inv_w);

		//camera_point.z = 1.0f / camera_point.z;

		camera_point_depth = 1.0f / camera_point.z;

		return vec2((ndc_point.x + 1.0f) * w_2, (ndc_point.y + 1.0f) * h_2);
	}

	vec3 view_transform_inv(vec3& camera_point)
	{
		vec4 p(camera_point.x, camera_point.y, camera_point.z, 1.0f);
		vec4 r = m_lookat_inv * p;

		return vec3(r.x, r.y, r.z);
	}
	virtual vec3 project_transform_inv(vec2& projected_point, float& camera_depth) = 0;
	vec2 viewport_transform_inv(vec2& screen_point)
	{
		return vec2(screen_point.x * inv_w_2 - 1.0f, screen_point.y * inv_h_2 - 1.0f);
	}
	vec3 unrasterize(vec2& screen_point, float& depth_camera)
	{
		//vec2 ndc_point = camera->viewportTransformInv(point_screen);
		//vec3 camera_point = camera->projectTransformInv(ndc_point, depth_camera);
		//vec3 world_point = camera->viewTransformInv(camera_point);

		vec2 ndc_point(screen_point.x * inv_w_2 - 1.0f, screen_point.y * inv_h_2 - 1.0f);
		vec3 camera_point = project_transform_inv(ndc_point, depth_camera);
		
		vec4 p(camera_point.x, camera_point.y, camera_point.z, 1.0f);
		vec4 r = m_lookat_inv * p;

		r = m_model_inv * r;

		return vec3(r.x, r.y, r.z);
	}
	

};

struct Perspective : public Camera
{
public:
	float m_fov, m_zoom;

	Perspective() : Camera() {}
	Perspective(vec3& position, vec3& target, vec3 up, uint32_t w_, uint32_t h_, const float near_ = 2, const float far_ = 5000, const float fov = 90.0f) : Camera(position, target, up, w_, h_, near_, far_), m_fov(fov)
	{
		m_zoom = 1 / tan((fov / 2) * 3.1415926535897 / 180);

		const float tan_half_fov = tan((m_fov / 2) * 3.1415926535897 / 180);
		const double z_range = m_far - m_near;


		m_project[0][0] = 1.0 / (tan_half_fov * aspect);     m_project[0][1] = 0;                                     m_project[0][2] = 0;                                 m_project[0][3] = 0;
		m_project[1][0] = 0;                                 m_project[1][1] = 1.0 / tan_half_fov;                    m_project[1][2] = 0;                                 m_project[1][3] = 0;
		m_project[2][0] = 0;                                 m_project[2][1] = 0;                                     m_project[2][2] = m_far / z_range;                   m_project[2][3] = (m_far * m_near) / -z_range;
		m_project[3][0] = 0;                                 m_project[3][1] = 0;                                     m_project[3][2] = 1;								   m_project[3][3] = 0;
		
		
		/*m_project[0] = 1.0 / (tan_half_fov * aspect);     m_project[1] = 0;                                     m_project[2] = 0;                                 m_project[3] = 0;
		m_project[4] = 0;                                 m_project[5] = 1.0 / tan_half_fov;                    m_project[6] = 0;                                 m_project[7] = 0;
		m_project[8] = 0;                                 m_project[9] = 0;                                     m_project[10] = m_far / z_range;                   m_project[11] = (m_far * m_near) / -z_range;
		m_project[12] = 0;                                 m_project[13] = 0;                                     m_project[14] = 1;								   m_project[15] = 0;
		*/
		//m_project = transpose(m_project);
		
		m_project_inv = inverse(m_project);//glm::inverse(m_project);
	}

	~Perspective() {}

	//const vec3 projectTransformInv(const vec2& projected_point, const double depth) const override
	vec3 project_transform_inv(vec2& projected_point, float& camera_depth)
	{
		vec4 point(projected_point.x * camera_depth, projected_point.y * camera_depth, camera_depth, 1.0f);

		vec4 r = m_project_inv * point;
		return vec3(r.x, r.y, r.z * camera_depth);
	}
};

class Orthographic : public Camera
{
public:
	Orthographic::Orthographic() : Camera() {}
	//Camera(vec3 eye_, vec3 center_, vec3 up_, int w_, int h_, float near = 2, float far = 5000) : eye(eye_), center(center_), up(up_), w(w_), h(h_), m_near(near), m_far(far)


	Orthographic::Orthographic(vec3& position, vec3& target, vec3 up, uint32_t w_, uint32_t h_, const float near_, const float far_)
		: Camera(position, target, up, w_, h_, near_, far_)
	{
		const double z_range = m_far - m_near;
		
		m_project[0][0] = 64.0f / w_;                m_project[0][1] = 0;                m_project[0][2] = 0;                      m_project[0][3] = 0;
		m_project[1][0] = 0;                          m_project[1][1] = 64.0f / h_;      m_project[1][2] = 0;                      m_project[1][3] = 0;
		m_project[2][0] = 0;                          m_project[2][1] = 0;                m_project[2][2] = 1.0 / z_range;          m_project[2][3] = m_near / z_range;
		m_project[3][0] = 0;                          m_project[3][1] = 0;                m_project[3][2] = 0;						m_project[3][3] = 1;
		

		/*m_project[0][0] = 1.0f;                m_project[0][1] = 0;                m_project[0][2] = 0;                      m_project[0][3] = 0;
		m_project[1][0] = 0;                          m_project[1][1] = 1.0f;      m_project[1][2] = 0;                      m_project[1][3] = 0;
		m_project[2][0] = 0;                          m_project[2][1] = 0;                m_project[2][2] = 1.0 / z_range;          m_project[2][3] = m_near / z_range;
		m_project[3][0] = 0;                          m_project[3][1] = 0;                m_project[3][2] = 0;						m_project[3][3] = 1;
		*/

		/*m_project[0] = 128.0f / w_;                m_project[1] = 0;                m_project[2] = 0;                      m_project[3] = 0;
		m_project[4] = 0;                          m_project[5] = 128.0f / h_;      m_project[6] = 0;                      m_project[7] = 0;
		m_project[8] = 0;                          m_project[9] = 0;                m_project[10] = 1.0 / z_range;         m_project[11] = m_near / z_range;
		m_project[12] = 0;                         m_project[13] = 0;               m_project[14] = 0;					m_project[15] = 1;
		*/

		m_project_inv = inverse(m_project);
	}

	Orthographic::~Orthographic() {

	}

	// NDC [-1,1] to Camera space
	vec3 Orthographic::project_transform_inv(vec2& projected_point, float& camera_depth) 
	{
		vec4 point(projected_point.x, projected_point.y, camera_depth, 1);
		
		vec4 r = m_project_inv * point;
		return vec3(r.x, r.y, camera_depth);
	}

	
};


void update_camera(Camera*& cam)
{
	vec3 view = cam->eye - cam->center;

	float radius = view.length();

	float phi = (float)atan2(view.x, view.z);
	float theta = (float)acosf(view.y / radius);

	float x_delta = window->mouse.orbit_delta.x / window->w;
	float y_delta = window->mouse.orbit_delta.y / window->h;

	radius *= (float)pow(0.95, window->mouse.wheel_delta);

	float factor = 1.5f * pi;

	phi += x_delta * factor;
	theta -= y_delta * factor;

	if (theta > pi)
		theta = pi - eps * 100;
	if (theta < 0)
		theta = eps * 100;

	cam->eye.x = cam->center.x + radius * sin(theta) * sin(phi);
	cam->eye.y = cam->center.y + radius * cos(theta);
	cam->eye.z = cam->center.z + radius * sin(theta) * cos(phi);

	factor = radius * (float)tan(pi * 0.1666f) * 2.2f;
	x_delta = window->mouse.fv_delta.x / window->w;
	y_delta = window->mouse.fv_delta.y / window->h;

	vec3 left = x_delta * factor * cam->x;
	vec3 up = y_delta * factor * cam->y;

	cam->eye += left - up;
	cam->center += left - up;
}

void handle_mouse(Camera*& cam)
{
	if (window->button[0])
	{
		vec2 cur_pos = get_mouse_pos();
		window->mouse.orbit_delta = -window->mouse.orbit_pos + cur_pos;
		window->mouse.orbit_pos = cur_pos;
	}
	if (window->button[1])
	{
		vec2 fv_pos = get_mouse_pos();
		window->mouse.fv_delta = -window->mouse.fv_pos + fv_pos;
		window->mouse.fv_pos = fv_pos;
	}
	update_camera(cam);
}

void handle_key(Camera*& cam)
{
	float distance = (cam->eye - cam->center).length();

	if (window->keys['W'])
		cam->eye -= 10.0f / window->w * cam->z * distance;
	if (window->keys['S'])
		cam->eye += 0.05f * cam->z;
	if (window->keys[VK_UP] | window->keys['Q'])
	{
		cam->eye += 0.05f * cam->y;
		cam->center += 0.05f * cam->y;
	}
	if (window->keys[VK_DOWN] | window->keys['E'])
	{
		cam->eye -= 0.05f * cam->y;
		cam->center -= 0.05f * cam->y;
	}
	if (window->keys[VK_LEFT] | window->keys['A'])
	{
		cam->eye -= 0.05f * cam->x;
		cam->center -= 0.05f * cam->x;
	}
	if (window->keys[VK_RIGHT] | window->keys['D'])
	{
		cam->eye += 0.05f * cam->x;
		cam->center += 0.05f * cam->x;
	}
	if (window->keys[VK_ESCAPE])
	{
		window->is_close = 1;
	}
}

void handle_event(Camera*& cam)
{
	cam->z = (cam->eye - cam->center).norm();
	cam->x = cross(cam->up, cam->z).norm();
	cam->y = cross(cam->z, cam->x).norm();

	handle_mouse(cam);
	handle_key(cam);
}

#endif // !_CAMERA_H_

