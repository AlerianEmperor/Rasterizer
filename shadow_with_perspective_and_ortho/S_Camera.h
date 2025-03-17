#ifndef _CAMERA_H_
#define _CAMERA_H_

#include <math.h>       /* sin */
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>

float minf(const float& a, const float& b) { return a < b ? a : b; }
float maxf(const float& a, const float& b) { return a > b ? a : b; }


class Camera {
protected:
	glm::vec3
		m_direction,
		m_right,
		m_up;

	glm::mat4
		m_lookat,
		m_lookat_inv,
		m_project,
		m_project_inv;

	float m_near,
		m_far;

	glm::vec3 m_position;
	uint32_t m_image_height,
		m_image_width;

public:
	Camera();
	Camera(const glm::vec3& position, const glm::vec3& target, const uint32_t image_width, const uint32_t image_height, const float near = 2, const float far = 5000);
	~Camera();

	inline const float get_near_plane() const { return m_near; }
	inline const float get_far_plane() const { return m_far; }

	inline const uint32_t get_width() const { return m_image_width; }
	inline const uint32_t get_height() const { return m_image_height; }

	const bool insideFrustrum(const glm::vec2& raster_point, const float depth) const;

	// Transformations
	const glm::vec3 viewTransform(const glm::vec3& point_world) const;
	const glm::vec3 viewTransformInv(const glm::vec3& camera_point) const;

	const glm::vec2 viewportTransform(const glm::vec2& ndc_point) const;
	const glm::vec2 viewportTransformInv(const glm::vec2& raster_point) const;

	const glm::vec2 projectTransform(const glm::vec3& camera_point) const;
	virtual const glm::vec3 projectTransformInv(const glm::vec2& projected_point, const double depth) const = 0;

	virtual const glm::vec3 viewDirection(const glm::vec3& point) const = 0;
protected:
	inline const double get_aspect() const { return (float)m_image_width / m_image_height; }
};

Camera::Camera() {

}

Camera::Camera(const glm::vec3& position, const glm::vec3& target, const uint32_t image_height, const uint32_t image_width, const float near, const float far)
	: m_position(position), m_image_height(image_height), m_image_width(image_width) 
{
	m_near = near;
	m_far = far;

	/*glm::vec3 f = (target - position);// .norm();
	f = normalize(f);

	glm::vec3 l = cross(glm::vec3(0, 1, 0), f);
	l = normalize(l);

	glm::vec3 u = cross(f, l);// .norm();
	u = normalize(u);

	glm::mat4 m;

	m[0] = glm::vec4(l.x, l.y, l.z, -dot(l, position));
	m[1] = glm::vec4(u.x, u.y, u.z, -dot(u, position));
	m[2] = glm::vec4(f.x, f.y, f.z, -dot(f, position));
	m[3] = glm::vec4(0, 0, 0, 1.0f);

	m_lookat = m;//glm::transpose(m);*/

	m_direction = (target - position);
	m_direction = normalize(m_direction);

	m_right = -cross(m_direction, glm::vec3(0, 1, 0));
	m_right = normalize(m_right);

	m_up = cross(m_direction, m_right);
	m_up = normalize(m_up);

	glm::mat4 orientation;
	orientation[0][0] = m_right.x;       orientation[0][1] = m_right.y;       orientation[0][2] = m_right.z;       orientation[0][3] = 0;
	orientation[1][0] = m_up.x;          orientation[1][1] = m_up.y;          orientation[1][2] = m_up.z;          orientation[1][3] = 0;
	orientation[2][0] = m_direction.x;   orientation[2][1] = m_direction.y;   orientation[2][2] = m_direction.z;   orientation[2][3] = 0;
	orientation[3][0] = 0;               orientation[3][1] = 0;               orientation[3][2] = 0;               orientation[3][3] = 1;


	glm::mat4 translation;
	translation[0][0] = 1;               translation[0][1] = 0;               translation[0][2] = 0;               translation[0][3] = -m_position.x;
	translation[1][0] = 0;               translation[1][1] = 1;               translation[1][2] = 0;               translation[1][3] = -m_position.y;
	translation[2][0] = 0;               translation[2][1] = 0;               translation[2][2] = 1;               translation[2][3] = -m_position.z;
	translation[3][0] = 0;               translation[3][1] = 0;               translation[3][2] = 0;               translation[3][3] = 1;

	m_lookat = translation * orientation;

	m_lookat_inv = inverse(m_lookat);
}

Camera::~Camera() {

}

glm::mat4 multiply(const glm::mat4& m1, const glm::mat4& m2)
{
	glm::mat4 result;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
		{
			float sum = 0;
			for (int k = 0; k < 4; ++k)
				sum += m1[i][k] * m2[k][j];
			result[i][j] = sum;
		}
	return result;
}

glm::vec4 multiply(const glm::mat4& m, const glm::vec4& v)
{
	glm::vec4 result;
	for (int i = 0; i < 4; ++i)
	{
		float sum = 0;
		for (int j = 0; j < 4; ++j)
			sum += m[i][j] * v[j];
		result[i] = sum;
	}
	return result;
}

glm::mat4 transpose(glm::mat4& m)
{
	glm::mat4 trans;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			//trans.e[4 * i + j] = e[4 * j + i];
			trans[i][j] = m[j][i];
	return trans;
}

//3D math primer for game development, 2nd, page 184 /845
float determine(glm::mat3& m)
{
	return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) +
		m[0][1] * (m[1][2] * m[2][0] - m[1][0] * m[2][2]) +
		m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

//minor cofactor 3D math primer for graphics and game development, page 185 / 845 on search tab 
float m_minor(glm::mat4 mat, int r, int c)
{
	//cut down matrix
	glm::mat3 result;

	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
		{
			int new_r = i < r ? i : i + 1;
			int new_c = j < c ? j : j + 1;

			result[i][j] = mat[new_r][new_c];
		}
	return determine(result);
}
float m_cofactor(glm::mat4 mat, int r, int c)
{
	int sign = (r + c) & 1 ? -1 : 1;

	float minor_det = m_minor(mat, r, c);

	return sign * minor_det;
}

glm::mat4 adjoint(glm::mat4 mat)
{
	glm::mat4 result;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			result[i][j] = m_cofactor(mat, i, j);
	return result;
}

//page 186 / 845
glm::mat4 inverse_transpose(glm::mat4& m)
{
	//M = sum(m[i][j] * C[i][j]) = sum(m[i][j] * (-1)^(i + j) * det[i][j])

	glm::mat4 adj = adjoint(m);

	float det = 0;
	for (int j = 0; j < 4; ++j)
		det += m[0][j] * adj[0][j];

	float inv_det = 1.0f / det;

	glm::mat4 result;
	//for (int i = 0; i < 16; ++i)
	//	result.e[i] = adj.e[i] * inv_det;

	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			result[i][j] = adj[i][j] * inv_det;

	return result;
}
glm::mat4 inverse(glm::mat4& m)
{
	return transpose(inverse_transpose(m));
}

float dot_vp(const glm::vec3& v, const glm::vec3& p)
{
	return v.x * p.x + v.y * p.y + v.z * p.z;
}

// Camera space to NDC [-1,1]
const glm::vec2 Camera::projectTransform(const glm::vec3& camera_point) const {
	// Prepare matrix for clipping
	const glm::vec4 point = glm::vec4(camera_point.x, camera_point.y, camera_point.z, 1);
	const glm::vec4 r = point * m_project;
	//const glm::vec4 r = m_project * point;

	//const glm::vec4 r = multiply(point, m_project);

	//const glm::vec4 r = multiply(m_project, point);

	float inv_w = 1.0f / r.w;
	// Perspective divide
	return glm::vec3(r.x * inv_w, r.y * inv_w, r.z * inv_w * 0.5f + 0.5f);
}

// World space to camera/view space
const glm::vec3 Camera::viewTransform(const glm::vec3& point_world) const {
	glm::vec4 p = glm::vec4(point_world.x, point_world.y, point_world.z, 1);
	glm::vec4 r = p * m_lookat;

	//glm::vec4 r = multiply(p, m_lookat);

	//glm::vec4 r = m_lookat * p;

	//glm::vec4 r = multiply(m_lookat, p);

	return glm::vec3(r.x, r.y, r.z);
}

// Camera/view space to World space
const glm::vec3 Camera::viewTransformInv(const glm::vec3& camera_point) const 
{
	glm::vec4 p = glm::vec4(camera_point.x, camera_point.y, camera_point.z, 1);
	
	//glm::vec4 r = multiply(m_lookat_inv, p);
	glm::vec4 r = p * m_lookat_inv;

	return glm::vec3(r.x, r.y, r.z);
}

// NDC [-1,1] to raster/screen space
const glm::vec2 Camera::viewportTransform(const glm::vec2& ndc_point) const {
	const double slopeX = m_image_width / 2;
	const double slopeY = m_image_height / 2;

	const glm::vec2 raster_point = {
		slopeX * (ndc_point.x + 1),
		slopeY * (ndc_point.y + 1)
	};

	return raster_point;
}

// Raster/screen space to NDC [-1,1]
const glm::vec2 Camera::viewportTransformInv(const glm::vec2& raster_point) const {
	const double slopeX = 2.0 / m_image_width;
	const double slopeY = 2.0 / m_image_height;

	const glm::vec2 ndc_point = {
		-1 + slopeX * raster_point.x,
		-1 + slopeY * raster_point.y
	};

	return ndc_point;
}

const bool Camera::insideFrustrum(const glm::vec2& raster_point, const float depth) const {
	return (raster_point.x < m_image_width && raster_point.x >= 0 &&
		raster_point.y < m_image_height && raster_point.y >= 0 &&
		depth >= m_near && depth <= m_far);
}


class Perspective : public Camera
{
public:
	float m_fov, m_zoom;

	Perspective() : Camera() {}
	Perspective(const glm::vec3& position, const glm::vec3& target, const uint32_t image_height, const uint32_t image_width, const float near = 2, const float far = 5000, const float fov = 90.0f) : Camera(position, target, image_height, image_width, near, far), m_fov(fov)
	{
		m_zoom = 1 / tan((fov / 2) * 3.1415926535897 / 180);


		const float tan_half_fov = tan((m_fov / 2) * 3.1415926535897 / 180);
		const double z_range = m_far - m_near;

		m_project[0][0] = 1.0 / (tan_half_fov * get_aspect());     m_project[0][1] = 0;                                     m_project[0][2] = 0;                                     m_project[0][3] = 0;
		m_project[1][0] = 0;                                     m_project[1][1] = 1.0 / tan_half_fov;                      m_project[1][2] = 0;                                     m_project[1][3] = 0;
		m_project[2][0] = 0;                                     m_project[2][1] = 0;                                     m_project[2][2] = m_far / z_range;                        m_project[2][3] = 1;
		m_project[3][0] = 0;                                     m_project[3][1] = 0;                                     m_project[3][2] = (m_far * m_near) / -z_range;       m_project[3][3] = 0;

		m_project = glm::transpose(m_project);


		//m_project_inv = glm::inverse(m_project);
		m_project_inv = inverse(m_project);//glm::inverse(m_project);
	}

	~Perspective() {}

	const glm::vec3 projectTransformInv(const glm::vec2& projected_point, const double depth) const override
	{
		glm::vec4 point = glm::vec4(projected_point.x * depth, projected_point.y * depth, depth, 1);

		glm::vec4 r = multiply(m_project_inv, point);
		return glm::vec3(r.x, r.y, r.z * depth);
	}

	const glm::vec3 viewDirection(const glm::vec3& point) const override
	{
		glm::vec3 view = point - m_position;
		//viewnormalize();
		return normalize(view);
	}
};

class Orthographic : public Camera
{
public:
	Orthographic::Orthographic() : Camera() {}

	Orthographic::Orthographic(const glm::vec3& position, const glm::vec3& target, const uint32_t image_height, const uint32_t image_width, const float near, const float far)
		: Camera(position, target, image_height, image_width, near, far) 
	{

		const double z_range = m_far - m_near;

		//float a = ;

		m_project[0][0] = 128.0f / image_width ;                m_project[0][1] = 0;                       m_project[0][2] = 0;                      m_project[0][3] = 0;
		m_project[1][0] = 0;                                     m_project[1][1] = 128.0f / image_height ;      m_project[1][2] = 0;                      m_project[1][3] = 0;
		m_project[2][0] = 0;                                     m_project[2][1] = 0;                       m_project[2][2] = 1.0 / z_range;          m_project[2][3] = 0;
		m_project[3][0] = 0;                                     m_project[3][1] = 0;                       m_project[3][2] = m_near / z_range;       m_project[3][3] = 1;
		

		/*m_project[0][0] = 1.0;                     m_project[0][1] = 0;        m_project[0][2] = 0;											m_project[0][3] = 0;
		m_project[1][0] = 0;                       m_project[1][1] = 1.0;      m_project[1][2] = 0;											m_project[1][3] = 0;
		m_project[2][0] = 0;                       m_project[2][1] = 0;        m_project[2][2] = 1.0;										m_project[2][3] = 0;
		m_project[3][0] = 0;                       m_project[3][1] = 0;        m_project[3][2] = -1.0f / (position - target).length();      m_project[3][3] = 1.0;
		*/

		m_project = glm::transpose(m_project);
		m_project_inv = glm::inverse(m_project);
	}

	Orthographic::~Orthographic() {

	}

	// NDC [-1,1] to Camera space
	const glm::vec3 Orthographic::projectTransformInv(const glm::vec2& projected_point, const double depth) const 
	{
		const glm::vec4 point = glm::vec4(projected_point.x, projected_point.y, depth, 1);
		const glm::vec4 r = point * m_project_inv;
		return glm::vec3(r.x, r.y, depth);
	}

	const glm::vec3 Orthographic::viewDirection(const glm::vec3& point) const {
		return m_direction;
	}
};

class Orthographic2 : public Camera
{
public:
	Orthographic2::Orthographic2() : Camera() {}

	Orthographic2::Orthographic2(const glm::vec3& position, const glm::vec3& target, const uint32_t image_height, const uint32_t image_width, float l, float r, float t, float b, const float n, const float f)
		: Camera(position, target, image_height, image_width, n, f)
	{

		/*const double z_range = m_far - m_near;

		m_project[0][0] = 2.0 / image_width;                     m_project[0][1] = 0;                       m_project[0][2] = 0;                      m_project[0][3] = 0;
		m_project[1][0] = 0;                                     m_project[1][1] = 2.0 / image_height;      m_project[1][2] = 0;                      m_project[1][3] = 0;
		m_project[2][0] = 0;                                     m_project[2][1] = 0;                       m_project[2][2] = 1.0 / z_range;          m_project[2][3] = 0;
		m_project[3][0] = 0;                                     m_project[3][1] = 0;                       m_project[3][2] = m_near / z_range;       m_project[3][3] = 1;
		*/

		m_project[0][0] = 2.0f / (r - l);  m_project[0][3] = -(r + l) / (r - l);
		m_project[1][1] = 2.0f / (t - b);  m_project[1][3] = -(t + b) / (t - b);
		m_project[2][2] = -2.0f / (f - n); m_project[2][3] = -(f + n) / (f - n);
		m_project[3][3] = 1.0f;

		m_project = glm::transpose(m_project);
		m_project_inv = glm::inverse(m_project);
	}

	Orthographic2::~Orthographic2() {

	}

	// NDC [-1,1] to Camera space
	const glm::vec3 Orthographic2::projectTransformInv(const glm::vec2& projected_point, const double depth) const
	{
		const glm::vec4 point = glm::vec4(projected_point.x, projected_point.y, depth, 1);
		const glm::vec4 r = point * m_project_inv;
		return glm::vec3(r.x, r.y, depth);
	}

	const glm::vec3 Orthographic2::viewDirection(const glm::vec3& point) const {
		return m_direction;
	}
};
#endif // !_CAMERA_H_