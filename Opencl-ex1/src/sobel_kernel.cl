#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

int getIndexGlobal(size_t countX, int i, int j) {
	return j * countX + i;
}
// Read value from global array a, return 0 if outside image
float getValueGlobal(const float *a, size_t countX, size_t countY, int i, int j) {
	if (i < 0 || (size_t)i >= countX || j < 0 || (size_t)j >= countY)
		return 0;
	else
		return a[getIndexGlobal(countX, i, j)];
}

__kernel void kernel1(__global const float* d_input, __global float* d_outputGpu)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);
	float Gx = getValueGlobal(d_input, countX, countY, i - 1, j - 1) + 2 * getValueGlobal(d_input, countX, countY, i - 1, j) + getValueGlobal(d_input, countX, countY, i - 1, j + 1)
				- getValueGlobal(d_input, countX, countY, i + 1, j - 1) - 2 * getValueGlobal(d_input, countX, countY, i + 1, j) - getValueGlobal(d_input, countX, countY, i + 1, j + 1);
	float Gy = getValueGlobal(d_input, countX, countY, i - 1, j - 1) + 2 * getValueGlobal(d_input, countX, countY, i, j - 1) + getValueGlobal(d_input, countX, countY, i + 1, j - 1)
				- getValueGlobal(d_input, countX, countY, i - 1, j + 1) - 2 * getValueGlobal(d_input, countX, countY, i, j + 1) - getValueGlobal(d_input, countX, countY, i + 1, j + 1);
	d_outputGpu[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
}

__kernel void kernel2(__global const float* d_input, __global float* d_outputGpu)
{
	size_t i = get_local_id(0);
	size_t j = get_local_id(1);
	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);

	float g0x__ = getValueGlobal(d_input, countX, countY, i - 1, j - 1);
	float g0_x_ = getValueGlobal(d_input, countX, countY, i - 1, j);
	float g0__x = getValueGlobal(d_input, countX, countY, i - 1, j + 1);

	float g1x__ = getValueGlobal(d_input, countX, countY, i, j - 1);
	float g1__x = getValueGlobal(d_input, countX, countY, i, j + 1);

	float g2x__ = getValueGlobal(d_input, countX, countY, i + 1, j - 1);
	float g2_x_ = getValueGlobal(d_input, countX, countY, i + 1, j);
	float g2__x = getValueGlobal(d_input, countX, countY, i + 1, j + 1);

	float Gx = g0x__ + 2 * g0_x_ + g0__x
		- g2x__ - 2 * g2_x_ - g2__x;
	float Gy = g0x__ + 2 * g1x__ + g2x__
		- g0__x - 2 * g1__x - g2__x;
	d_outputGpu[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
}

// Read value from global array a, return 0 if outside image
const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
float getValueImage(__read_only image2d_t a, int i, int j) {
	//if (i < 0 || i >= countX || j < 0 || j >= countY)
	//return 0;
	return read_imagef(a, sampler, (int2) { i, j }).x;
}
__kernel void kernel3(__read_only image2d_t d_input, __global float* d_output) {
	int i = get_global_id(0);
	int j = get_global_id(1);

	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);

	float Gx = getValueImage(d_input, i - 1, j - 1) + 2 * getValueImage(d_input, i - 1, j) + getValueImage(d_input, i - 1, j + 1)
		- getValueImage(d_input, i + 1, j - 1) - 2 * getValueImage(d_input, i + 1, j) - getValueImage(d_input, i + 1, j + 1);
	float Gy = getValueImage(d_input, i - 1, j - 1) + 2 * getValueImage(d_input, i, j - 1) + getValueImage(d_input, i + 1, j - 1)
		- getValueImage(d_input, i - 1, j + 1) - 2 * getValueImage(d_input, i, j + 1) - getValueImage(d_input, i + 1, j + 1);
	d_output[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
}