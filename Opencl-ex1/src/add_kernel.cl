#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

__kernel void kernel1(__global uint *d_output, __global uint *num1, __global uint *num2)
{
	size_t id1 = get_global_id(0);

	d_output[id1] = num1[id1] + num2[id1];
}