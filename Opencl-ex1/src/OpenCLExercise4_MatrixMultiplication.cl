#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

__kernel void matrixMulKernel1(
	__global float *d_inputA, 
	__global float *d_inputB, 
	__global float *d_outputC, 
	uint *countAX_BY) 
{
	//TODO
	int i = get_global_id(0);
	int j = get_global_id(1);

	size_t countBX = get_global_size(0);
	size_t countAY = get_global_size(1);
	//size_t countAX_BY = get_global_size(2);

	float sum = 0;

	for (uint k = 0; k < countAX_BY; k++) {
		float a = d_inputA[k + j * countAX_BY];
		float b = d_inputB[i + k * countBX];
		sum += a * b;
	}
	d_outputC[i + j * countBX] = sum;
}

// The preprocessor constant WG_SIZE will contain the size of a work group in X/Y-direction

//__attribute__((reqd_work_group_size(WG_SIZE, WG_SIZE, 1)))
//__kernel void matrixMulKernel2(/*...*/) {
	//TODO
//}
