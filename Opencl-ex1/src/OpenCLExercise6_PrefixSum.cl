#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
__kernel void prefixSumKernel(__global int* d_input,
	__global int* d_output, 
	__global int* d_block_output) 
{
	int gi = get_global_id(0);
	int li = get_local_id(0);

	__local int l_A[WG_SIZE];
	l_A[li] = d_input[gi];
	barrier(CLK_LOCAL_MEM_FENCE);

	int sum = l_A[li];
	int offset = 1;

	for (int jump = 1; jump <= li; jump *= 2)
	{
		sum += l_A[li - jump];
		barrier(CLK_LOCAL_MEM_FENCE);
		l_A[li] = sum;
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	d_output[gi] = sum;
	if (d_block_output && li == WG_SIZE - 1)
	{
		d_block_output[get_group_id(0)] = sum;
	}
	
}

__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
__kernel void blockAddKernel(__global int *d_output, __global int *d_blocks) {
	//TODO
	int gi = get_global_id(0);
	int wi = get_group_id(0);

	if (wi > 0)
	{
		d_output[gi] += d_blocks[wi - 1];
	}
}
