#include "Opencl-ex1.h"
#include <stdio.h>

#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

void calculateHost(const std::vector<float>& h_input, std::vector<float>& h_output) {
	for (std::size_t i = 0; i < h_output.size(); i++)
		h_output[i] = std::cos(h_input[i]);
}

int main()
{
	std::cout << " Hello World" << std::endl;

	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	if (platforms.size() == 0)
	{
		std::cerr << "No platforms found" << std::endl;
		return 1;
	}
	int platformId = 0;
	cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[platformId](), 0, 0 };
	std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '" << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
	cl::Context context(CL_DEVICE_TYPE_GPU, prop);
	std::cout << "Context has " << context.getInfo<CL_CONTEXT_DEVICES>().size() << " devices" << std::endl;
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);

	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
	cl::Program program = OpenCL::loadProgramSource(context, "D:/INFOTECH/2nd Semester/High Performance Programming with graphic cards/Exercise/Opencl-ex1/Opencl-ex1/src/try_kernel.cl");
	OpenCL::buildProgram(program, devices);

	std::size_t vc_size = 100000;
	std::size_t wgSize = 64;
	std::size_t count = 128 * 100000;
	std::size_t size = count * sizeof(float);
	std::vector<float> h_input(count);
	std::vector<float> h_outputCpu(count);
	std::vector<float> h_outputGpu(count);

	//Create Buffer and allocate space
	cl::Buffer d_input(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_output(context, CL_MEM_READ_WRITE, size);

	std::cout << "Size : " << size << std::endl;

	memset(h_input.data(), 255, size);
	memset(h_outputCpu.data(), 255, size);
	memset(h_outputGpu.data(), 255, size);
	queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data());
	queue.enqueueWriteBuffer(d_output, true, 0, size, h_outputGpu.data());

	for (int i = 0; i < count; i++)
		h_input[i] = ((i * 1009) % 31) * 0.1;

	Core::TimeSpan cpuStart = Core::getCurrentTime();
	calculateHost(h_input, h_outputCpu);
	Core::TimeSpan cpuEnd = Core::getCurrentTime();
	
	cl::Event copy1;
	queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data(), NULL, &copy1);
	cl::Event execution;
	cl::Kernel kernel1(program, "kernel1");
	kernel1.setArg<cl::Buffer>(0, d_input);
	kernel1.setArg<cl::Buffer>(1, d_output);

	queue.enqueueNDRangeKernel(kernel1, 0, count, wgSize, NULL, &execution);

	cl::Event copy2;
	queue.enqueueReadBuffer(d_output, true, 0, size, h_outputGpu.data(), NULL, &copy2);
	
	Core::TimeSpan cpuTime = cpuEnd - cpuStart;
	Core::TimeSpan gpuTime = OpenCL::getElapsedTime(execution);
	Core::TimeSpan copyTime1 = OpenCL::getElapsedTime(copy1);
	Core::TimeSpan copyTime2 = OpenCL::getElapsedTime(copy2);
	Core::TimeSpan copyTime = copyTime1 + copyTime2;
	Core::TimeSpan overallGpuTime = gpuTime + copyTime;
	std::cout << "CPU Time: " << cpuTime.toString() << std::endl;
	std::cout << "Memory copy Time: " << copyTime.toString() << std::endl;
	std::cout << "GPU Time w/o memory copy: " << gpuTime.toString() << " (speedup = " << (cpuTime.getSeconds() / gpuTime.getSeconds()) << ")" << std::endl;
	std::cout << "GPU Time with memory copy: " << overallGpuTime.toString() << " (speedup = " << (cpuTime.getSeconds() / overallGpuTime.getSeconds()) << ")" << std::endl;
	std::cout << "Success" << std::endl;
	return 0;
}