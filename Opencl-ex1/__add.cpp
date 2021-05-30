#include <stdio.h>

#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

void vec_add(std::vector<size_t>& num1, std::vector<size_t>& num2, std::vector<size_t>& h_output)
{
	for(size_t i = 0; i< num1.size(); i++)
		h_output[i] = num1[i] + num2[i];
}
int main()
{
	std::cout << "Author: Abdul Rehman" << std::endl;

	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cerr << "No platforms found" << std::endl;
		return 1;
	}
	int platformId = 0;
	for (size_t i = 0; i < platforms.size(); i++) {
		if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "Nvidia Accelerated Parallel Processing") {
			platformId = i;
			break;
		}
	}
	cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[platformId](), 0, 0 };
	std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '" << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
	cl::Context context(CL_DEVICE_TYPE_GPU, prop);

	// Get the first device of the context
	std::cout << "Context has " << context.getInfo<CL_CONTEXT_DEVICES>().size() << " devices" << std::endl;
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);

	// Create a command queue
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	cl::Program program = OpenCL::loadProgramSource(context, "D:/INFOTECH/2nd Semester/High Performance Programming with graphic cards/Exercise/Opencl-ex1/Opencl-ex1/src/add_kernel.cl");
	OpenCL::buildProgram(program, devices);

	std::size_t wgSize = 16; // Number of work items per work group in X direction
	std::size_t count = wgSize * 1000000;
	std::size_t size = count * sizeof(u_int);
	std::vector<size_t> a(count);
	std::vector<size_t> b(count);
	std::vector<size_t> h_outputCpu(count);
	std::vector<size_t> h_outputGpu(count);

	cl::Buffer d_input1(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_input2(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_output(context, CL_MEM_READ_WRITE, size);

	memset(a.data(), 255, size);
	memset(b.data(), 255, size);
	memset(h_outputCpu.data(), 255, size);
	memset(h_outputGpu.data(), 255, size);

	queue.enqueueWriteBuffer(d_input1, true, 0, size, a.data());
	queue.enqueueWriteBuffer(d_input2, true, 0, size, b.data());
	queue.enqueueWriteBuffer(d_output, true, 0, size, h_outputGpu.data());

	for (auto i = 0; i < count; i++)
	{
		a[i] = i + 10;
		b[i] = i + 5;
	}
	Core::TimeSpan cpuStart = Core::getCurrentTime();
	vec_add(a, b, h_outputCpu);
	Core::TimeSpan cpuEnd = Core::getCurrentTime();

	cl::Event copy1;
	queue.enqueueWriteBuffer(d_input1, true, 0, size, a.data(), NULL, &copy1);
	cl::Event copyx;
	queue.enqueueWriteBuffer(d_input2, true, 0, size, b.data(), NULL, &copyx);

	cl::Kernel kernel1(program, "kernel1");

	kernel1.setArg<cl::Buffer>(0, d_output);
	kernel1.setArg<cl::Buffer>(1, d_input1);
	kernel1.setArg<cl::Buffer>(2, d_input2);

	cl::Event execution;
	queue.enqueueNDRangeKernel(kernel1, 0, count, wgSize, NULL, &execution);
	cl::Event download;
	queue.enqueueReadBuffer(d_output, true, 0, size, h_outputGpu.data(), NULL, &download);
	Core::TimeSpan cpuTime = cpuEnd - cpuStart;
	std::cout << "CPU: Number at 2 : " << h_outputCpu[2] << std::endl;
	std::cout << "GPU: Number at 2 : " << h_outputGpu[2] << std::endl;

	Core::TimeSpan copytime = OpenCL::getElapsedTime(copy1) + OpenCL::getElapsedTime(copyx);
	Core::TimeSpan downloadtime = OpenCL::getElapsedTime(download);
	Core::TimeSpan total_copytime = downloadtime+ copytime;
	Core::TimeSpan exec_time = OpenCL::getElapsedTime(execution);
	Core::TimeSpan total_gputime = total_copytime + exec_time;
	std::cout << "Copy Time : " << total_copytime.toString() << std::endl;
	std::cout << "Download Time : " << downloadtime.toString() << std::endl;
	std::cout << "Execution Time GPU: " << exec_time.toString() << std::endl;
	std::cout << "Execution Time CPU: " << cpuTime.toString() << std::endl;
	std::cout << "Execution Speedup: " << cpuTime.getSeconds() / exec_time.getSeconds() << std::endl;
	std::cout << "Overall Speedup for 1 iteration: " << cpuTime.getSeconds() / total_gputime.getSeconds() << std::endl;

	return 0;
}