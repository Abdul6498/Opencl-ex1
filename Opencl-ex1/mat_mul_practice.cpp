// includes
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

#include <boost/lexical_cast.hpp>

//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////
void matrixMulHost(const std::vector<float>& h_inputA, const std::vector<float>& h_inputB, 
	std::vector<float>& h_outputC, std::size_t countAX_BY, std::size_t countAY, std::size_t countBX) 
{
	for (std::size_t j = 0; j < countAY; j++) {
		for (std::size_t i = 0; i < countBX; i++) {
			float sum = 0;
			for (std::size_t k = 0; k < countAX_BY; k++) {
				float a = h_inputA[k + j * countAX_BY];
				float b = h_inputB[i + k * countBX];
				sum += a * b;
			}
			h_outputC[i + j * countBX] = sum;
		}
	}
}

void printperformance_data(std::string cpu_time, std::string sendtime, std::string gpu_time, std::string rec_time, float speedup, float speedup_w_m)
{
	std::cout << "CPU Time : " << cpu_time << std::endl;
	std::cout << "Copy Time : " << sendtime << std::endl;
	std::cout << "GPU Time : " << gpu_time << std::endl;
	std::cout << "Download Time : " << rec_time << std::endl;
	std::cout << "Speed Up : " << speedup << std::endl;
	std::cout << "Speed Up with memory copy : " << speedup_w_m << std::endl;
}

int main(int argc, char** argv)
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
		if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "NVIDIA Accelerated Parallel Processing") {
			platformId = i;
			break;
		}
	}
	cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[platformId](), 0, 0 };
	std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '" << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
	cl::Context context(CL_DEVICE_TYPE_GPU, prop);


	// Get a device of the context
	int deviceNr = argc < 2 ? 1 : atoi(argv[1]);
	std::cout << "Using device " << deviceNr << " / " << context.getInfo<CL_CONTEXT_DEVICES>().size() << std::endl;
	ASSERT(deviceNr > 0);
	ASSERT((size_t)deviceNr <= context.getInfo<CL_CONTEXT_DEVICES>().size());
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceNr - 1];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);


	// Create a command queue
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	cl::Program program = OpenCL::loadProgramSource(context, "D:/INFOTECH/2nd Semester/High Performance Programming with graphic cards/Exercise/Opencl-ex1/Opencl-ex1/src/mat_mul_kernel.cl");
	OpenCL::buildProgram(program, devices);

	std::size_t wgSize = 16;
	std::size_t countAX_BY = 512;
	std::size_t countAY = 1024;
	std::size_t countBX = 768;

	std::size_t countCX = countBX;
	std::size_t countCY = countAY;
	std::size_t countA = countAX_BY * countAY;
	std::size_t countB = countBX * countAX_BY;
	std::size_t countC = countCX * countCY;
	std::size_t sizeA = countA * sizeof(float);
	std::size_t sizeB = countB * sizeof(float);
	std::size_t sizeC = countC * sizeof(float);

	std::vector<float> h_inputA(countA);
	std::vector<float> h_inputB(countB);
	std::vector<float> h_outputCCpu(countC);
	std::vector<float> h_outputCGpu(countC);

	cl::Buffer d_inputA(context, CL_MEM_READ_WRITE, sizeA);
	cl::Buffer d_inputB(context, CL_MEM_READ_WRITE, sizeB);
	cl::Buffer d_outputC(context, CL_MEM_READ_WRITE, sizeC);

	memset(h_inputA.data(), 255, sizeA);
	memset(h_inputB.data(), 255, sizeB);
	memset(h_outputCCpu.data(), 255, sizeC);
	memset(h_outputCGpu.data(), 255, sizeC);

	
	queue.enqueueWriteBuffer(d_inputA, true, 0, sizeA, h_inputA.data());
	queue.enqueueWriteBuffer(d_inputB, true, 0, sizeB, h_inputB.data());
	queue.enqueueWriteBuffer(d_outputC, true, 0, sizeC, h_outputCGpu.data());

	for (std::size_t i = 0; i < countA; i++)
		h_inputA[i] = (rand() % 100) / 5.0f - 10.0f;
	for (std::size_t i = 0; i < countB; i++)
		h_inputB[i] = (rand() % 100) / 5.0f - 10.0f;

	Core::TimeSpan cpuStart = Core::getCurrentTime();
	matrixMulHost(h_inputA, h_inputB, h_outputCCpu, countAX_BY, countAY, countBX);
	Core::TimeSpan cpuEnd = Core::getCurrentTime();
	Core::TimeSpan cpuexec = cpuEnd - cpuStart;
	auto cputime = (cpuEnd - cpuStart).toString();

	queue.enqueueWriteBuffer(d_inputA, true, 0, sizeA, h_inputA.data());
	queue.enqueueWriteBuffer(d_inputB, true, 0, sizeB, h_inputB.data());
	queue.enqueueWriteBuffer(d_outputC, true, 0, sizeC, h_outputCGpu.data());

	cl::Kernel kernel1(program, "kernel1");
	kernel1.setArg<cl::Buffer>(0, d_inputA);
	kernel1.setArg<cl::Buffer>(1, d_inputB);
	kernel1.setArg<cl::Buffer>(2, d_outputC);
	kernel1.setArg<cl_uint>(3, countAX_BY);
	kernel1.setArg<cl_uint>(4, countAY);
	kernel1.setArg<cl_uint>(5, countBX);

	queue.enqueueNDRangeKernel(kernel1, cl::NullRange, cl::NDRange(countCX, countCY), cl::NDRange(wgSize, wgSize));

	queue.enqueueReadBuffer(d_outputC, true, 0, sizeC, h_outputCGpu.data());

	std::cout << "Data at 5 CPU: " << h_outputCCpu[5] << std::endl;
	std::cout << "Data at 5 GPU: " << h_outputCGpu[5] << std::endl;

	return 0;
}
