//////////////////////////////////////////////////////////////////////////////
// OpenCL exercise 6: Prefix sum (Scan)
//////////////////////////////////////////////////////////////////////////////

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
#include <iomanip>
#include <sstream>

#include <boost/lexical_cast.hpp>

//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////
void prefixSumHost(const std::vector<cl_int>& h_input, std::vector<cl_int>& h_output) {
	if (h_input.size () == 0)
		return;
	cl_int sum = h_input[0];
	h_output[0] = sum;
	for (std::size_t i = 1; i < h_input.size (); i++) {
		sum += h_input[i];
		h_output[i] = sum;
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
	std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
	if(gpu_time > cpu_time)
		std::cout << "GPU WIN in execution" << std::endl;
	else
		std::cout << "CPU WIN in execution" << std::endl;
	if (speedup_w_m < 1)
		std::cout << "GPU Lose with memory copy" << std::endl;
	else
		std::cout << "GPU WIN with memory copy" << std::endl;
}

//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	// Create a context
	//cl::Context context(CL_DEVICE_TYPE_GPU);
        std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cerr << "No platforms found" << std::endl;
		return 1;
	}
	int platformId = 0;
	for (size_t i = 0; i < platforms.size(); i++) {
		if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing") {
			platformId = i;
			break;
		}
	}
	cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[platformId] (), 0, 0 };
	std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '" << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
	cl::Context context(CL_DEVICE_TYPE_GPU, prop);

	// Get a device of the context
	int deviceNr = argc < 2 ? 1 : atoi(argv[1]);
	std::cout << "Using device " << deviceNr << " / " << context.getInfo<CL_CONTEXT_DEVICES>().size() << std::endl;
	ASSERT (deviceNr > 0);
	ASSERT ((size_t) deviceNr <= context.getInfo<CL_CONTEXT_DEVICES>().size());
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceNr - 1];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);

	// Create a command queue
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	// Declare some values
	std::size_t wgSize = 64; // Number of work items per work group
	std::size_t count = wgSize * wgSize * wgSize; // Number of values
	std::size_t count_temp1 = wgSize * wgSize; // Number of values
	std::size_t count_temp = wgSize; // Number of values
	std::size_t size_temp1 = wgSize * wgSize * sizeof(cl_int);
	std::size_t size_temp = wgSize * sizeof(cl_int);

	std::size_t size = count * sizeof (cl_int);

	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "D:/INFOTECH/2nd Semester/High Performance Programming with graphic cards/Exercise/Opencl-ex1/Opencl-ex1/src/OpenCLExercise6_PrefixSum.cl");
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	// This will pass the value of wgSize as a preprocessor constant "WG_SIZE" to the OpenCL C compiler
	OpenCL::buildProgram(program, devices, "-DWG_SIZE=" + boost::lexical_cast<std::string>(wgSize));
	std::cout << "Program Complied" << std::endl;
	// Allocate space for output data from CPU and GPU on the host
	std::vector<cl_int> h_input (count);
	std::vector<cl_int> h_outputCpu (count);
	std::vector<cl_int> h_temp1 (wgSize * wgSize);
	std::vector<cl_int> h_temp2 (wgSize);
	std::vector<cl_int> h_outputGpu (count);

	// Allocate space for input and output data on the device
	//TODO
	cl::Buffer d_input(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_output(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_temp1(context, CL_MEM_READ_WRITE, size_temp1);
	cl::Buffer d_temp(context, CL_MEM_READ_WRITE, size);
	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
	memset(h_input.data(), 255, size);
	memset(h_temp1.data(), 255, wgSize * wgSize * sizeof (cl_int));
	memset(h_temp2.data(), 255, wgSize * sizeof (cl_int));
	memset(h_outputCpu.data(), 255, size);
	memset(h_outputGpu.data(), 255, size);
	//TODO: GPU

	queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data());
	queue.enqueueWriteBuffer(d_output, true, 0, size, h_outputGpu.data());


	//////// Generate input data ////////////////////////////////
	// Use random input data
	for (std::size_t i = 0; i < count; i++)
		h_input[i] = rand() % 100 - 40;
	// Or: Use consecutive integer numbers as data
	/*
	for (std::size_t i = 0; i < count; i++)
		h_input[i] = i;
	// */

	// Do calculation on the host side
	Core::TimeSpan cpuStart = Core::getCurrentTime();
	prefixSumHost(h_input, h_outputCpu);
	Core::TimeSpan cpuEnd = Core::getCurrentTime();
	Core::TimeSpan cpuexec = cpuEnd - cpuStart;
	auto cputime = (cpuEnd - cpuStart).toString();

	// Create kernels
	//TODO
	cl::Kernel prefixSumKernel(program, "prefixSumKernel");
	cl::Kernel blockAddKernel(program, "blockAddKernel");

	// Copy input data to device
	//TODO
	cl::Event copy1;
	queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data(), NULL, &copy1);
	// Call the kernels
	//TODO
	prefixSumKernel.setArg<cl::Buffer>(0, d_input);
	prefixSumKernel.setArg<cl::Buffer>(1, d_output);
	prefixSumKernel.setArg<cl::Buffer>(2, d_temp1);
	cl::Event execution;
	queue.enqueueNDRangeKernel(prefixSumKernel, cl::NullRange, count, wgSize, NULL, &execution);

	prefixSumKernel.setArg<cl::Buffer>(0, d_temp1);
	prefixSumKernel.setArg<cl::Buffer>(1, d_temp1);
	prefixSumKernel.setArg<cl::Buffer>(2, d_temp);
	queue.enqueueNDRangeKernel(prefixSumKernel, cl::NullRange, count_temp1, wgSize, NULL, &execution);

	prefixSumKernel.setArg<cl::Buffer>(0, d_temp);
	prefixSumKernel.setArg<cl::Buffer>(1, d_temp);
	prefixSumKernel.setArg<cl::Buffer>(2, cl::Buffer());
	queue.enqueueNDRangeKernel(prefixSumKernel, cl::NullRange, count_temp, wgSize, NULL, &execution);

	blockAddKernel.setArg<cl::Buffer>(0, d_temp1);
	blockAddKernel.setArg<cl::Buffer>(1, d_temp);
	queue.enqueueNDRangeKernel(blockAddKernel, cl::NullRange, count_temp1, wgSize, NULL, &execution);

	blockAddKernel.setArg<cl::Buffer>(0, d_output);
	blockAddKernel.setArg<cl::Buffer>(1, d_temp1);
	queue.enqueueNDRangeKernel(blockAddKernel, cl::NullRange, count, wgSize, NULL, &execution);

	// Copy output data back to host
	//TODO
	cl::Event download;
	queue.enqueueReadBuffer(d_output, true, 0, size, h_outputGpu.data(), NULL, &download);
	// Print performance data
	//TODO
	Core::TimeSpan copy_time = OpenCL::getElapsedTime(copy1);
	Core::TimeSpan exec_time = OpenCL::getElapsedTime(execution);
	Core::TimeSpan download_time = OpenCL::getElapsedTime(download);
	auto sendtime = copy_time.toString();
	auto gpu_time = exec_time.toString();
	auto rec_time = download_time.toString();
	auto speedup = cpuexec.getSeconds() / exec_time.getSeconds();
	auto speedup_w_m = cpuexec.getSeconds() / (exec_time.getSeconds() + copy_time.getSeconds() + download_time.getSeconds());
	printperformance_data(cputime, sendtime, gpu_time, rec_time, speedup, speedup_w_m);
	// Check whether results are correct
	std::size_t errorCount = 0;
	for (size_t i = 0; i < count; i = i + 1) {
		if (h_outputCpu[i] != h_outputGpu[i]) {
			if (errorCount < 15)
				std::cout << "Result at " << i << " is incorrect: GPU value is " << h_outputGpu[i] << ", CPU value is " << h_outputCpu[i] << std::endl;
			else if (errorCount == 15)
				std::cout << "..." << std::endl;
			errorCount++;
		}
	}
	if (errorCount != 0) {
		std::cout << "Found " << errorCount << " incorrect results" << std::endl;
		return 1;
	}

	std::cout << "Success" << std::endl;

	return 0;
}
