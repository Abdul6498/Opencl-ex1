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

void printperformance_data(std::string cpu_time, std::string sendtime, std::string gpu_time, std::string rec_time, float speedup, float speedup_w_m);
//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////
int getIndexGlobal(std::size_t countX, int i, int j) {
	return j * countX + i;
}
// Read value from global array a, return 0 if outside image
float getValueGlobal(const std::vector<float>& a, std::size_t countX, std::size_t countY, int i, int j) {
	if (i < 0 || (size_t)i >= countX || j < 0 || (size_t)j >= countY)
		return 0;
	else
		return a[getIndexGlobal(countX, i, j)];
}
void sobelHost(const std::vector<float>& h_input, std::vector<float>& h_outputCpu, std::size_t countX, std::size_t countY) {
	for (int i = 0; i < (int)countX; i++) {
		for (int j = 0; j < (int)countY; j++) {
			float Gx = getValueGlobal(h_input, countX, countY, i - 1, j - 1) + 2 * getValueGlobal(h_input, countX, countY, i - 1, j) + getValueGlobal(h_input, countX, countY, i - 1, j + 1)
				- getValueGlobal(h_input, countX, countY, i + 1, j - 1) - 2 * getValueGlobal(h_input, countX, countY, i + 1, j) - getValueGlobal(h_input, countX, countY, i + 1, j + 1);
			float Gy = getValueGlobal(h_input, countX, countY, i - 1, j - 1) + 2 * getValueGlobal(h_input, countX, countY, i, j - 1) + getValueGlobal(h_input, countX, countY, i + 1, j - 1)
				- getValueGlobal(h_input, countX, countY, i - 1, j + 1) - 2 * getValueGlobal(h_input, countX, countY, i, j + 1) - getValueGlobal(h_input, countX, countY, i + 1, j + 1);
			h_outputCpu[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
		}
	}
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

	cl::Program program = OpenCL::loadProgramSource(context, "D:/INFOTECH/2nd Semester/High Performance Programming with graphic cards/Exercise/Opencl-ex1/Opencl-ex1/src/sobel_kernel.cl");
	OpenCL::buildProgram(program, devices);
	
	std::size_t wgSizeX = 16;	//Number of workitems per group in X direction --size of work groups
	std::size_t wgSizeY = 16;	//Number of workitems per group in Y direction
	std::size_t countX = wgSizeX * 40;	//Image X direction 640 = 16*40 -- Total Work items in X direction
	std::size_t countY = wgSizeY * 30;	//Image X direction 480 = 16*30
	std::size_t	count = countX * countY;	//Total elements
	std::size_t size = count * sizeof(float); //size of data in bytes

	std::vector<float> h_outputCpu(count);
	std::vector<float> h_outputGpu(count);
	std::vector<float> h_input(count);

	memset(h_input.data(), 255, size);
	memset(h_outputCpu.data(), 255, size);
	memset(h_outputGpu.data(), 255, size);

	cl::Buffer d_input(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_output(context, CL_MEM_READ_WRITE, size);

	queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data());
	queue.enqueueWriteBuffer(d_output, true, 0, size, h_outputGpu.data());

	//////// Load input data ////////////////////////////////
	{
		std::vector<float> inputData;
		std::size_t inputWidth, inputHeight;
		Core::readImagePGM("D:/INFOTECH/2nd Semester/High Performance Programming with graphic cards/Exercise/Opencl-ex1/Opencl-ex1/Valve.pgm", inputData, inputWidth, inputHeight);
		std::cout << "Input Data size: " << inputData.size() << std::endl;
		for (size_t j = 0; j < countY; j++) {
			for (size_t i = 0; i < countX; i++) {
				h_input[i + countX * j] = inputData[(i % inputWidth) + inputWidth * (j % inputHeight)]; //converted to 1D input data = 307200, Range: 0 to 307199
			}
		}
	}

	// Do calculation on the host side
	Core::TimeSpan cpuStart = Core::getCurrentTime();
	sobelHost(h_input, h_outputCpu, countX, countY);
	Core::TimeSpan cpuEnd = Core::getCurrentTime();
	Core::TimeSpan cpuexec = cpuEnd - cpuStart;
	auto cputime = (cpuEnd - cpuStart).toString();
	//////// Store CPU output image ///////////////////////////////////
	Core::writeImagePGM("practice_sobel_cpu.pgm", h_outputCpu, countX, countY);

	std::cout << std::endl;

	// Iterate over all implementations (task 1 - 3)
	 for (int impl = 1; impl <= 3; impl++) {
		std::cout << "Implementation #" << impl << ":" << std::endl;

		// Reinitialize output memory to 0xff
		memset(h_outputGpu.data(), 255, size);
		//TODO: GPU

		// Copy input data to device
		//TODO
		cl::Event copy1;
		cl::Image2D image;
		if (impl == 3)
		{
			// Task 3
			image = cl::Image2D(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_R, CL_FLOAT), countX, countY);
			cl::size_t<3> origin;
			origin[0] = origin[1] = origin[2] = 0;
			cl::size_t<3> region;
			region[0] = countX;
			region[1] = countY;
			region[2] = 1;
			queue.enqueueWriteImage(image, true, origin, region, countX * sizeof(float), 0, h_input.data(), NULL, &copy1);
			std::cout << "Test 1" << std::endl;
		}
		else
		{
			//Task 1 and 2
			queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data(), NULL, &copy1);
		}
		// Create a kernel object
		std::cout << "Test impl: " << impl << std::endl;
		std::string kernelName = "kernel" + boost::lexical_cast<std::string> (impl);
		cl::Kernel sobelKernel(program, kernelName.c_str());
		

		// Launch kernel on the device
		//TODO
		if (impl == 3)
		{
			std::cout << "Test 3.1" << std::endl;
			sobelKernel.setArg<cl::Image2D>(0, image);
			sobelKernel.setArg<cl::Buffer>(1, d_output);
			std::cout << "Test 3" << std::endl;
		}
		else
		{
			sobelKernel.setArg<cl::Buffer>(0, d_input);
			sobelKernel.setArg<cl::Buffer>(1, d_output);
		}

		cl::Event execution;
		queue.enqueueNDRangeKernel(sobelKernel, cl::NullRange, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &execution);
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
		auto speedup_w_m = cpuexec.getSeconds() / (exec_time.getSeconds()+ copy_time.getSeconds() + download_time.getSeconds());
		printperformance_data(cputime, sendtime, gpu_time, rec_time, speedup, speedup_w_m);
		//////// Store GPU output image ///////////////////////////////////
		Core::writeImagePGM("practice_sobel_gpu_" + boost::lexical_cast<std::string> (impl) + ".pgm", h_outputGpu, countX, countY);

		// Check whether results are correct
		std::size_t errorCount = 0;
		for (size_t i = 0; i < countX; i = i + 1) { //loop in the x-direction
			for (size_t j = 0; j < countY; j = j + 1) { //loop in the y-direction
				size_t index = i + j * countX;
				// Allow small differences between CPU and GPU results (due to different rounding behavior)
				if (!(std::abs(h_outputCpu[index] - h_outputGpu[index]) <= 1e-5)) {
					if (errorCount < 15)
						std::cout << "Result for " << i << "," << j << " is incorrect: GPU value is " << h_outputGpu[index] << ", CPU value is " << h_outputCpu[index] << std::endl;
					else if (errorCount == 15)
						std::cout << "..." << std::endl;
					errorCount++;
				}
			}
		}
		if (errorCount != 0) {
			std::cout << "Found " << errorCount << " incorrect results" << std::endl;
			return 1;
		}

		std::cout << std::endl;
	}
	
	std::cout << "Success" << std::endl;
	return 0;
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