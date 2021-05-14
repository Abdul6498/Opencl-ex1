//////////////////////////////////////////////////////////////////////////////
// OpenCL exercise 2: Mandelbrot
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

//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////
void mandelbrotHost (std::vector<cl_uint>& h_output, size_t countX, size_t countY, cl_uint niter, float xmin, float xmax, float ymin, float ymax) {
	for (size_t i = 0; i < countX; i = i + 1) { //loop in the x-direction
		float xc = xmin + (xmax - xmin) / (countX - 1) * i; //xc=real(c)
		for (size_t j = 0; j < countY; j = j + 1) { //loop in the y-direction
			float yc = ymin + (ymax - ymin) / (countY - 1) * j; //yc=imag(c)
			float x = 0.0; //x=real(z_k)
			float y = 0.0; //y=imag(z_k)
			for (size_t k = 0; k < niter; k = k + 1) { //iteration loop
				float tempx = x * x - y * y + xc; //z_{n+1}=(z_n)^2+c;
				y = 2 * x * y + yc;
				x = tempx;
				float r2 = x * x + y * y; //r2=|z_k|^2
				if ((r2 > 4) || k == niter - 1) { //divergence condition
					h_output[i + j * countX] = k;
					break;
				}
			}
		}
	}
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

	// Get the first device of the context
	std::cout << "Context has " << context.getInfo<CL_CONTEXT_DEVICES>().size() << " devices" << std::endl;
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);

	// Create a command queue
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "D:/INFOTECH/2nd Semester/High Performance Programming with graphic cards/Exercise/Opencl-ex1/Opencl-ex1/src/OpenCLExercise2_Mandelbrot.cl");
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	// Create a kernel object
	cl::Kernel mandelbrotKernel(program, "mandelbrotKernel");

	// Parameters for the mandelbrot set
	cl_uint niter; // maximum number of iterations
	float xmin, xmax, ymin, ymax; // limits for c=x+i*y
	int64_t maxError; // maximum difference between CPU and GPU solution (to account for rounding errors)

	// First parameter set
	niter = 20;
	xmin = -2;
	xmax = 1;
	ymin = -1.5;
	ymax = 1.5;
	maxError = 1;

	/* Second parameter set
	niter = 110;
	xmin = -0.813;
	xmax = -0.791;
	ymin = -0.188;
	ymax = -0.166;
	// */

	// Declare some values
	std::size_t wgSizeX = 16; // Number of work items per work group in X direction
	std::size_t wgSizeY = 16;
	std::size_t countX = wgSizeX * 128; // Overall number of work items in X direction = Number of elements in X direction
	std::size_t countY = wgSizeY * 128;
	std::size_t count = countX * countY; // Overall number of elements
	std::size_t size = count * sizeof (cl_uint); // Size of data in bytes

	// Allocate space for output data from CPU and GPU on the host
	std::vector<cl_uint> h_outputCpu (count);
	std::vector<cl_uint> h_outputGpu (count);

	// Allocate space for output data on the device
	//TODO
	cl::Buffer d_output(context, CL_MEM_READ_WRITE, size);
	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
	memset(h_outputCpu.data(), 255, size);
	memset(h_outputGpu.data(), 255, size);
	//TODO
	cl::Event copy1;
	queue.enqueueWriteBuffer(d_output, true, 0, size, h_outputGpu.data());

	// Do calculation on the host side
	Core::TimeSpan cpuStart = Core::getCurrentTime();
	mandelbrotHost(h_outputCpu, countX, countY, niter, xmin, xmax, ymin, ymax);
	Core::TimeSpan cpuEnd = Core::getCurrentTime();

	// Launch kernel on the device
	//TODO
	cl::Event kernel_exec_event = cl::Event();
	mandelbrotKernel.setArg<cl::Buffer>(0, d_output);
	mandelbrotKernel.setArg<cl_uint>(1, niter);
	mandelbrotKernel.setArg<cl_float>(2, xmin);
	mandelbrotKernel.setArg<cl_float>(3, xmax);
	mandelbrotKernel.setArg<cl_float>(4, ymin);
	mandelbrotKernel.setArg<cl_float>(5, ymax);
	queue.enqueueNDRangeKernel(mandelbrotKernel, 0, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &kernel_exec_event);
	queue.finish();
	// Copy output data back to host
	//TODO
	cl::Event download_event;
	auto kernel_compute_time = OpenCL::getElapsedTime(kernel_exec_event);
	queue.enqueueReadBuffer(d_output, true, 0, size, h_outputGpu.data(), NULL, &download_event);
	auto download_time = OpenCL::getElapsedTime(download_event);
	
	// Print performance data
	//TODO
	Core::TimeSpan cpuTime = cpuEnd - cpuStart;
	std::cout << "CPU Time: " << cpuTime.toString() << std::endl;
	std::cout << "GPU Time: " << kernel_compute_time.toString() << std::endl;
	std::cout << "GPU Download: " << download_time.toString() << std::endl;
	std::cout << "Speedup: " << cpuTime.getMilliseconds() / kernel_compute_time.getMilliseconds() << std::endl;
	std::cout << "Speedup with overhead: " << cpuTime.getMilliseconds() / (kernel_compute_time + download_time).getMilliseconds() << std::endl;


	//////// Store output images ///////////////////////////////////
	std::vector<float> imageDataCpu(count);
	std::vector<float> imageDataGpu(count);
	for (size_t i = 0; i < countX; i++) {
		for (size_t j = 0; j < countY; j++) {
			// Invert y-axis, convert to float
			imageDataCpu[i + countX * (countY - j - 1)] = 1 - 1.0f * h_outputCpu[i + j * countX] / (niter - 1);
			imageDataGpu[i + countX * (countY - j - 1)] = 1 - 1.0f * h_outputGpu[i + j * countX] / (niter - 1);
		}
	}
	Core::writeImagePGM("output_mandelbrot_bw_cpu.pgm", imageDataCpu, countX, countY);
	Core::writeImagePGM("output_mandelbrot_bw_gpu.pgm", imageDataGpu, countX, countY);
	Core::writeImagePPM("output_mandelbrot_col_cpu.ppm", imageDataCpu, countX, countY);
	Core::writeImagePPM("output_mandelbrot_col_gpu.ppm", imageDataGpu, countX, countY);

	// Check whether results are correct
	std::size_t errorCount = 0;
	for (size_t i = 0; i < countX; i = i + 1) { //loop in the x-direction
		for (size_t j = 0; j < countY; j = j + 1) { //loop in the y-direction
			size_t index = i + j * countX;
			// Allow small differences between CPU and GPU results (due to different rounding behavior)
			if (!(std::abs ((int64_t) h_outputCpu[index] - (int64_t) h_outputGpu[index]) <= maxError)) {
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

	std::cout << "Success" << std::endl;

	return 0;
}
