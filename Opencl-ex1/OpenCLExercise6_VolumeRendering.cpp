//////////////////////////////////////////////////////////////////////////////
// OpenCL exercise 4: Volume rendering
//////////////////////////////////////////////////////////////////////////////

// includes
#include <stdio.h>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/glx.h>

#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>
#include <CT/DataFiles.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <sstream>

#include <boost/lexical_cast.hpp>

bool runCpu = true;	//Run- True, Stop- False
bool runGpu = true;	//Run- True, Stop- False
bool displayGpu = true;	//Run- True, Stop- False
bool writeImages = false;	//Run- True, Stop- False

void keyboardGL(unsigned char key, int x, int y);
void displayGL();	//display
void setIdle();
void idleGL();

//////////////////////////////////////////////////////////////////////////////
// float3 implementation for CPU
//////////////////////////////////////////////////////////////////////////////
struct float3 {
	float x, y, z;
	float3 () {}	//constructor
	float3 (float f) : x(f), y(f), z(f) {}	//constructor
	float3 (float x, float y, float z) : x(x), y(y), z(z) {}	//constructor
	float3 operator+(float3 f) {
		return float3(x+f.x, y+f.y, z+f.z);
	}
	float3 operator-(float3 f) {
		return float3(x-f.x, y-f.y, z-f.z);
	}
	float3 operator*(float3 f) {
		return float3(x*f.x, y*f.y, z*f.z);
	}
	float3 operator/(float3 f) {
		return float3(x/f.x, y/f.y, z/f.z);
	}
};
inline float3 min(float3 f1, float3 f2) {
	return float3(std::min(f1.x, f2.x), std::min(f1.y, f2.y), std::min(f1.z, f2.z));
}
inline float3 max(float3 f1, float3 f2) {
	return float3(std::max(f1.x, f2.x), std::max(f1.y, f2.y), std::max(f1.z, f2.z));
}
inline float dot(float3 f1, float3 f2) {
	return f1.x*f2.x + f1.y*f2.y + f1.z*f2.z;
}
inline float3 normalize(float3 f) {
	return f / std::sqrt(dot(f, f));
}

//////////////////////////////////////////////////////////////////////////////
// Trilinear interpolation (done using images on GPU)
//////////////////////////////////////////////////////////////////////////////
inline float interp3_get(const float* data, std::size_t countX, std::size_t countY, std::size_t countZ, int x, int y, int z) {
	if (x < 0 || (std::size_t) x >= countX
		|| y < 0 || (std::size_t) y >= countY
		|| z < 0 || (std::size_t) z >= countZ)
		return 0;
	return data[x + countX * (y + countY * z)];
}
inline float interp3(const float* data, std::size_t countX, std::size_t countY, std::size_t countZ, float3 pos) {
	pos = pos - 0.5f;
	int x = (int) pos.x;
	int y = (int) pos.y;
	int z = (int) pos.z;
	float alphaX = pos.x - x;
	float alphaY = pos.y - y;
	float alphaZ = pos.z - z;
	return (1-alphaX) * (1-alphaY) * (1-alphaZ) * interp3_get (data, countX, countY, countZ, x, y, z)
		+ (alphaX) * (1-alphaY) * (1-alphaZ) * interp3_get (data, countX, countY, countZ, x+1, y, z)
		+ (1-alphaX) * (alphaY) * (1-alphaZ) * interp3_get (data, countX, countY, countZ, x, y+1, z)
		+ (alphaX) * (alphaY) * (1-alphaZ) * interp3_get (data, countX, countY, countZ, x+1, y+1, z)
		+ (1-alphaX) * (1-alphaY) * (alphaZ) * interp3_get (data, countX, countY, countZ, x, y, z+1)
		+ (alphaX) * (1-alphaY) * (alphaZ) * interp3_get (data, countX, countY, countZ, x+1, y, z+1)
		+ (1-alphaX) * (alphaY) * (alphaZ) * interp3_get (data, countX, countY, countZ, x, y+1, z+1)
		+ (alphaX) * (alphaY) * (alphaZ) * interp3_get (data, countX, countY, countZ, x+1, y+1, z+1);
}

//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////
// r_o - Ray prigin, r_d - Ray direction
int intersectBox(float3 r_o, float3 r_d, float3 boxmin, float3 boxmax, float *tnear, float *tfar) {
	// compute intersection of ray with all six bbox planes
	float3 invR = float3(1.0f, 1.0f, 1.0f) / r_d;
	float3 tbot = invR * (boxmin - r_o);
	float3 ttop = invR * (boxmax - r_o);

	// re-order intersections to find smallest and largest on each axis
	float3 tmin = min(ttop, tbot);
	float3 tmax = max(ttop, tbot);

	// find the largest tmin and the smallest tmax
	float largest_tmin = std::max(std::max(tmin.x, tmin.y), tmin.z);
	float smallest_tmax = std::min(std::min(tmax.x, tmax.y), tmax.z);

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}
void renderHost(const float* h_input, std::vector<float>& h_output, std::size_t countX, std::size_t countY, std::size_t countZ, std::size_t outX, std::size_t outY, const float* invViewMatrix, float tstep, float brightness) {
	for (size_t y = 0; y < outY; y++) {
		for (size_t x = 0; x < outX; x++) {
			uint index = (y * outX) + x;

			float u = (x / (float) (outX - 1))*2.0f-1.0f;
			float v = (y / (float) (outY - 1))*2.0f-1.0f;

			float3 boxMin = float3(0, 0, 0);
			float3 boxMax = float3(countX, countY, countZ);	//Volume size

			// calculate eye ray in world space
			float3 eyeRay_o;
			float3 eyeRay_d;

			eyeRay_o = float3(invViewMatrix[3], invViewMatrix[7], invViewMatrix[11]);

			float3 temp = normalize(float3(u, v, -2.0f));
			eyeRay_d.x = dot(temp, (float3(invViewMatrix[0],invViewMatrix[1],invViewMatrix[2])));	//covert to world matrix
			eyeRay_d.y = dot(temp, (float3(invViewMatrix[4],invViewMatrix[5],invViewMatrix[6])));
			eyeRay_d.z = dot(temp, (float3(invViewMatrix[8],invViewMatrix[9],invViewMatrix[10])));

			// find intersection with box
			float tnear, tfar;
			int hit = intersectBox(eyeRay_o, eyeRay_d, boxMin, boxMax, &tnear, &tfar);
			if (!hit) {
				// set output to 0
				h_output[index] = 0;
				continue;
			}
			if (tnear < 0.0f)
				tnear = 0.0f;     // clamp to near plane

			// march along ray from back to front, accumulating color
			float sum = 0;
			for (float t = tfar; t >= tnear; t -= tstep) {
				float3 pos = eyeRay_o + eyeRay_d*t;

				// do 3D interpolation
				float sample = interp3(h_input, countX, countY, countZ, pos);

				// accumulate result
				sum += sample;
			}

			// write output value
			h_output[index] = sum * brightness;
		}
	}
}

//////////////////////////////////////////////////////////////////////////////
// Variables used by render function
//////////////////////////////////////////////////////////////////////////////
cl::Context context;
cl::CommandQueue queue;
cl::Buffer d_output;
cl::Buffer d_invViewMatrix;
cl::Image3D d_input;
cl::Kernel renderKernel;
std::size_t outX;
std::size_t outY;
std::size_t sizeOutput;
const float* h_input;
std::vector<float> h_outputCpu;
std::vector<float> h_outputGpu;
cl::Event copyToDev;
std::size_t countOutput;
std::size_t countX;
std::size_t countY;
std::size_t countZ;
GLuint pbo = 0;
bool animate = true;
float alpha = 0; // Current angle

//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	outX = 1024;	//GL window size
	outY = 768;

	ASSERT(runCpu || runGpu);
	ASSERT(runGpu || !displayGpu);
	ASSERT(runCpu || displayGpu);

	// Load input data
	//TODO: Run the code with different input files
	boost::shared_ptr<Volume> volumeFile = HDF5::matlabDeserialize<Volume> ("../rpi-16.hdf5");
	//boost::shared_ptr<Volume> volumeFile = HDF5::matlabDeserialize<Volume> ("/usr/local.nfs/pas/teaching/gpulab/2016-1/rpi-2.hdf5");
	boost::shared_ptr<const Math::Array<float, 3> > volumeData = volumeFile->transformedTransposedVolume ();
	countX = volumeData->size<0> ();
	countY = volumeData->size<1> ();
	countZ = volumeData->size<2> ();
	h_input = volumeData->data ();

	// Calculate some values
	countOutput = outX * outY;
	sizeOutput = countOutput * sizeof (float);

	// initialize GLUT
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - outX/2,
							glutGet(GLUT_SCREEN_HEIGHT)/2 - outY/2);
	glutInitWindowSize(outX, outY);
	glutCreateWindow("OpenCL exercise 6: Volume rendering");
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

	// register glut callbacks
	glutDisplayFunc(displayGL);
	glutKeyboardFunc(keyboardGL);
	setIdle();

	// Initialize necessary OpenGL extensions
	glewInit();
	GLboolean bGLEW = glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object");
	ASSERT (bGLEW);

	// Initialize OpenGL
	glViewport(0, 0, outX, outY);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

	// Create OpenGL Buffer
	glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, sizeOutput, 0, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	// Create a context
	context = cl::Context(CL_DEVICE_TYPE_GPU);

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
	queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "D:/INFOTECH/2nd Semester/High Performance Programming with graphic cards/Exercise/Opencl-ex1/Opencl-ex1/src/OpenCLExercise6_VolumeRendering.cl");
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	// Allocate space for output data from CPU and GPU on the host
	h_outputCpu.resize (countOutput);
	h_outputGpu.resize (countOutput);

	// Allocate space for input and output data on the device
	//TODO 
	d_output = cl::Buffer(context, CL_MEM_READ_WRITE, sizeOutput);
	//TODO 
	d_invViewMatrix = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * 16);
	//TODO 
	d_input = cl::Image3D(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_R, CL_FLOAT), countX, countY, countZ);

	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
	memset(h_outputCpu.data(), 255, sizeOutput);
	memset(h_outputGpu.data(), 255, sizeOutput);
	//TODO GPU

	// Create a kernel object
	//TODO

	// Copy input data to device
	//TODO

	glutMainLoop ();

	return 0;
}

//////////////////////////////////////////////////////////////////////////////
// Render function
//////////////////////////////////////////////////////////////////////////////
void render () {
	float s = std::sin(alpha);
	float c = std::cos(alpha);
	float dist = 1.0f * std::max(countX, std::max(countY, countZ));
	float invViewMatrix[16] = {
		c*1.0f, 0.0f, -s*1.0f, -s*dist + countX/2.0f,
		0.0f, 1.0f, 0.0f, countY/2.0f,
		s*1.0f, 0.0f, c*1.0f, c*dist + countZ/2.0f,
		0.0f, 0.0f, 0.0f, 1.0f
	};
	float tstep = 1;
	float brightness = 0.2f * tstep / std::max(countX, std::max(countY, countZ));

	// Do calculation on the host side
	if (runCpu) {
		renderHost(h_input, h_outputCpu, countX, countY, countZ, outX, outY, invViewMatrix, tstep, brightness);
	}

	cl::Event kernelExecution;
	if (runGpu) {
		// Copy invViewMatrix to GPU
		//TODO

		// Call the kernel
		//TODO

		// Copy output data back to host
		//TODO
	}

	// map the PBO to copy the data into it
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	// map the buffer object into client's memory
	float* ptr = (float*)glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB);
	ASSERT (ptr);
	memcpy(ptr, displayGpu ? h_outputGpu.data() : h_outputCpu.data(), sizeOutput);
	glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB);

	if (writeImages) {
		Core::writeImagePGM("output_volume_cpu.pgm", h_outputCpu, outX, outY);
		Core::writeImagePGM("output_volume_gpu.pgm", h_outputGpu, outX, outY);
	}

	// Print performance data
	//TODO

	if (runCpu && runGpu) {
		// Check whether results are correct
		float maxError = 1e-2;
		std::size_t errorCount = 0;
		for (size_t y = 0; y < outY; y++) {
			for (size_t x = 0; x < outX; x++) {
				size_t index = x + y * outX;
				// Allow small differences between CPU and GPU results (due to different rounding behavior)
				if (!(std::abs (h_outputCpu[index] - h_outputGpu[index]) <= maxError)) {
					if (errorCount < 15)
						std::cout << "Result for " << x << "," << y << " is incorrect: GPU value is " << h_outputGpu[index] << ", CPU value is " << h_outputCpu[index] << std::endl;
					else if (errorCount == 15)
						std::cout << "..." << std::endl;
					errorCount++;
				}
			}
		}
		if (errorCount != 0) {
			std::cout << "Found " << errorCount << " incorrect results" << std::endl;
		}
	}

	// draw image from PBO
	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);
	glRasterPos2i(0, 0);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glDrawPixels(outX, outY, GL_LUMINANCE, GL_FLOAT, 0);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	// flip backbuffer to screen
	glutSwapBuffers();

	std::cout << "Done" << std::endl;
}

//////////////////////////////////////////////////////////////////////////////
// OpenGL callbacks
//////////////////////////////////////////////////////////////////////////////
void displayGL() {
	render();
}

void idleGL() {
	alpha += 0.03;
	if (alpha >= 2 * M_PI)
		alpha -= 2 * M_PI;
	render();
}

void setIdle() {
	if (animate)
		glutIdleFunc(idleGL);
	else
		glutIdleFunc(NULL);
}

void keyboardGL(unsigned char key, int x, int y) {
	switch(key) {
	case 'Q': case 'q': case '\e':
		glutLeaveMainLoop();
		break;
	case ' ':
		animate = !animate;
		setIdle();
		break;
	default:
		break;
	}
}
