EXECUTABLE := gemm

CU_FILES   := lin.cu ../../../nvml_tools.cu

CU_DEPS    :=

CC_FILES   := 

all: $(EXECUTABLE) $(REFERENCE)

LOGS	   := logs

###########################################################

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart -lz -lnvidia-ml -lcudnn -lpthread
NVCC=nvcc
NVCCFLAGS= -gencode=arch=compute_86,code=\"sm_86,compute_86\" -std=c++17 -I/home/diogomatos/cudnn-frontend-1.5.0/include

OBJS= $(OBJDIR)/lin.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE) $(LOGS)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
