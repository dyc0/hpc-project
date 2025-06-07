CXX=nvcc
LD=${CXX}
CXXFLAGS+=-Xcompiler="-Wall -Wextra" -std=c++11 -I${HDF5_ROOT}/include # -Werror
LDFLAGS+=-lm $(CXXFLAGS) -L${HDF5_ROOT}/lib -lhdf5


BUILD ?= release
ifeq ($(BUILD),debug)
    CXXFLAGS += -g -O0 -DDEBUG
endif
ifeq ($(BUILD),profile)
    CXXFLAGS += -pg -O3
    LDFLAGS  += -pg -O3
endif
ifeq ($(BUILD),release)
	CXXFLAGS += -O3
	LDFLAGS  += -O3
endif


SRC_DIR = .
BUILD_DIR = build

CC_SRCS = $(wildcard $(SRC_DIR)/*.cc)
CU_SRCS = $(wildcard $(SRC_DIR)/*.cu)
SRCS = $(CC_SRCS) $(CU_SRCS)
CC_OBJS = $(patsubst $(SRC_DIR)/%.cc,$(BUILD_DIR)/%.o,$(CC_SRCS))
CU_OBJS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SRCS))
OBJS = $(CC_OBJS) $(CU_OBJS)
TARGET = $(BUILD_DIR)/swe

all: $(TARGET)

$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(LD) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cc | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean: clean_output
	rm -rf $(BUILD_DIR)
	
clean_output:
	rm -f perf.data
	rm -f slurm-*.out
	rm -f gmon.out
	rm -rf parallel_tests/*
