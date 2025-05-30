CXX=mpicxx	#g++
LD=${CXX}
CXXFLAGS+=-Wall -Wextra -pedantic -std=c++11 -I${HDF5_ROOT}/include # -Werror
LDFLAGS+=-lm $(CXXFLAGS) -L${HDF5_ROOT}/lib -lhdf5 -lhdf5_cpp


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

SRCS = $(wildcard $(SRC_DIR)/*.cc)
OBJS = $(patsubst $(SRC_DIR)/%.cc,$(BUILD_DIR)/%.o,$(SRCS))
TARGET = $(BUILD_DIR)/swe

all: $(TARGET)

$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(LD) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cc | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean: clean_otput
	rm -rf $(BUILD_DIR)
	
clean_otput:
	rm -f perf.data
	rm -f slurm-*.out
	rm -f gmon.out
