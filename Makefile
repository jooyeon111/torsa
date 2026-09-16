# ============================================================================
# Makefile — build and run the TORSA cycle-level simulator.
#
#   make                 # build `sim` (all .cpp except main_optimizer.cpp)
#   make run             # run sim on the default array/layer config
#   make run-burst       # run sim on the default burst (ToroidalDual) config
#   make dramsim3        # sim -> DRAM trace -> DRAMsim3 -> merged results
#   make optimizer       # build the cd/SRAM optimizer binary
#   make optimize        # build and run the optimizer
#   make clean           # remove objects and binaries
#
# Any variable below can be overridden on the command line, e.g.
#   make run ARRAY_CFG=array_config/A2_ReuseB_64x64x1x1x1.cfg DRAM_MODEL=analytic
# ============================================================================

# ===== Toolchain =====
CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -g

# ===== Files =====
TARGET   := sim
# main_optimizer.cpp has its own main(), so it is excluded from the sim build
# and compiled separately into $(OPT_BIN).
OPT_SRC  := main_optimizer.cpp
OPT_OBJ  := $(OPT_SRC:.cpp=.o)
SRCS     := $(filter-out $(OPT_SRC),$(wildcard *.cpp))
OBJS     := $(SRCS:.cpp=.o)
HEADERS  := $(wildcard *.hpp)
OPT_BIN  := optimizer

# ===== Run arguments =====
# SRAM bandwidth is never specified: sim derives it from the array config and
# automatically picks the matching (proper) row from the CACTI library in
# SRAM_DIR. Inputs can be overridden, e.g.
#   make run ARRAY_CFG=array_config/A2_ReuseB_64x64x1x1x1.cfg
# SRAM_DIR is the CACTI results root. Both run and run-burst, and every role
# (input/weight/output), select their SRAM from here.
SRAM_DIR     := cacti_results
CONFIG_DIR   := array_config
LAYER_DIR    := layer/huggyllama__llama-7b/prefill
ARRAY_CFG       := $(CONFIG_DIR)/A3_ReuseB_64x64x1x1x1.cfg
ARRAY_CFG_BURST := $(CONFIG_DIR)/test_a3burst_4x4.cfg
LAYER_CFG    := $(LAYER_DIR)/00_q_proj.cfg

# SRAM capacities (KB). These are still positional arguments of sim but are
# currently IGNORED: every CACTI source is fixed at 256KB, so the values are
# kept only for command-line compatibility.
# In burst mode the inputBurst SRAM is the input SRAM used with wide access,
# so there is no separate BURST_CAP argument.
INPUT_CAP    := 256
WEIGHT_CAP   := 512
OUTPUT_CAP   := 1024

# sim argument order (same with or without burst):
#   arrayConfig layer inputCapKB weightCapKB outputCapKB [sramDir]
#   - Burst vs. normal is decided by the arrayConfig `kind` (no extra argument).
#   - In burst mode the inputBurst SRAM reuses inputCapKB.
#   - The accumulator was removed (accumulation is an output-SRAM RMW), so there
#     is no accumulatorConfig argument.
RUN_ARGS       := $(ARRAY_CFG)       $(LAYER_CFG) $(INPUT_CAP) $(WEIGHT_CAP) $(OUTPUT_CAP) $(SRAM_DIR)
RUN_BURST_ARGS := $(ARRAY_CFG_BURST) $(LAYER_CFG) $(INPUT_CAP) $(WEIGHT_CAP) $(OUTPUT_CAP) $(SRAM_DIR)

# ===== DRAM energy backend =====
# Selects how `sim` computes DRAM energy (read by main.cpp via the DRAM_MODEL env var).
#   dramsim3 (default) : trace -> DRAMsim3 + merge_results.py; DRAM excluded from sim totals.
#   analytic           : constant per-access energy, self-contained (no trace/merge).
# Override on the command line:  make run DRAM_MODEL=analytic
DRAM_MODEL ?= dramsim3

# ===== Rules =====
.PHONY: all clean run run-burst optimize dramsim3 dramsim3-burst

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Every .o depends on every header, so editing any header rebuilds everything.
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Normal mode (arrayConfig kind = Grouped/Toroidal).
run: $(TARGET)
	DRAM_MODEL=$(DRAM_MODEL) ./$(TARGET) $(RUN_ARGS)

# Burst mode (arrayConfig kind = ToroidalDual). ToroidalDual only applies to
# GEMV layers (M == 1); sim falls back to Toroidal ReuseB for M > 1.
run-burst: $(TARGET)
	DRAM_MODEL=$(DRAM_MODEL) ./$(TARGET) $(RUN_BURST_ARGS)

# Full pipeline: sim -> DRAM trace -> DRAMsim3 -> merge_results.py.
# DRAM results are appended to sim's result file (results/sim_*.txt).
# run_pipeline.sh needs the trace, so it relies on the default DRAM_MODEL=dramsim3
# (DRAM_MODEL is not passed through here).
dramsim3: $(TARGET)
	./run_pipeline.sh $(RUN_ARGS)

dramsim3-burst: $(TARGET)
	./run_pipeline.sh $(RUN_BURST_ARGS)

# Optimizer: for one (array config, layer) it enumerates SRAM combinations and
# sweeps the contracting dimension (cd), then reports the best by OBJECTIVE.
# All other code is header-only, so it builds from the single .cpp.
# `make optimizer` -> build only; `make optimize` -> build, then run.
# OBJECTIVE: energy | cycles | edp | tops (TOPS/W/mm^2).
#   make optimize ARRAY_CFG=array_config/foo.cfg LAYER_CFG=layer/.../bar.cfg OBJECTIVE=energy
OBJECTIVE := tops
$(OPT_BIN): $(OPT_SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(OPT_SRC)

optimize: $(OPT_BIN)
	./$(OPT_BIN) $(ARRAY_CFG) $(LAYER_CFG) $(SRAM_DIR) $(OBJECTIVE)

clean:
	rm -f $(OBJS) $(OPT_OBJ) $(TARGET) $(OPT_BIN)