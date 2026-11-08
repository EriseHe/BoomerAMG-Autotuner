ROOT := $(abspath $(CURDIR)/../..)
INTERFACE_DIR := $(ROOT)/hypre/interfaces
SHARED_EXT := $(if $(filter Darwin,$(shell uname -s)),dylib,so)
TARGET := $(INTERFACE_DIR)/libamg_runtime.$(SHARED_EXT)
MPICC ?= mpicc

.PHONY: all clean print-vars

all:
	$(MAKE) -C $(INTERFACE_DIR) libamg_runtime.$(SHARED_EXT) MPICC="$(MPICC)"

clean:
	$(MAKE) -C $(INTERFACE_DIR) clean

print-vars:
	@echo "ROOT=$(ROOT)"
	@echo "INTERFACE_DIR=$(INTERFACE_DIR)"
	@echo "TARGET=$(TARGET)"
	@echo "MPICC=$(MPICC)"
