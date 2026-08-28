# ------------------------------------------------------------------------------
# Test / data maintenance helpers for epiclass
# ------------------------------------------------------------------------------

PYTEST ?= pytest
SACCER3_HDF5_DIR := ./fixtures/saccer3/hdf5/saccer3_2016-07
NUMBA_CACHE_DIR := ./.numba_cache

.PHONY: help test test-fast test-slow demo-test clean-saccer3 clean-numba-cache clean-all

help:
	@echo ""
	@echo "Available targets:"
	@echo "  test               Run full test suite"
	@echo "  test-fast          Run tests excluding slow/integration"
	@echo "  test-slow          Run only slow/integration tests"
	@echo "  demo-test          Run demo training/predict using saccer3."
	@echo "  clean-saccer3      Remove extracted saccer3 HDF5 files"
	@echo "  clean-numba-cache  Remove cached numba kernels (forces recompile)"
	@echo "  clean-all          Clean test data + pytest cache"
	@echo "----------------------------------------"
	@echo ""

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

test:
	$(PYTEST) -n auto -v

test-fast:
	$(PYTEST) -n auto -v -m "not slow"

test-slow:
	$(PYTEST) -n auto -v -m slow

test-last-failed:
	@pytest -v --lf --last-failed-no-failures none -p no:xdist; \
	code=$$?; \
	if [ $$code -eq 5 ]; then \
		echo "No last failed tests to run (exit 5 ignored)"; \
		exit 0; \
	else \
		exit $$code; \
	fi

demo-test:
	\pytest -vs ./mains/predict_test.py ./mains/general_training_test.py

# ------------------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------------------

clean-saccer3:
	@echo "Cleaning saccer3 extracted HDF5 files..."
	@if [ -d "$(SACCER3_HDF5_DIR)" ]; then \
		rm -rf "$(SACCER3_HDF5_DIR)"; \
		echo "Removed $(SACCER3_HDF5_DIR)"; \
	else \
		echo "No saccer3 extracted data found."; \
	fi

clean-numba-cache:
	@echo "Cleaning numba kernel cache..."
	@if [ -d "$(NUMBA_CACHE_DIR)" ]; then \
		rm -rf "$(NUMBA_CACHE_DIR)"; \
		echo "Removed $(NUMBA_CACHE_DIR)"; \
	else \
		echo "No numba cache found."; \
	fi

clean-all: clean-saccer3 clean-numba-cache
	@echo "Cleaning pytest cache..."
	@rm -rf .pytest_cache
	@echo "All clean."
