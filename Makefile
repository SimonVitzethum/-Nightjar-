.PHONY: all glm portable test check cuda-test clean serve stop gates run tools bench fwd batch mtp

# `serve`, `stop`, `gates`, `run` and friends belong to the qwen3.5 engine; the rest are
# the original GLM-5.2 targets. Both live in c/.
all glm portable test check cuda-test clean serve stop gates run tools bench fwd batch mtp:
	$(MAKE) -C c $@
