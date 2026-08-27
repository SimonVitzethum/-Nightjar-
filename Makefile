.PHONY: all glm deepseek-v4 portable test check cuda-test clean install uninstall \
        nightjar nightjar-test nightjar-serve nightjar-stop nightjar-run nightjar-clean \
        nightjar-gates nightjar-bench nightjar-tools nightjar-uitest

# The Colibri engines and Nightjar (qwen35/qwen35moe) both live in c/ and both want targets
# called all, test and clean, so Nightjar's are namespaced. See c/Makefile.nightjar.
all glm deepseek-v4 portable test check cuda-test clean install uninstall \
nightjar nightjar-test nightjar-serve nightjar-stop nightjar-run nightjar-clean \
nightjar-gates nightjar-bench nightjar-tools nightjar-uitest:
	$(MAKE) -C c $@
