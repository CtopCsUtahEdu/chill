###               ###
### SVN variables ###
###               ###
SVN_USER=dhuth

###                       ###
### Notification          ###
### (not implemented yet) ###
NOTIFY_ON_FAILURE=False



### Derived variables from config ###
CHILLHOME?=$(STAGING_DIR)/chill
SVN_CHILL=svn+ssh://$(SVN_USER)@shell.cs.utah.edu/uusoc/facility/res/hallresearch/svn_repo/resRepo/projects/chill
SVN_CHILL_DEV=$(SVN_CHILL)/branches/cuda-chill-rose
SVN_CHILL_RELEASE=$(SVN_CHILL)/release
CHILL_DEV_SRC=$(STAGING_DIR)/chill-dev
CHILL_RELEASE_SRC=$(STAGING_DIR)/chill-release
OMEGAHOME?=$(STAGING_DIR)/omega
SVN_OMEGA=svn+ssh://$(SVN_USER)@shell.cs.utah.edu/uusoc/facility/res/hallresearch/svn_repo/resRepo/projects/omega
SVN_OMEGA_DEV=$(SVN_OMEGA)/branches/cuda-omega-rose
SVN_OMEGA_RELEASE=$(SVN_OMEGA)/release
OMEGA_DEV_SRC=$(STAGING_DIR)/omega-dev
OMEGA_RELEASE_SRC=$(STAGING_DIR)/omega-release


### Staging ###
STAGING_DIR=$(shell pwd)/.staging
STAGING_DIR_BIN=$(STAGING_DIR)/bin
STAGING_DIR_WD=$(STAGING_DIR)/wd


### Local ###
UNIT_TEST_DIR=$(shell pwd)/unit-tests/
CHILL_DEV_TESTCASE_DIR=$(shell pwd)/test-cases/chill
CHILL_DEV_TESTCASES_SCRIPT=$(shell find $(CHILL_DEV_TESTCASE_DIR) -name "*.script")
CHILL_DEV_TESTCASES_STDOUT=$(patsubst %.script,%.stdout,$(CHILL_DEV_TESTCASES_SCRIPT))

### Python environment variables ###
PYTHON_2:=$(shell which python)
PYTHON_3:=$(shell which python3)

ifneq ($(PYTHON_3),)
PYTHON_3_VERSION=$(shell $(PYTHON_3) -c "import sysconfig; print(sysconfig.get_config_var('VERSION'))")
endif
PYTHON_2_VERSION=$(shell $(PYTHON_2) -c "import sysconfig; print sysconfig.get_config_var('VERSION')")
PYTHON_VERSION=$(firstword $(PYTHON_3_VERSION) $(PYTHON_2_VERSION))
PYTHON=$(shell which python$(PYTHON_VERSION))
### ---------------------------- ###


EXPORT=export CHILL_DEV_SRC=$(CHILL_DEV_SRC); \
       export CHILL_RELEASE_SRC=$(CHILL_RELEASE_SRC); \
       export OMEGA_DEV_SRC=$(OMEGA_DEV_SRC); \
       export OMEGA_RELEASE_SRC=$(OMEGA_RELEASE_SRC); \
       export STAGING_DIR_BIN=$(STAGING_DIR_BIN); \
       export STAGING_DIR_WD=$(STAGING_DIR_WD);

### deump environment ###
# define quiet to shut this part up #
ifndef quiet
$(info notify on failure?          $(NOTIFY_ON_FAILURE))
$(info staging directory           $(STAGING_DIR))
$(info binary directory            $(STAGING_DIR_BIN))
$(info working directory           $(STAGING_DIR_WD))
$(info omega home                  $(OMEGAHOME))
$(info chill home                  $(CHILLHOME))
$(info chill svn dev repo          $(SVN_CHILL_DEV))
$(info chill svn release repo      $(SVN_CHILL_RELEASE))
$(info chill dev src               $(CHILL_DEV_SRC))
$(info chill release src           $(CHILL_RELEASE_SRC))
$(info omega svn dev repo          $(SVN_OMEGA_DEV))
$(info omega svn release repo      $(SVN_OMEGA_RELEASE))
$(info omega dev src               $(OMEGA_DEV_SRC))
$(info omega release src           $(OMEGA_RELEASE_SRC))
$(info python                      $(PYTHON))
$(info unit tests                  $(UNIT_TEST_DIR))
#$(info chill-dev test cases        $(CHILL_DEV_TESTCASES_SCRIPT))
#$(info chill-dev test case stdouts $(CHILL_DEV_TESTCASES_STDOUT))
endif
### ----------------- ###

DIRTY_EXTS=pyc o log pickle
DIRTY_FILES=$(foreach de,$(DIRTY_EXTS),$(shell find . -name "*.$(de)"))
DIRTY_DIRS=$(shell find . -name '__pycache__' -and -type d) $(STAGING_DIR) pylang coverage_report

CORE_TESTS:=_extract util gcov _cpp_validate_env cpp_validate test __main__
OMEGA_TESTS:=omega
CHILL_TESTS:=chill

CORE_TESTS:=$(addsuffix .py,$(addprefix unit-tests/test_,$(CORE_TESTS)))
OMEGA_TESTS:=$(addsuffix .py,$(addprefix unit-tests/test_,$(OMEGA_TESTS)))
CHILL_TESTS:=$(addsuffix .py,$(addprefix unit-tests/test_,$(CHILL_TESTS)))

### The all target ###
.PHONY: all
all:
	$(MAKE) clean quiet=1
	$(MAKE) install quiet=1


### This will install the chill_test module ###
.PHONY: install
install: pylang
	$(PYTHON) makeparser.py
	#TODO: maybe run a setup or something



### This will uninstall teh chill_test module ###
.PHONY: uninstall
uninstall:
	#TODO: can python modules be uninstalled?



### Simply removes all files listed in DIRTY_FILES ###
.PHONY: clean
clean:
	rm -rf $(DIRTY_FILES)
	rm -rf $(DIRTY_DIRS)


pylang:
	git clone https://github.com/dhuth/pylang.git pylang-tmp
	$(PYTHON) pylang-tmp/make_grammar_parsers.py
	cp -r pylang-tmp/pylang pylang
	rm -rf pylang-tmp

### Test the test harness ###
.PHONY: test
test: $(STAGING_DIR_BIN) $(OMEGA_DEV_SRC) $(OMEGA_RELEASE_SRC) $(CHILL_DEV_SRC) $(CHILL_RELEASE_SRC)
	@echo "-----------------------------------------------------------"
	@echo "Note: This target tests the test suite it's self, not chill"
	@echo "To test chill, run python -m testchill ..."
	@echo "-----------------------------------------------------------"
	- $(EXPORT) $(PYTHON) -m unittest $(OMEGA_TESTS) $(CORE_TESTS) $(CHILL_TESTS)
	@ rm -rf $(STAGING_DIR)


.PHONY: test-chill
test-chill: $(STAGING_DIR_BIN) $(OMEGA_DEV_SRC) $(OMEGA_RELEASE_SRC) $(CHILL_DEV_SRC) $(CHILL_RELEASE_SRC)
	- $(EXPORT) $(PYTHON) -m unittest $(OMEGA_TESTS) $(CHILL_TESTS)
	@ rm -rf $(STAGING_DIR)


.PHONY: test-omega
test-omega: $(STAGING_DIR_BIN) $(OMEGA_DEV_SRC) $(OMEGA_RELEASE_SRC)
	- $(EXPORT) $(PYTHON) -m unittest $(OMEGA_TESTS)
	@ rm -rf $(STAGING_DIR)


.PHONY: test-core
test-core: $(STAGING_DIR_BIN) $(OMEGA_DEV_SRC) $(OMEGA_RELEASE_SRC) $(CHILL_DEV_SRC) $(CHILL_RELEASE_SRC) make-omega
	- $(EXPORT) $(PYTHON) -m unittest $(CORE_TESTS)
	@ rm -rf $(STAGING_DIR)


.PHONY:
test-core-%: $(STAGING_DIR_BIN)
	- $(EXPORT) $(PYTHON) -m unittest unit-tests/test_$*.py


.PHONY: test-debug
debug:
	@### NOTHING ###


### benchmarking (don't use if your're not me) ###
$(CHILL_DEV_TESTCASES_STDOUT): %.stdout: %.script
	$(EXPORT) cd $(STAGING_DIR_WD); $(STAGING_DIR_BIN)/chill $< > $@


.PHONY: benchmark-dev
benchmark-dev: test-chill $(CHILL_DEV_TESTCASES_STDOUT)
	# do nothing


### checking out and making directories ###
$(STAGING_DIR_BIN):
	mkdir -p $(STAGING_DIR_BIN)
	mkdir -p $(STAGING_DIR_WD)

$(CHILL_DEV_SRC): $(OMEGA_DEV_SRC) $(STAGING_DIR_BIN)
	svn export $(SVN_CHILL_DEV) $(CHILL_DEV_SRC)

$(CHILL_RELEASE_SRC): $(OMEGA_RELEASE_SRC) $(STAGIN_DIR_BIN)
	svn export $(SVN_CHILL_RELEASE) $(CHILL_RELEASE_SRC)

$(OMEGA_DEV_SRC): $(STAGING_DIR_BIN)
	svn export $(SVN_OMEGA_DEV) $(OMEGA_DEV_SRC)
	#cd $(OMEGA_DEV_SRC); $(MAKE) depend
	#cd $(OMEGA_DEV_SRC); $(MAKE)

$(OMEGA_RELEASE_SRC): $(STAGING_DIR_BIN)
	svn export $(SVN_OMEGA_RELEASE) $(OMEGA_RELEASE_SRC)
	#cd $(OMEGA_RELEASE_SRC); $(MAKE) depend
	#cd $(OMEGA_RELEASE_SRC): $(MAKE)

.PHONY: make-omega
make-omega:
	cd $(OMEGA_DEV_SRC); $(MAKE) depend
	cd $(OMEGA_DEV_SRC); $(MAKE)
	cd $(OMEGA_RELEASE_SRC); $(MAKE) depend
	cd $(OMEGA_RELEASE_SRC); $(MAKE)

#$(STAGING_DIR):
#	mkdir -p $(STAGING_DIR)

