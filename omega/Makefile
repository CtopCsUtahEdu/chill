# DON'T EDIT -- put changes in Makefile.config.

BASEDIR = ./
include $(BASEDIR)/Makefile.config

all:
	cd omega_lib/obj; $(MAKE)
ifeq ($(BUILD_CODEGEN), true)
	cd code_gen/obj; $(MAKE)
endif
	cd omega_calc/obj; $(MAKE)

depend:
	cd omega_lib/obj; $(MAKE) depend
ifeq ($(BUILD_CODEGEN), true)
	cd code_gen/obj; $(MAKE) depend
endif
	cd omega_calc/obj; $(MAKE) depend

clean:
	cd omega_lib/obj; $(MAKE) clean
	cd code_gen/obj; $(MAKE) clean
	cd omega_calc/obj; $(MAKE) clean

veryclean:
	cd omega_lib/obj; $(MAKE) veryclean
	cd code_gen/obj; $(MAKE) veryclean
	cd omega_calc/obj; $(MAKE) veryclean

install:
	cp -rL bin lib include $(DEST_DIR)

