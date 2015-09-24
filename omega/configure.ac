### note: I have no idea what version of omega this is.
###       If it's not 0.1.0, feel free to change it
AC_INIT([omega],[0.1.0])
AC_CONFIG_AUX_DIR([.])
AM_INIT_AUTOMAKE([-Wall -Werror foreign subdir-objects])

### Use the C++ compiler for linking (C is default) ###
AC_SUBST([CCLD],["$""(CXX)"])
AC_DEFINE([OMEGA_VERSION],["0.1.0"],[Omega version])

AM_PROG_AR
AM_PROG_LEX
AC_PROG_CC
AC_PROG_CXX
AC_PROG_RANLIB
AC_PROG_YACC

AC_ARG_WITH([rose],[
    AS_HELP_STRING([--with-rose],[set rose home])
    ],[
    AC_SUBST([ROSEHOME], [$withval])
    AC_DEFINE([BUILD_ROSE],[],[Use Rose])
    ],[
    AC_SUBST([ROSEHOME], ["${ROSEHOME}"])])

AC_ARG_WITH([boost],[
    AS_HELP_STRING([--with-boost],[set boost home])],[
    AC_SUBST([BOOSTHOME], [$withval])],[
    AC_SUBST([BOOSTHOME], ["${BOOSTHOME}"])])

AC_CONFIG_HEADERS([include/config.h])
AC_CONFIG_FILES([
    omega_lib/obj/Makefile
    omega_calc/obj/Makefile
    code_gen/obj/Makefile])

AC_OUTPUT
    

