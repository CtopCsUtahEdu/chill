AC_INIT([chill],[0.2.1])
AC_CONFIG_AUX_DIR([.])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])

#AC_SUBST([prefix],["$(pwd)/bin"])

### Use the C++ compiler for linking (C is default) ###
AC_SUBST([CCLD],["$""(CXX)"])
AC_SUBST([CHILL_BUILD_VERSION],["\\\"0.2.1\\\""])

AM_PROG_LEX
AC_PROG_CC
AC_PROG_CXX
AC_PROG_RANLIB
AC_PROG_YACC

AM_PATH_PYTHON([2.7],[
    AC_SUBST([PYTHON_INCDIR],
        [`${PYTHON} -c "from distutils import sysconfig; print(sysconfig.get_python_inc())"`])
    AC_SUBST([PYTHON_LIBDIR],
        [`${PYTHON} -c "from distutils import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"`])
    AC_SUBST([with_python],["yes"])],[
    AC_SUBST([with_python],["no"])])


AC_ARG_ENABLE([cuda],[
    AS_HELP_STRING([--enable-cuda],[builds cudachill])],[
    AS_IF([test "x$enableval" != xno],[
        AC_SUBST([enable_cuda],[yes])],[])])
AM_CONDITIONAL([CUDACHILL_OPT],[test "x$enable_cuda" == xyes])


AC_ARG_ENABLE([coverage],[
    AS_HELP_STRING([--enable-coverage],[enables coverage testing])],[
    
    AS_IF(["x$enableval" != xno],[
        AC_SUBST([enable_coverage],[yes])
        AC_SUBST([CFLAGS],["-g -fprofile-arcs -ftest-coverage"])
        AC_SUBST([CPPFLAGS],["-g -fprofile-arcs -ftest-coverage"])],[])])


AC_ARG_WITH([lua],[
    AS_HELP_STRING([--with-lua],[set lua home])],[
    AC_SUBST([LUAHOME], [$withval])],[
    AC_SUBST([LUAHOME], ["${LUAHOME}"])])

AC_ARG_WITH([rose],[
    AS_HELP_STRING([--with-rose],[set rose home])],[
    AC_SUBST([ROSEHOME], [$withval])
    AC_DEFINE([BUILD_ROSE],[],[Use ROSE])],[
    AC_SUBST([ROSEHOME], ["${ROSEHOME}"])
    AC_DEFINE([BUILD_ROSE],[],[Use ROSE])])

AC_ARG_WITH([boost],[
    AS_HELP_STRING([--with-boost],[set boost home])],[
    AC_SUBST([BOOSTHOME], [$withval])],[
    AC_SUBST([BOOSTHOME], ["${BOOSTHOME}"])])

AC_ARG_WITH([omega],[
    AS_HELP_STRING([--with-omega],[set omega home])],[
    AC_SUBST([OMEGAHOME], [$withval])],[
    AC_SUBST([OMEGAHOME], ["$(pwd)/omega"])])


AC_ARG_WITH([interface],[
    AS_HELP_STRING([--with-interface],[select interface language])],[
    AC_SUBST([interface_lang],[$withval])],[
    AC_SUBST([interface_lang],[default])])

AS_IF([test "x$interface_lang" == xdefault],[
    AS_IF([test "x$enable_cuda" != xno],[
        AC_SUBST([use_python],[no])
        AC_SUBST([use_lua],[yes])
        ],[
        AC_SUBST([use_python],[no])
        AC_SUBST([use_lua],[no])])],[])

AS_IF([test "x$interface_lang" == xlua],[
    AC_SUBST([use_lua],[yes])],[])
AS_IF([test "x$interface_lang" == xpython],[
    AC_SUBST([use_python],[yes])],[])

AM_CONDITIONAL([PYTHON_OPT],[test "x$use_python" == xyes])
AM_CONDITIONAL([LUA_OPT],[test "x$use_lua" == xyes])

AC_CONFIG_HEADERS([include/config.h])
AC_CONFIG_FILES([Makefile])
AC_CONFIG_SUBDIRS([omega])
AC_OUTPUT
