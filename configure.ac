AC_INIT([chill],[`cat CHILL_VERSION`])
AC_CONFIG_AUX_DIR([.])

AM_INIT_AUTOMAKE([-Wall -Werror foreign subdir-objects])

### Use the C++ compiler for linking (C is default) ###
AC_SUBST([CCLD],["$""(CXX)"])
AC_SUBST([CHILL_BUILD_VERSION],["\\\"`cat CHILL_VERSION`\\\""])

AC_DEFINE([CHILL_BUILD_VERSION],["`cat CHILL_VERSION`"],[CHiLL Version])

AM_PROG_AR
AM_PROG_LEX
AC_PROG_CC
AC_PROG_CXX
AC_PROG_RANLIB
AC_PROG_YACC

AM_PATH_PYTHON([2.7],[
    AC_SUBST([PYTHON_INCDIR],
        [`python2.7 -c "from distutils import sysconfig; print(sysconfig.get_python_inc())"`])
    AC_SUBST([PYTHON_LIBDIR],
        [`python2.7 -c "from distutils import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"`])
    AC_SUBST([python_builds],[yes])],[
    AC_SUBST([python_builds],[no])])


AC_ARG_ENABLE([coverage],
   [AS_HELP_STRING([--enable-coverage],[enables coverage testing])],
   [AS_IF(["x$enableval" != xno],
       [AC_SUBST([enable_coverage],[yes])
        AC_SUBST([CFLAGS],["-g -fprofile-arcs -ftest-coverage"])
        AC_SUBST([CXXFLAGS],["-g -fprofile-arcs -ftest-coverage"])],[])])

AC_ARG_ENABLE([debug],
   [AS_HELP_STRING([--enable-debug],[Enable debug messages])],
   [AS_IF(["x$enableval" != xno],
       [AC_SUBST([enable_debug],[yes])])])

AM_CONDITIONAL([DEBUG_OPT],[test "x$enable_debug" != xyes])
AS_IF([test "x$enable_debug" == xyes],
    [AC_DEFINE([DEBUGCHILL], [], [Enable CHiLL debug messages])
     AC_SUBST([CFLAGS],["-g2 -O0"])
     AC_SUBST([CXXFLAGS],["-g2 -O0"])],
    [AC_DEFINE([NO_DEBUGCHILL], [], [Disable CHiLL debug messages])])

AC_ARG_WITH([rose],
    [AS_HELP_STRING([--with-rose],[set rose home])],
    [AC_SUBST([ROSEHOME], [$withval])
     AC_DEFINE([BUILD_ROSE],[],[Use ROSE])],
    [AC_SUBST([ROSEHOME], ["${ROSEHOME}"])
     AC_DEFINE([BUILD_ROSE],[],[Use ROSE])])

AC_ARG_WITH([boost],[
    AS_HELP_STRING([--with-boost],[set boost home])
    ],[
    AC_SUBST([BOOSTHOME], [$withval])
    ],[
    AC_SUBST([BOOSTHOME], ["${BOOSTHOME}"])])

AC_ARG_WITH([iegen],[
    AS_HELP_STRING([--with-iegen],[set path to iegenlib])
    ],[
    AC_SUBST([IEGENHOME], [$withval])
    ],[
    AC_SUBST([IEGENHOME], ["${IEGENHOME}"])])

AC_ARG_WITH([omega],[
    AS_HELP_STRING([--with-omega],[set omega home])],[
    AC_SUBST([OMEGAHOME], [$withval])],[
    AC_SUBST([OMEGAHOME], ["$(pwd)/omega"])])

AC_ARG_WITH([python],
   [AS_HELP_STRING([--with-python],[set python as the interface languge])],
   [AC_SUBST([interface_lang],[python])],
   [AC_SUBST([interface_lang],[default])])

AC_ARG_WITH([lua],
   [AS_HELP_STRING([--with-lua],[set lua as the interface langauge.])],
   [AC_SUBST([LUAHOME], ["${LUAHOME}"])
    AC_SUBST([interface_lang], [lua])],
   [AC_SUBST([LUAHOME], ["${LUAHOME}"])])

AC_ARG_WITH([script],
   [AS_HELP_STRING([--with-script],[use the legacy script interface language.])],
   [AC_SUBST([interface_lang], [script])],
   [])

#AC_ARG_WITH([interface],[
#    AS_HELP_STRING([--with-interface],[select interface language])],[
#    AC_SUBST([interface_lang],[$withval])],[
#    AC_SUBST([interface_lang],[default])])

AS_IF([test "x$interface_lang" == xdefault],
   [AS_IF([test "x$enable_cuda" == xyes],
       [AC_SUBST([use_python],[yes])
        AC_SUBST([use_lua],[no])
        AC_SUBST([interface_lang],[python])],
       [AC_SUBST([use_python],[yes])
        AC_SUBST([use_lua],[no])
        AC_SUBST([interface_lang],[python])])],
       [])

AS_IF([test "x$interface_lang" == xlua],
   [AC_SUBST([use_python],[no])
    AC_SUBST([use_lua],[yes])
    AC_SUBST([use_script],[no])],[])
AS_IF([test "x$interface_lang" == xpython],
   [AC_SUBST([use_python],[yes])
    AC_SUBST([use_lua],[no])
    AC_SUBST([use_script],[no])],[])
AS_IF([test "x$interface_lang" == xscript],
   [AC_SUBST([use_python],[no])
    AC_SUBST([use_lue],[no])
    AC_SUBST([use_script],[yes])],[])

AM_CONDITIONAL([PYTHON_OPT],[test "x$use_python" = xyes])
AM_CONDITIONAL([LUA_OPT],[test "x$use_lua" = xyes])
AM_CONDITIONAL([SCRIPT_OPT],[test "x$use_script" = xyes])

AC_CONFIG_HEADERS([include/config.h])
AC_CONFIG_FILES([Makefile])
AC_CONFIG_SUBDIRS([omega])
AC_OUTPUT
