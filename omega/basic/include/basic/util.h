#if ! defined Already_Included_Util
#define Already_Included_Util

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string>
#include <sstream>
#include <stdexcept>

namespace omega {
  
#define LONG_LONG_COEF 1

#if LONG_LONG_COEF
#if defined BOGUS_LONG_DOUBLE_COEF
typedef long double coef_t;  // type of coefficients
#define coef_fmt  "%llf"
#define posInfinity (1e+24)
#define negInfinity (-1e+24)
#else
#ifdef WIN32
typedef _int64 coef_t;  // type of coefficients
#else
typedef long long coef_t;
#endif
#define coef_fmt  "%lld"
#define posInfinity (0x7ffffffffffffffLL)
#define negInfinity (-0x7ffffffffffffffLL)
#endif
#else
typedef int coef_t;  // type of coefficients
#define coef_fmt  "%d"
#define posInfinity (0x7ffffff)
#define negInfinity (-0x7ffffff)
#endif


template<typename T> inline const T& max(const T &x, const T &y) {
	if (x >= y) return x; else return y;
}


template<typename T> inline const T& max(const T &x, const T &y, const T &z) {
  return max(x, max(y, z));
}

template<typename T> inline const T& min(const T &x, const T &y) {
	if (x <= y) return x; else return y;
}

template<typename T> inline const T& min(const T &x, const T &y, const T &z) {
  return min(x, min(y, z));
}

template<class T> inline void set_max(T &m, const T &x) {
	if (m < x) m = x;
}

template<class T> inline void set_min(T &m, const T &x) {
	if (m > x) m = x;
}

/* template<class T> inline void swap(T &i, T &j) { */
/*   T tmp; */
/*   tmp = i; */
/*   i = j; */
/*   j = tmp; */
/* } */

/* template<class T> inline T copy(const T &t) { return t; } */


/* inline coef_t check_pos_mul(coef_t x, coef_t y) { */
/*   if (y >= 48051280 && y < posInfinity) */
/*     debug_fprintf(stderr, "%d %d\n", x, y); */
/* /\* #if !defined NDEBUG *\/ */
/* /\*   if (x != 0) *\/ */
/* /\*     assert(((MAXINT)/4) / x > y); *\/ */
/* /\* #elif defined STILL_CHECK_MULT *\/ */
/* /\*   if (x != 0 && !(((MAXINT)/4) / x > y)) { *\/ */
/* /\*     assert(0&&"Integer overflow during multiplication (util.h)"); *\/ */
/* /\*   } *\/ */
/* /\* #endif *\/ */
/* #if !defined NDEBUG */
/*   if (x != 0 && y != 0) */
/*     assert(x*y > 0); */
/* #elif defined STILL_CHECK_MULT */
/*   if (x != 0 && y != 0 && x*y < 0) */
/*     assert(0&&"Integer overflow during multiplication (util.h)"); */
/* #endif */
/*   return x * y; */
/* } */


/* inline int */
/* check_pos_mul(int x, int y) { */
/* #if !defined NDEBUG */
/*   if (x != 0) */
/*     assert(((posInfinity)/4) / x > y); */
/* #elif defined STILL_CHECK_MULT */
/*   if (x != 0 && !(((posInfinity)/4) / x > y)) { */
/*     assert(0&&"Integer overflow during multiplication (util.h)"); */
/*   } */
/* #endif */
/*   return x * y; */
/* } */

/* inline LONGLONG */
/* check_pos_mul(LONGLONG x, LONGLONG y) { */
/* #if !defined NDEBUG */
/*   if (x != 0) */
/*     assert(((posInfinity)/4) / x > y); */
/* #elif defined STILL_CHECK_MULT */
/*   if (x != 0 && !(((posInfinity)/4) / x > y)) { */
/*     assert(0&&"Integer overflow during multiplication (util.h)"); */
/*   } */
/* #endif */
/*   return x * y; */
/* } */

/* inline LONGLONG abs(LONGLONG c) { return (c>=0?c:(-c)); }  */

template<typename T> inline T check_mul(const T &x, const T &y) {
#if defined NDEBUG && ! defined STILL_CHECK_MULT
  return x*y;
#else
  if (x == 0 || y == 0)
    return 0;
 
  T z = x*y;
  int sign_x = (x>0)?1:-1;
  int sign_y = (y>0)?1:-1;
  int sign_z = (z>0)?1:-1;

  if (sign_x * sign_y != sign_z)
    throw std::overflow_error("coefficient multiply overflow");

  return z;

  /* if (x > 0) { */
  /*   if (y > 0) { */
  /*     assert(x*y > 0); */
  /*   } */
  /*   else */
  /*     assert(x*y < 0); */
  /* } */
  /* else { */
  /*   if (y > 0) */
  /*     assert(x*y < 0); */
  /*   else */
  /*     assert(x*y > 0); */
  /* } */
  /* return x*y; */
#endif
}

template<typename T> inline T abs(const T &v) {
  return (v >= static_cast<T>(0))?v:-v;
}

template<class T> inline T int_div(const T &a, const T &b) {
  T result;
	assert(b > 0);
	if (a>0) result = a/b;
	else     result = -((-a+b-1)/b);
	return result;
}

template<class T> inline T int_mod(const T &a, const T &b) {
	return a-b*int_div(a,b);
}

template<class T> inline T int_mod_hat(const T &a, const T &b) {
	T r;
	assert(b > 0);
	r = a-b*int_div(a,b);
	if (r > -(r-b)) r -= b;
	return r;
}

template<typename T> inline T gcd(T b, T a) {/* First argument is non-negative */
  assert(a >= 0);
  assert(b >= 0);
  if (b == 1)
    return (1);
  while (b != 0) {
    T t = b;
    b = a % b;
    a = t;
	}
  return (a);
}

template<typename T> inline T lcm(T b, T a) { /* First argument is non-negative */
  assert(a >= 0);
  assert(b >= 0);
  return check_mul(a/gcd(a,b), b);
}

template<typename T> T square_root(const T &n, T precision = 1) {
  T guess = 1;

  while (true) {
    T next_guess = 0.5*(guess+n/guess);
    if (abs(next_guess-guess) <= precision)
      return next_guess;
    else
      guess = next_guess;
  }
}

template<typename T> T factor(const T &n) {
  assert(n >= 0);
  if (n == 1) return 1;
  
  static int prime[30] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113};

  if (n <= 113*113) {
    for (int i = 0; i < 30; i++)
      if (n % static_cast<T>(prime[i]) == 0)
        return static_cast<T>(prime[i]);

    return n;
  }
  
  T i = 1;
  T k = 2;
  T x = static_cast<T>(rand())%n;
  T y = x;
  while(i < square_root<float>(n, 1)) {
    i++;
    x = (x*x-1) % n;
    T d = gcd(abs(y-x), n);
    if(d != 1 && d != n)
      return factor(d);
    if(i == k) {
      y = x;
      k *= 2;
    }
  }
  return n;
}

/* #define implies(A,B) (A==(A&B)) */

template<typename T> std::string to_string(const T &t) {
  std::ostringstream ss;
  ss << t;
  return ss.str();
}

template<typename T> T from_string(const std::string &s) {
  std::istringstream ss(s);
  ss.exceptions(std::ios::failbit);
  T t;
  ss >> t;
  return t;
}

} // namespace

#endif
