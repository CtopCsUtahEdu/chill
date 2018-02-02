/*
 * datagen.hpp
 *
 *  Created on: Jan 19, 2018
 *      Author: derick
 */

#ifndef SRC_CHECK_DATAGEN_HPP_
#define SRC_CHECK_DATAGEN_HPP_

#include <cstring>

#include <iostream>
#include <numeric>
#include <random>
#include <type_traits>

#define PASS            0
#define FAIL            1
#define ERROR           99
#define SKIP            77

/*
 * random number generation
 */

static std::random_device                      rand_dev;
static std::mt19937_64                         rand_gen(rand_dev());
static std::uniform_real_distribution<double>  rand_dist(-1, 1);

template<typename T>
static inline T rand();

template<> double rand<double>() { return         rand_dist(rand_gen); }
template<> float  rand<float>()  { return (float) rand_dist(rand_gen); }

template<typename T, size_t I>
inline static void fill_random(T (&arr)[I]) {
    for(size_t i = 0; i < I; i++) {
        arr[i] = rand<T>();
    }
}

template<typename T, size_t I, size_t J>
inline static void fill_random(T (&arr)[I][J]) {
    for(size_t i = 0; i < I; i++) {
        for(size_t j = 0; j < J; j++) {
            arr[i][j] = rand<T>();
        }
    }
}

template<typename T>
static T* __alloc_and_fill_random(size_t r) {
    auto arr = reinterpret_cast<T*>(malloc(r * sizeof(T*)));
    for(int i = 0; i < r; i++) {
        arr[i] = rand<T>();
    }
    return arr;
}

template<typename T>
static T** __alloc_and_fill_random(size_t r, size_t c) {
    auto arr = reinterpret_cast<T**>(malloc(c*sizeof(T*)));
    for(int i = 0; i < r; i++) {
        arr[i] = __alloc_and_fill_random<T>(c);
    }
    return arr;
}

template<typename TPtr>
inline static void alloc_and_fill_random(TPtr& arr, size_t r, size_t c) {
    typedef typename std::remove_pointer<
                typename std::remove_pointer<TPtr>::type>::type fp_t;
    arr = __alloc_and_fill_random<fp_t>(r, c);
}

/*
 * zero generator
 */

template<typename T>
inline static void setzero(T* addr) {
    memset((void*)addr, 0, sizeof(T));
}

template<> inline void setzero<float> (float*  addr) { *addr = 0; }
template<> inline void setzero<double>(double* addr) { *addr = 0; }

template<typename T, size_t I>
inline static void fill_zero(T (&arr)[I]) {
    for(size_t i = 0; i < I; i++) {
        setzero((T*)&arr[I]);
    }
}

template<typename T>
static T* __alloc_and_fill_zero(size_t r) {
    auto arr = reinterpret_cast<T*>(malloc(r * sizeof(T*)));
    for(int i = 0; i < r; i++) {
        setzero<T>(&arr[i]);
    }
    return arr;
}

template<typename T>
static T** __alloc_and_fill_zero(size_t r, size_t c) {
    auto arr = reinterpret_cast<T**>(malloc(c*sizeof(T*)));
    for(int i = 0; i < r; i++) {
        arr[i] = __alloc_and_fill_zero<T>(c);
    }
    return arr;
}

template<typename TPtr>
inline static void alloc_and_fill_zero(TPtr& arr, size_t r, size_t c) {
    typedef typename std::remove_pointer<
                    typename std::remove_pointer<TPtr>::type>::type fp_t;
    arr = __alloc_and_fill_zero<fp_t>(r, c);
}

/*
 * compare
 */

static inline float  compare(float  x, float  y) { return x - y; }
static inline double compare(double x, double y) { return x - y; }

template<typename T, size_t I>
static auto compute_error(T (&lhs)[I], T (&rhs)[I])
        -> decltype(compare(lhs[0], rhs[0])) {
    decltype(compare(lhs[0], rhs[0])) err = 0;
    for(size_t i = 0; i < I; i++) {
        err += compare(lhs[i], rhs[i]);
    }
    return err/I;
}

template<typename T>
static auto compute_error(T* lhs, T* rhs, size_t c)
        -> decltype(compare(*lhs, *rhs)) {
    decltype(compare(*lhs, *rhs)) err = 0;
    for(size_t i = 0; i < c; i++) {
        err += compare(lhs[i], rhs[i]);
    }
    return err/c;
}

template<typename T>
static auto compute_error(T** lhs, T** rhs, size_t r, size_t c)
        -> decltype(compare(**lhs, **rhs)) {
    decltype(compare(**lhs, **rhs)) err = 0;
    for(size_t i = 0; i < r; i++) {
        err += compute_error(lhs[i], rhs[i], c);
    }
    return err/r;
}

template<typename T, size_t I, size_t J>
static inline auto compute_error(T (&lhs)[I][J], T (&rhs)[I][J])
        -> decltype(compare(lhs[0][0], rhs[0][0])) {
    decltype(compare(lhs[0][0], rhs[0][0])) err = 0;
    for(size_t i = 0; i < I; i++) {
        for(size_t j = 0; j < J; j++) {
            err += compare(lhs[I][J], rhs[I][J]);
        }
    }
    return err/(I * J);
}

template<typename T>
static inline T min_error() { return std::numeric_limits<T>::epsilon(); }

/*
 * Print
 */

template<typename T>
static void print_matrix(T** m, size_t r, size_t c, std::ostream& s) {
    for(int i = 0; i < r; i++) {
        s << "[";
        for(int j = 0; j < c; j++) {
            s << m[j][i] << ", ";
        }
        s << "]" << std::endl;
    }
}

#endif /* SRC_CHECK_DATAGEN_HPP_ */
