/*
 * datagen.hpp
 *
 *  Created on: Jan 19, 2018
 *      Author: derick
 */

#ifndef SRC_CHECK_DATAGEN_HPP_
#define SRC_CHECK_DATAGEN_HPP_

#include <cstring>

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

/*
 * compare
 */

static inline float  compare(float  x, float  y) { return x - y; }
static inline double compare(double x, double y) { return x - y; }

template<typename T, size_t I>
static inline auto compute_error(T (&lhs)[I], T (&rhs)[I])
        -> decltype(compare(lhs[0], rhs[0])) {
    decltype(compare(lhs[0], rhs[0])) err = 0;
    for(size_t i = 0; i < I; i++) {
        err += compare(lhs[i], rhs[i]);
    }
    return err/I;
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

#endif /* SRC_CHECK_DATAGEN_HPP_ */
