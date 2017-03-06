//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#define egblas_unused(x) ((void)x)

#ifdef NDEBUG

#define egblas_assert(condition, message) ((void)0)

#else

#define egblas_assert(condition, message)                                      \
  (condition ? ((void)0)                                                       \
             : assertion_failed_msg(#condition, message, __PRETTY_FUNCTION__,  \
                                    __FILE__, __LINE__))

template <typename CharT>
inline void assertion_failed_msg(const CharT* expr, const char* msg, const char* function, const char* file, long line) {
    std::cerr
        << "***** Internal Program Error - assertion (" << expr << ") failed in "
        << function << ":\n"
        << file << '(' << line << "): " << msg << std::endl;
    std::abort();
}

#endif //NDEBUG
