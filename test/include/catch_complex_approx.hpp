//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains the TestComplex class to compare complex numbers with a margin of error
 */

#pragma once

#include <cmath>
#include <limits>
#include <string>
#include <complex>

/*!
 * \brief Utility class to compare two complex numbers with a margin of error
 */
template <typename T>
struct TestComplex {
    /*!
     * \brief Construct a TestComplex for the given complex value
     * \param value the expected complex value
     */
    explicit TestComplex(const std::complex<T>& value, T eps = std::numeric_limits<float>::epsilon() * 10000)
            : eps(eps), value(value) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a TestComplex for the given complex value
     * \param real the expected real part
     * \param imag the expected imaginary part
     */
    TestComplex(T real, T imag, T eps = std::numeric_limits<float>::epsilon() * 10000)
            : eps(eps), value(real, imag) {
        //Nothing else to init
    }

    TestComplex(const TestComplex& other) = default;

    /*!
     * \brief Compare two numbers for approx equality
     * \param lhs The number
     * \param rhs The expected number
     * \return true if they are approximately the same
     */
    static bool check_two(T lhs, T rhs, T eps){
        if(std::isinf(lhs) || std::isnan(lhs)){
            return std::isinf(rhs) || std::isnan(rhs);
        }

        return (std::abs(lhs - rhs) < eps * (T(1) + std::max(std::abs(lhs), std::abs(rhs))));
    }

    /*!
     * \brief Compare a complex number with an expected value
     * \param lhs The complex number (the number to test)
     * \param rhs The expected complex number
     * \return true if they are approximately the same
     */
    friend bool operator==(const std::complex<T>& lhs, const TestComplex& rhs) {
        bool left = check_two(lhs.real(), rhs.value.real(), rhs.eps);
        bool right = check_two(lhs.imag(), rhs.value.imag(), rhs.eps);

        return left && right;
    }

    /*!
     * \brief Compare a complex number with an expected value
     * \param lhs The expected complex number
     * \param rhs The complex number (the number to test)
     * \return true if they are approximately the same
     */
    friend bool operator==(const TestComplex& lhs, const std::complex<T>& rhs) {
        return operator==(rhs, lhs);
    }

    /*!
     * \brief Compare a complex number with an expected value for inequality
     * \param lhs The complex number (the number to test)
     * \param rhs The expected complex number
     * \return true if they are not approximatily the same
     */
    friend bool operator!=(const std::complex<T>& lhs, const TestComplex& rhs) {
        return !operator==(lhs, rhs);
    }

    /*!
     * \brief Compare a complex number with an expected value for inequality
     * \param lhs The expected complex number
     * \param rhs The complex number (the number to test)
     * \return true if they are not approximatily the same
     */
    friend bool operator!=(const TestComplex& lhs, const std::complex<T>& rhs) {
        return !operator==(rhs, lhs);
    }

    /*!
     * \brief Returns a textual representation of the operand for Catch
     * \return a std::string representing this operand
     */
    std::string toString() const {
        std::ostringstream oss;
        oss << "TestComplex(" << value << ")";
        return oss.str();
    }

    friend std::ostream & operator<<(std::ostream & os, const TestComplex & value) {
        return os << value.toString();
    }

private:
    T eps;                 ///< The epsilon for comparison
    std::complex<T> value; ///< The expected value
};
