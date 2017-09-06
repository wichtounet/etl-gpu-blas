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
     * \brief Compare a complex number with an expected value
     * \param lhs The complex number (the number to test)
     * \param rhs The expected complex number
     * \return true if they are approximatily the same
     */
    friend bool operator==(const std::complex<T>& lhs, const TestComplex& rhs) {
        bool left  =
                (std::isinf(lhs.real()) && std::isinf(rhs.value.real()))
            ||  (std::isnan(lhs.real()) && std::isnan(rhs.value.real()))
            ||  (std::abs(lhs.real() - rhs.value.real()) < rhs.eps * (T(1) + std::max(std::abs(lhs.real()), std::abs(rhs.value.real()))));

        bool right =
                (std::isinf(lhs.imag()) && std::isinf(rhs.value.imag()))
            ||  (std::isnan(lhs.imag()) && std::isnan(rhs.value.imag()))
            ||  (std::abs(lhs.imag() - rhs.value.imag()) < rhs.eps * (T(1) + std::max(std::abs(lhs.imag()), std::abs(rhs.value.imag()))));

        return left && right;
    }

    /*!
     * \brief Compare a complex number with an expected value
     * \param lhs The expected complex number
     * \param rhs The complex number (the number to test)
     * \return true if they are approximatily the same
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

private:
    T eps;                 ///< The epsilon for comparison
    std::complex<T> value; ///< The expected value
};

namespace Catch {

/*!
 * \brief Overload of Catch::toString for TestComplex<float>
 */
template <>
inline std::string toString<TestComplex<float>>(const TestComplex<float>& value) {
    return value.toString();
}

/*!
 * \brief Overload of Catch::toString for TestComplex<float>
 */
template <>
inline std::string toString<TestComplex<double>>(const TestComplex<double>& value) {
    return value.toString();
}

} // end of namespace Catch
