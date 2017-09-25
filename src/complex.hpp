//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

__forceinline__ __device__ float abs(const cuComplex z){
    auto x = z.x;
    auto y = z.y;

    auto s = max(abs(x), abs(y));

    if(s == 0.0f){
        return 0.0f;
    }

    x = x / s;
    y = y / s;

    return s * sqrt(x * x + y * y);
}

__forceinline__ __device__ double abs(const cuDoubleComplex z){
    auto x = z.x;
    auto y = z.y;

    auto s = max(abs(x), abs(y));

    if(s == 0.0){
        return 0.0;
    }

    x = x / s;
    y = y / s;

    return s * sqrt(x * x + y * y);
}

__forceinline__ __device__ float arg(const cuComplex z){
    auto x = z.x;
    auto y = z.y;

    return atan2f(y, x);
}

__forceinline__ __device__ double arg(const cuDoubleComplex z){
    auto x = z.x;
    auto y = z.y;

    return atan2(y, x);
}

__forceinline__ __device__ cuComplex sqrt(cuComplex z){
    auto x = z.x;
    auto y = z.y;

    if(x == 0.0f){
        auto t = sqrt(abs(y) / 2);
        return make_cuComplex(t, y < 0.0f ? -t : t);
    } else {
        auto t = sqrt(2 * (abs(z) + abs(x)));
        auto u = t / 2;

        if(x > 0.0f){
            return make_cuComplex(u, y / t);
        } else {
            return make_cuComplex(abs(y) / t, y < 0.0f ? -u : u);
        }
    }
}

__forceinline__ __device__ cuDoubleComplex sqrt(cuDoubleComplex z){
    auto x = z.x;
    auto y = z.y;

    if(x == 0.0){
        auto t = sqrt(abs(y) / 2);
        return make_cuDoubleComplex(t, y < 0.0 ? -t : t);
    } else {
        auto t = sqrt(2 * (abs(z) + abs(x)));
        auto u = t / 2;

        if(x > 0.0){
            return make_cuDoubleComplex(u, y / t);
        } else {
            return make_cuDoubleComplex(abs(y) / t, y < 0.0 ? -u : u);
        }
    }
}

__forceinline__ __device__ cuComplex cbrt(cuComplex z){
    auto z_abs = abs(z);
    auto z_arg = arg(z);

    auto new_abs = cbrt(z_abs);
    auto new_arg = z_arg / 3.0f;

    return make_cuComplex(new_abs * cos(new_arg), new_abs * sin(new_arg));
}

__forceinline__ __device__ cuDoubleComplex cbrt(cuDoubleComplex z){
    auto z_abs = abs(z);
    auto z_arg = arg(z);

    auto new_abs = cbrt(z_abs);
    auto new_arg = z_arg / 3.0;

    return make_cuDoubleComplex(new_abs * cos(new_arg), new_abs * sin(new_arg));
}
