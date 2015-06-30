/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * ---------------------------------------------------------------------------
 * Copyright 2014 Nervana Systems Inc.  All rights reserved.
 *
 * * Kernel support for various operations including soft threshold, dropout,
 *   tanh, log1plusexp, gamma, where, etc.
 * * Quantization operation support
 * ---------------------------------------------------------------------------
 *
 *  * ---------------------------------------------------------------------------
 * Altered by Tim Dettmers 2015.  All rights reserved.
 * ---------------------------------------------------------------------------
 *
 */

#ifndef NVMATRIX_OPERATORS_CUH
#define	NVMATRIX_OPERATORS_CUH

class NVMatrixOps {
public:
    class SoftThreshold {
   private:
        const float p;
    public:
        SoftThreshold(const float _p) : p(_p) {
        }
        __device__ inline float operator()(const float a) const {
            return  a > 0 ? fmaxf(0., a - p) : fminf(0., a + p);
        }
    };

    class Fill {
    private:
        float _fill_value;
    public:
        Fill(float fill_value) : _fill_value(fill_value) {
        }
        __device__ inline float operator()(const float a) const {
            return  _fill_value;
        }
    };

    class DropoutKernelOperator {
    private:
        float _keep, _scale;
    public:
        DropoutKernelOperator(float keep) : _keep(keep), _scale(1.0f/keep) {
        }
        __device__ inline float operator()(const float x) const {
            return (x < _keep) * _scale;
        }
    };

    class ClipUpperLower {
    private:
        float _lower, _upper;
    public:
        ClipUpperLower(float lower, float upper) : _lower(lower), _upper(upper) {
        }
        __device__ inline float operator()(const float x) const {
            return (x > _upper) ? _upper : (x < _lower ? _lower : x);
        }
    };

    class Tanh {
    public:
        __device__ inline float operator()(const float a) const {
            return  1.0f - 2.0f / (__expf(2.0f * a) + 1.0f);
        }
    };

    class Log1PlusExp {
    public:
        __device__ inline float operator()(const float a) const {
            return (__logf(1.0f + __expf(-a)) + (a > 0 ? a : 0));
        }
    };

    class Exp {
    public:
        __device__ inline float operator()(const float a) const {
            return __expf(a);
        }
    };

    class Gamma {
    public:
        __device__ inline float operator()(const float a) const {
            return tgammaf(a);
        }
    };

    class LGamma {
    public:
        __device__ inline float operator()(const float a) const {
            return lgammaf(a);
        }
    };

    class Logistic {
    public:
        __device__ inline float operator()(const float a) const {
            return __fdividef(1.0f, 1.0f + __expf(-a));
        }
    };

    class Logistic_grad {
    public:
        __device__ inline float operator()(const float a) const {
            return a*(1.0f-a);
        }
    };

    class RectifiedLinear {
    public:
        __device__ inline float operator()(const float a) const {
            return a < 0.0f ? 0.0f : a;
        }
    };

    class RectifiedLinear_grad {
    public:
        __device__ inline float operator()(const float a) const {
            return a < 0.0f ? 0.0f : 1.0f;
        }
    };

    class Log {
    public:
        __device__ inline float operator()(const float a) const {
            return __logf(a);
        }
    };

    class Square {
    public:
        __device__ inline float operator()(const float a) const {
            return a * a;
        }
    };

    class Sqrt {
    public:
        __device__ inline float operator()(const float a) const {
            return sqrtf(a);
        }
    };

    class SqrtAbs {
    public:
        __device__ inline float operator()(const float a) const {
            return sqrtf(fabsf(a));
        }
    };

    class Reciprocal {
    public:
        __device__ inline float operator()(const float a) const {
            return 1.0f / a;
        }
    };

    class Abs {
    public:
        __device__ inline float operator()(const float a) const {
            return a > 0 ? a : -a;
        }
    };

    class Sign {
    public:
        __device__ inline float operator()(const float a) const {
            return (a > 0) - (a < 0);
        }
    };
    
    class Identity {
    public:
        __device__ inline float operator()(const float a) const {
            return a;
        }
    };

    class Zero {
    public:
        __device__ inline float operator()(const float a) const {
            return 0;
        }
    };

    class Value {
    private:
        const float scalar;
    public:
        Value(const float _scalar) : scalar(_scalar) {
        }
        __device__ inline float operator()(const float a) const {
            return scalar;
        }
    };

    class One {
    public:
        __device__ inline float operator()(const float a) const {
            return 1;
        }
    };
    
    class Const {
    private:
        const float scalar;
    public:
        Const(const float _scalar) : scalar(_scalar) {
        }
        __device__ inline float operator()(const float a) const {
            return scalar;
        }
    };

    class OneMinus {
    public:
        __device__ inline float operator()(const float x) const {
            return 1.0f - x;
        }
    };

    class Linear {
    protected:
        float _a, _b;
    public:
        __device__ inline float operator()(float x) const {
            return _a * x + _b;
        }
        Linear(float a, float b) : _a(a), _b(b) {
        }
    };

    class IsNan {
    public:
        __device__ inline float operator()(const float a) const {
            return isnan(a);
        }
    };

    class IsInf {
    public:
        __device__ inline float operator()(const float a) const {
            return isinf(a);
        }
    };

    class SmallerThanScalar {
    private:
        const float scalar;
    public:
        SmallerThanScalar(const float _scalar) : scalar(_scalar) {
        }
        __device__ inline float operator()(const float a) const {
            return a < scalar;
        }
    };

    class BiggerThanScalar {
    private:
        const float scalar;
    public:
        BiggerThanScalar(const float _scalar) : scalar(_scalar) {
        }
        __device__ inline float operator()(const float a) const {
            return a > scalar;
        }
    };

    class EqualsScalar {
    private:
        const float scalar;
    public:
        EqualsScalar(const float _scalar) : scalar(_scalar) {
        }
        __device__ inline float operator()(const float a) const {
            return a == scalar;
        }
    };

    class AddScalar {
    private:
        const float scalar;
    public:
        AddScalar(const float _scalar) : scalar(_scalar) {
        }
        __device__ inline float operator()(const float a) const {
            return a + scalar;
        }
    };

    class WeightedAddScalar {
    private:
        const float weight, scalar;
    public:
        WeightedAddScalar(const float _weight, const float _scalar) : weight(_weight), scalar(_scalar) {
        }
        __device__ inline float operator()(const float a) const {
            return weight * a + scalar;
        }
    };

    class MultByScalar {
    private:
        const float scalar;
    public:
        MultByScalar(const float _scalar) : scalar(_scalar) {
        }
        __device__ inline float operator()(const float a) const {
            return a * scalar;
        }
    };

    class DivByScalar {
    private:
        const float scalar;
    public:
        DivByScalar(const float _scalar) : scalar(_scalar) {
        }
        __device__ inline float operator()(const float a) const {
            return __fdividef(a, scalar);
        }
    };

    class Pow {
    private:
        const float p;
    public:
        Pow(const float _p) : p(_p) {
        }
        __device__ inline float operator()(const float a) const {
            return __powf(a, p);
        }
    };

    template <bool exclusive>
    class InRange {
    private:
        const float lower, upper;
    public:
        InRange(const float _lower, const float _upper) : lower(_lower), upper(_upper) {
        }
        __device__ inline float operator()(const float a) const {
            return exclusive ? a > lower && a < upper : a >= lower && a <= upper;
        }
    };

    class MinWithScalar {
    private:
        const float scalar;
    public:
        MinWithScalar(const float _scalar) : scalar(_scalar) {
        }
        __device__ inline float operator()(const float a) const {
            return a > scalar ? scalar : a;
        }
    };

    class MaxWithScalar {
    private:
        const float scalar;
    public:
        MaxWithScalar(const float _scalar) : scalar(_scalar) {
        }
        __device__ inline float operator()(const float a) const {
            return a > scalar ? a : scalar;
        }
    };

    class Quantize {
    private:
	float prec;
        float qmax;
    public:
        Quantize(const int _iw, const int _fw) {
            prec = powf(2.0,-_fw);
            qmax = powf(2.0, _iw) - prec;
        }
        __device__ inline float operator()(const float a) const {
            // if (fabsf(a)<prec) 
            //     return a > 0 ? prec : -prec; 
            // else
                return fminf(fmaxf(rintf(a/prec)*prec, -qmax), qmax);
            // return fminf(fmaxf(truncf(a/prec + 0.5)*prec, -qmax), qmax);
        }
    };
};

class NVMatrixBinaryOps {
public:
    class BinaryOp {
    public:
    };
    class Equals : public BinaryOp {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return a == b;
        }
    };

    class Maximum : public BinaryOp {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return fmaxf(a,b);
        }
    };

    class Minimum : public BinaryOp {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return fminf(a,b);
        }
    };

    class Power : public BinaryOp {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return __powf(a, b);
        }
    };

    class SmallerThan : public BinaryOp {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return a < b;
        }
    };

    class BiggerThan : public BinaryOp {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return a > b;
        }
    };

    class Divide : public BinaryOp {
    public:
        __device__ inline float operator()(const float a, const float b) const  {
            return __fdividef(a, b);
        }
    };

    class Multiply : public BinaryOp {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return a * b;
        }
    };

    class SquaredDiff : public BinaryOp {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return (a - b) * (a - b);
        }
    };

    class SubSquared : public BinaryOp {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return (a - b * b);
        }
    };

    class Dropout : public BinaryOp {
    private:
        const float threshold;
    public:
        Dropout(const float _threshold) : threshold(_threshold) {
        }
        Dropout() : threshold(0.5f){ // Compiler complains about no default constructor?
        }
        __device__ inline float operator()(const float a, const float b) const {
            return b > threshold ? a : 0.0f;
        }
    };

    class WeightedAdd : public BinaryOp {
    private:
        const float scaleA, scaleB;
    public:
        WeightedAdd(const float _scaleA, const float _scaleB) : scaleA(_scaleA), scaleB(_scaleB) {
        }
        WeightedAdd() : scaleA(0), scaleB(0) { // Compiler complains about no default constructor?
        }
        __device__ inline float operator()(const float a, const float b) const {
            return a * scaleA + b * scaleB;
        }
    };

    class WeightedAdd1 : public BinaryOp {
    private:
        const float scaleB;
    public:
        WeightedAdd1(const float _scaleB) : scaleB(_scaleB) {
        }
        __device__ inline float operator()(const float a, const float b) const {
            return a + b * scaleB;
        }
    };

    class ScaledAdd : public BinaryOp {
    private:
        const float scaleB;
    public:
        ScaledAdd(const float _scaleB) : scaleB(_scaleB) {
        }
        __device__ inline float operator()(const float a, const float b) const {
            return a + b * scaleB;
        }
    };

    class AxPBysq : public BinaryOp {
    private:
        const float scaleA, scaleB;
    public:
        AxPBysq(const float _scaleA, const float _scaleB) : scaleA(_scaleA), scaleB(_scaleB) {
        }
        AxPBysq() : scaleA(0), scaleB(0) { // Compiler complains about no default constructor?
        }
        __device__ inline float operator()(const float a, const float b) const {
            return a * scaleA + b * b * scaleB;
        }
    };

    class Add : public BinaryOp {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return a + b;
        }
    };
    
    class First : public BinaryOp {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return a;
        }
    };
    
    class Second : public BinaryOp {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return b;
        }
    };
    
    class SecondScaled : public BinaryOp {
    private:
        const float scale;
    public:
        SecondScaled(const float _scale) : scale(_scale) {
        }

        SecondScaled() : scale(0) { // Compiler complains about no default constructor?
        }
        __device__ inline float operator()(const float a, const float b) const {
            return scale * b;
        }
    };

    template<class UnaryOp, class BinaryOp>
    class CompositeSecond : public BinaryOp {
    private:
        UnaryOp _uop;
        BinaryOp _bop;
    public:
        CompositeSecond(UnaryOp uop, BinaryOp bop) : _uop(uop), _bop(bop) {

        }
        __device__ inline float operator()(const float a, const float b) const {
            return _bop(a, _uop(b));
        }
    };
};

class NVMatrixAggs {
public:
    class Sum {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return a + b;
        }
        __device__ inline float getBaseValue() {
            return 0;
        }
    };

// old
    class SumSq {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return a + b * b;
        }
        __device__ inline float getBaseValue() {
            return 0;
        }
    };
// new
    class SumAbs {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return a + fabsf(b);
        }
        __device__ inline float getBaseValue() {
            return 0;
        }
    };

    class Max {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return a > b ? a : b;
        }
        __device__ inline float getBaseValue() {
            return -2e38;
        }
    };

    class MaxAbs {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return fabsf(b) > a ? fabsf(b) : a;
        }
        __device__ inline float getBaseValue() {
            return 0;
        }
    };

    class Min {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return a > b ? b : a;
        }
        __device__ inline float getBaseValue() {
            return 2e38;
        }
    };

    class CountNan {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return a + isnan(b);
        }
        __device__ inline float getBaseValue() {
            return 0;
        }
    };

    class CountInf {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return a + isinf(b);
        }
        __device__ inline float getBaseValue() {
            return 0;
        }
    };

    template<class UnaryOperator>
    class ArgMax {
    private:
       UnaryOperator u;
    public:
       ArgMax(UnaryOperator _u) : u(_u) {
       }
       __device__ inline float operator()(const float a, const float b) const {
           return u(a) > u(b) ? a : b;
       }
       __device__ inline float getBaseValue() {
           return u.getArgMin();
       }
    };
};

class NVMatrixTernaryOps {
public:
    class Add {
    public:
        __device__ inline float operator()(const float a, const float b, const float c) const {
            return a + b + c;
        }
    };
    class Where {
    public:
        __device__ inline float operator()(const float a, const float b, const float c) const {
            return a ? b : c;
        }
    };
    class WeightedAdd {
    private:
        const float scaleA, scaleB, scaleC;
    public:
        WeightedAdd(const float _scaleA, const float _scaleB, const float _scaleC) : scaleA(_scaleA), scaleB(_scaleB), scaleC(_scaleC) {
        }
        __device__ inline float operator()(const float a, const float b, const float c) const {
            return a * scaleA + b * scaleB + c * scaleC;
        }
    };
    class SqrtRatioMult {
    private:
        const float epsilon;
    public:
        SqrtRatioMult(const float _epsilon) : epsilon(_epsilon) {
        }
        __device__ inline float operator()(const float a, const float b, const float c) const {
            return -sqrtf( (a+epsilon) / (b+epsilon) ) * c;
        }
    };
};

#endif	/* NVMATRIX_OPERATORS_CUH */

