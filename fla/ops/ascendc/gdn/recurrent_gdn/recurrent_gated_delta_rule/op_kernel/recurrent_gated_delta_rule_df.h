/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef RECURRENT_GDN_DOUBLE_FLOAT_H
#define RECURRENT_GDN_DOUBLE_FLOAT_H

// Two FP32 components retain preprocessing residuals until alpha is rounded.
// Build this operator with reassociation and contraction disabled.
#define RGDN_DF_FN __aicore__ inline

#include <cstdint>

namespace RecurrentGdnDf {
struct Df { float hi; float lo; };

RGDN_DF_FN float High(float x)
{
    union Bits { float f; uint32_t u; } value;
    value.f = x;
    value.u &= 0xfffff000U;
    return value.f;
}

RGDN_DF_FN Df Sum(float a, float b)
{
    const float s = a + b;
    const float v = s - a;
    return {s, (a - (s - v)) + (b - v)};
}

RGDN_DF_FN Df Add(Df a, Df b)
{
    const Df s = Sum(a.hi, b.hi);
    return Sum(s.hi, s.lo + (a.lo + b.lo));
}

RGDN_DF_FN Df Neg(Df x) { return {-x.hi, -x.lo}; }
RGDN_DF_FN Df Sub(Df a, Df b) { return Add(a, Neg(b)); }

RGDN_DF_FN Df Product(float a, float b)
{
    const float ah = High(a), al = a - ah;
    const float bh = High(b), bl = b - bh;
    const float p = a * b;
    const float e = (((ah * bh - p) + ah * bl) + al * bh) + al * bl;
    return {p, e};
}

RGDN_DF_FN Df Mul(Df a, Df b)
{
    const Df p = Product(a.hi, b.hi);
    const float e = ((p.lo + a.hi * b.lo) + a.lo * b.hi) + a.lo * b.lo;
    return Sum(p.hi, e);
}

RGDN_DF_FN Df Divide(Df a, Df b)
{
    const float q = a.hi / b.hi;
    const Df r = Sub(a, Mul(b, {q, 0.0f}));
    return Sum(q, (r.hi + r.lo) / b.hi);
}

// Coefficient pairs are nearest FP32 value and nearest FP32 residual.
RGDN_DF_FN Df InvFactorial(int degree)
{
    switch (degree) {
        case 12: return {2.0876755879584152e-09f, 1.1082839809204342e-16f};
        case 11: return {2.5052107943679403e-08f, 4.4176230446483665e-16f};
        case 10: return {2.755731998149713e-07f, -7.575112209051195e-15f};
        case 9: return {2.7557318844628753e-06f, 3.793571224297229e-14f};
        case 8: return {2.4801587642286904e-05f, -3.40699609366682e-13f};
        case 7: return {0.00019841270113829523f, -2.725596874933456e-12f};
        case 6: return {0.0013888889225199819f, -3.3631094437103215e-11f};
        case 5: return {0.008333333767950535f, -4.34617203337595e-10f};
        case 4: return {0.0416666679084301f, -1.2417634698280722e-09f};
        case 3: return {0.1666666716337204f, -4.967053879312289e-09f};
        case 2: return {0.5f, 0.0f};
        default: return {1.0f, 0.0f};
    }
}

RGDN_DF_FN Df ReciprocalOdd(int k)
{
    // k <= 25; float integer conversion and division are exact/rounded FP32.
    // One remainder correction creates a double-float reciprocal without FMA.
    return Divide({1.0f, 0.0f}, {static_cast<float>(k), 0.0f});
}

RGDN_DF_FN Df ScalePow2(Df x, int n)
{
    // Each normal-range scaling is exact. This implementation does not depend
    // on a scalbn intrinsic or integer bit reinterpretation of exponents.
    const float factor = n >= 0 ? 2.0f : 0.5f;
    const int count = n >= 0 ? n : -n;
    for (int i = 0; i < count; ++i) {
        x.hi *= factor;
        x.lo *= factor;
    }
    return x;
}

RGDN_DF_FN Df Exp(Df x)
{
    if (x.hi < -104.0f) return {0.0f, 0.0f};
    // Callers check the finite domain before any integer conversion.
    const float approx = x.hi * 1.4426950408889634f;
    const int n = static_cast<int>(approx + (approx >= 0.0f ? 0.5f : -0.5f));
    const Df ln2{0.693147182464599609375f, -1.9046542121259336e-09f};
    const Df r = Sub(x, Mul({static_cast<float>(n), 0.0f}, ln2));
    Df value = InvFactorial(12);
    for (int k = 11; k >= 0; --k) value = Add(Mul(value, r), InvFactorial(k));
    return ScalePow2(value, n);
}

RGDN_DF_FN Df Log1p(Df x)
{
    const Df y = Divide(x, Add({2.0f, 0.0f}, x));
    const Df y2 = Mul(y, y);
    Df value = ReciprocalOdd(25);
    for (int k = 23; k >= 1; k -= 2) value = Add(Mul(value, y2), ReciprocalOdd(k));
    return Mul({2.0f, 0.0f}, Mul(y, value));
}

RGDN_DF_FN float RawAlpha(float gate, float aLog, float bias)
{
    // Preserve the public FP32 preprocessing sequence, while evaluating each
    // transcendental with a retained residual before its FP32 rounding.
    const float z = gate + bias;
    const float absolute = z < 0.0f ? -z : z;
    const Df tailPair = Exp({-absolute, 0.0f});
    const float tail = tailPair.hi + tailPair.lo;
    const Df logarithm = Log1p({tail, 0.0f});
    const float softplus = (z > 0.0f ? z : 0.0f) + (logarithm.hi + logarithm.lo);
    const Df magnitudePair = Exp({aLog, 0.0f});
    const float magnitude = magnitudePair.hi + magnitudePair.lo;
    if (softplus > 104.0f / magnitude) return 0.0f;
    const float logGate = -magnitude * softplus;
    const Df alpha = Exp({logGate, 0.0f});
    return alpha.hi + alpha.lo;
}
} // namespace RecurrentGdnDf
#undef RGDN_DF_FN
#endif
