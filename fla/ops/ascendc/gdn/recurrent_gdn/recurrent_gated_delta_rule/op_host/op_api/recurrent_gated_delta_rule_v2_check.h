/** Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0. */
#ifndef RECURRENT_GDN_V2_CHECK_H
#define RECURRENT_GDN_V2_CHECK_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace rgdn_v2 {
struct Params {
    const aclTensor *q, *k, *v, *beta;
    aclTensor *state;
    const aclTensor *lengths, *indices, *g, *gk, *accepted, *aLog, *bias;
    float scale;
    bool norm, rawGate, sigmoid, twice;
    aclTensor *out;
};

inline uint64_t ElementBytes(const aclTensor *t)
{
    return t->GetDataType() == op::DataType::DT_BF16 ? 2 : 4;
}

inline bool Type(const aclTensor *t, std::initializer_list<op::DataType> allowed)
{
    return !t || std::find(allowed.begin(), allowed.end(), t->GetDataType()) != allowed.end();
}

inline bool Shape(const aclTensor *t, std::initializer_list<int64_t> expected)
{
    if (!t || t->GetViewShape().GetDimNum() != expected.size()) return false;
    size_t i = 0;
    for (auto n : expected) if (t->GetViewShape().GetDim(i++) != n) return false;
    return true;
}

// Validate every multiply/add before using an address or narrowing a DMA gap.
inline bool ViewValid(const aclTensor *t)
{
    if (!t) return true;
    const auto &shape = t->GetViewShape();
    const auto &stride = t->GetViewStrides();
    const auto &storageShape = t->GetStorageShape();
    uint64_t storage = 1;
    const uint64_t limit = static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / ElementBytes(t);
    for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
        int64_t n = storageShape.GetDim(i);
        if (n <= 0 || static_cast<uint64_t>(n) > limit / storage) return false;
        storage *= static_cast<uint64_t>(n);
    }
    const int64_t offset = t->GetViewOffset();
    if (offset < 0 || static_cast<uint64_t>(offset) >= storage || stride.size() != shape.GetDimNum()) return false;
    uint64_t last = static_cast<uint64_t>(offset);
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        int64_t n = shape.GetDim(i), s = stride[i];
        if (n <= 0 || s < 0) return false;
        if (n > 1 && static_cast<uint64_t>(s) > (storage - 1 - last) / static_cast<uint64_t>(n - 1)) return false;
        last += static_cast<uint64_t>(n - 1) * static_cast<uint64_t>(s);
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(t->GetStorageAddr());
    return base != 0 && (last + 1) <= (std::numeric_limits<uintptr_t>::max() - base) / ElementBytes(t);
}

using Interval = std::pair<uintptr_t, uintptr_t>;
inline Interval Extent(const aclTensor *t)
{
    uint64_t last = 0;
    for (size_t i = 0; i < t->GetViewShape().GetDimNum(); ++i)
        last += (t->GetViewShape().GetDim(i) - 1) * t->GetViewStrides()[i];
    uintptr_t begin = reinterpret_cast<uintptr_t>(t->GetStorageAddr()) + t->GetViewOffset() * ElementBytes(t);
    return {begin, begin + (last + 1) * ElementBytes(t)};
}

inline std::vector<Interval> Intervals(const aclTensor *t)
{
    std::vector<std::pair<uint64_t, uint64_t>> axes;
    for (size_t i = 0; i < t->GetViewShape().GetDimNum(); ++i)
        if (t->GetViewShape().GetDim(i) > 1)
            axes.emplace_back(t->GetViewStrides()[i], t->GetViewShape().GetDim(i));
    std::sort(axes.begin(), axes.end());
    uint64_t span = 1;
    size_t dense = 0;
    while (dense < axes.size() && axes[dense].first == span) {
        span *= axes[dense].second;
        ++dense;
    }
    std::vector<uintptr_t> starts{Extent(t).first};
    for (size_t i = dense; i < axes.size(); ++i) {
        std::vector<uintptr_t> expanded;
        for (auto start : starts)
            for (uint64_t j = 0; j < axes[i].second; ++j)
                expanded.push_back(start + j * axes[i].first * ElementBytes(t));
        starts.swap(expanded);
    }
    std::sort(starts.begin(), starts.end());
    std::vector<Interval> result;
    for (auto start : starts) result.emplace_back(start, start + span * ElementBytes(t));
    return result;
}

inline bool Intersect(const std::vector<Interval> &a, const std::vector<Interval> &b)
{
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (std::max(a[i].first, b[j].first) < std::min(a[i].second, b[j].second)) return true;
        if (a[i].second <= b[j].second) ++i; else ++j;
    }
    return false;
}

inline bool Alias(const aclTensor *a, const aclTensor *b)
{
    if (!a || !b) return false;
    auto x = Extent(a), y = Extent(b);
    if (x.second <= y.first || y.second <= x.first) return false;
    return Intersect(Intervals(a), Intervals(b));
}

inline bool MatrixView(const aclTensor *t, size_t rowAxis)
{
    for (auto stride : t->GetViewStrides()) if (stride <= 0) return false;
    const auto &s = t->GetViewStrides();
    return s[s.size() - 1] == 1 && s[rowAxis] >= 128 &&
        static_cast<uint64_t>(s[rowAxis] - 128) <= std::numeric_limits<uint32_t>::max() / ElementBytes(t);
}

inline bool Check(const Params &p)
{
    using op::DataType;
    if (!p.q || !p.k || !p.v || !p.beta || !p.state || !p.lengths || !p.indices || !p.out || p.gk)
        return false;
    if (!std::isfinite(p.scale) || (p.rawGate && (!p.g || !p.aLog))) return false;
    for (auto t : {p.q, p.k, p.v, static_cast<const aclTensor *>(p.out)})
        if (!Type(t, {DataType::DT_BF16})) return false;
    for (auto t : {p.beta, p.g, static_cast<const aclTensor *>(p.state)})
        if (!Type(t, {DataType::DT_BF16, DataType::DT_FLOAT})) return false;
    for (auto t : {p.aLog, p.bias}) if (!Type(t, {DataType::DT_FLOAT})) return false;
    for (auto t : {p.lengths, p.indices, p.accepted}) if (!Type(t, {DataType::DT_INT32})) return false;
    if (p.q->GetViewShape().GetDimNum() != 3 || p.v->GetViewShape().GetDimNum() != 3 ||
        p.state->GetViewShape().GetDimNum() != 4 || p.lengths->GetViewShape().GetDimNum() != 1) return false;
    const int64_t t = p.q->GetViewShape().GetDim(0), hk = p.q->GetViewShape().GetDim(1);
    const int64_t hv = p.v->GetViewShape().GetDim(1), blocks = p.state->GetViewShape().GetDim(0);
    const int64_t b = p.lengths->GetViewShape().GetDim(0) - 1;
    if (t <= 0 || hk <= 0 || hv <= 0 || blocks <= 0 || b <= 0 || hv % hk || hv / hk > 16 ||
        b > std::numeric_limits<int64_t>::max() / hv) return false;
    if (!Shape(p.q, {t,hk,128}) || !Shape(p.k, {t,hk,128}) || !Shape(p.v, {t,hv,128}) ||
        !Shape(p.beta, {t,hv}) || !Shape(p.state, {blocks,hv,128,128}) || !Shape(p.out, {t,hv,128})) return false;
    if (p.indices->GetViewShape().GetDimNum() == 1) {
        if (!Shape(p.indices, {t})) return false;
    } else if (p.indices->GetViewShape().GetDimNum() == 2) {
        if (p.indices->GetViewShape().GetDim(0) != b || p.indices->GetViewShape().GetDim(1) <= 0) return false;
    } else return false;
    if ((p.g && !Shape(p.g, {t,hv})) || (p.aLog && !Shape(p.aLog, {hv})) ||
        (p.bias && !Shape(p.bias, {hv})) || (p.accepted && !Shape(p.accepted, {b}))) return false;
    for (auto x : {p.q,p.k,p.v,p.beta,static_cast<const aclTensor *>(p.state),p.lengths,p.indices,
                   p.g,p.accepted,p.aLog,p.bias,static_cast<const aclTensor *>(p.out)}) if (!ViewValid(x)) return false;
    if (!MatrixView(p.q,0) || !MatrixView(p.k,0) || !MatrixView(p.v,0) || !MatrixView(p.state,2) || !op::IsContiguous(p.out)) return false;
    // Fast proof for padding and permutations; exact intervals admit legal
    // interleaved State views instead of treating an inconclusive proof as failure.
    std::vector<std::pair<uint64_t,uint64_t>> axes;
    const auto &shape = p.state->GetViewShape();
    for (size_t i=0; i<4; ++i) if (shape.GetDim(i)>1)
        axes.emplace_back(p.state->GetViewStrides()[i],shape.GetDim(i));
    std::sort(axes.begin(),axes.end());
    uint64_t span=1; bool disjoint=true;
    for (auto axis:axes) {
        if (axis.first<span) {disjoint=false; break;}
        span+=(axis.second-1)*axis.first;
    }
    if (!disjoint) {
        auto intervals=Intervals(p.state);
        for (size_t i=1;i<intervals.size();++i) if(intervals[i-1].second>intervals[i].first) return false;
    }
    for (auto input : {p.q,p.k,p.v,p.beta,p.lengths,p.indices,p.g,p.accepted,p.aLog,p.bias})
        if (Alias(p.state,input) || Alias(p.out,input)) return false;
    return !Alias(p.state,p.out);
}
} // namespace rgdn_v2
#endif
