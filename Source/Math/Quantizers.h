//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once
#include <vector>
#include <type_traits>

#pragma warning(disable : 4127) // conditional expression is constant

namespace Microsoft { namespace MSR { namespace CNTK {

// RawType - input type to the quantizer
// QuantizedType - output type of the quantizer
template <class RawType, class QuantizedType>
class IQuantizerBase 
{
public:
    virtual void Quantize(const std::vector<RawType>& input, std::vector<QuantizedType>& output) = 0;

    virtual void Dequantize(const std::vector<QuantizedType>& input, std::vector<RawType>& output) = 0;
};

template <class RawType, class QuantizedType>
class SymmetricQuantizer : public IQuantizerBase<RawType, QuantizedType>
{
    RawType m_quantizer;
public:
    // extraBits decreases the quantization normalizer to prevent integer overflow during BLAS routines.
    // Higher extraBits will decrease precision of quantization, but will make BLAS routines less prone to overflow.
    // For quantization with shorts, recommended value of extraBits is 1 or 2.
    SymmetricQuantizer(std::vector<RawType>& elements, const int& extraBits)
    {
        assert(elements.size() > 0);
        RawType absMax = FindAbsMax(elements);
        SymmetricQuantizer(absMax, extraBits);
    }

    // absoluteMax - the range of the quantizer (normally represents maximum absolute value of the values in the collection to be quantized.
    // extraBits - see comment in another ctor
    SymmetricQuantizer(RawType& absoluteMax, const int extraBits)
    {
        m_quantizer = absoluteMax * (1 << extraBits);
        if (std::is_same<QuantizedType, short>::value)
        {
            //signed short
            m_quantizer = 32768 / m_quantizer;
        }
        else
        {
            LogicError("Provided type is not yet supported by the quantizer");
        }
    }

    virtual void Quantize(const std::vector<RawType>& input, std::vector<QuantizedType>& output)
    {
        LogicError("TODO");
    }

    virtual void Dequantize(const std::vector<QuantizedType>& input, std::vector<RawType>& output)
    {
        LogicError("TODO");
    }

private: 
    RawType FindAbsMax(std::vector<RawType>& elements) 
    {
        RawType maxElem, minElem = elements[0];
        for (auto element : elements)
        {
            maxElem = std::max(maxElem, element);
            minElem = std::min(minElem, element);
        }

        return std::max(maxElem, std::abs(minElem));
    }
};


}}}