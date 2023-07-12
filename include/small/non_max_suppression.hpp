//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2023 by The SMaLL Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM23-0126
//****************************************************************************

#pragma once

#include <algorithm>

//#define NMS_DEBUG

//****************************************************************************
// NMS from
// https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/
//
/// @todo deal with each class separately
//
//****************************************************************************

namespace small
{

//****************************************************************************
// Bounding boxes for object detections (specifically for NMS...for now)
//****************************************************************************
struct BoundingBox
{
    float x, y;  // center coordinate
    float w, h;  // extents

    //************************************************************************
    static float overlap(float x1, float w1, float x2, float w2)
    {
        float l1 = x1 - w1/2;
        float l2 = x2 - w2/2;
        float left = l1 > l2 ? l1 : l2;
        float r1 = x1 + w1/2;
        float r2 = x2 + w2/2;
        float right = r1 < r2 ? r1 : r2;
        return right - left;
    }

    //************************************************************************
    float iou(BoundingBox const &other) const
    {
        float area_a = w*h;
        float area_b = other.w*other.h;

        // intersection
        float wo = overlap(x, w, other.x, other.w);
        float ho = overlap(y, h, other.y, other.h);
        float intersection_area = ((wo < 0.f) || (ho < 0.f)) ? 0.f : wo*ho;

        // union
        float union_area = area_a + area_b - intersection_area;
        return intersection_area / union_area;
    }
};

std::ostream &operator<<(std::ostream &ostr, BoundingBox const &bbox)
{
    ostr << "center(" << bbox.x << "," << bbox.y
         << "),size("  << bbox.w << "," << bbox.h
         << ")";
    return ostr;
}

//****************************************************************************
// NMS detection object
//****************************************************************************
struct Detection
{
    BoundingBox bbox;
    float       objectness;   // objectness confidence score
    float       class_score;  // class confidence score
    size_t      class_id;     // max class id (index)
    // probability = objectness * class_score;
};

std::ostream &operator<<(std::ostream &ostr, Detection const &detection)
{
    ostr << "[" << detection.bbox
         << ",objectness=" << detection.objectness
         << ",class_id=" << detection.class_id
         << ",p(class)=" << detection.class_score
         << ",prob=" << (detection.class_score * detection.objectness) << "]";
    return ostr;
}

//****************************************************************************
std::vector<Detection> basic_nms(
    std::vector<Detection> predictions,        // copy intended
    float                  iou_threshold)      //= 0.45f
{
    std::sort(predictions.begin(), predictions.end(),
              [](auto &a, auto &b) { return a.objectness > b.objectness; });

#ifdef NMS_DEBUG
    std::cout << "Collected " << predictions.size() << " predictions.\n";
    for (auto const &detection : predictions)
    {
        std::cout << detection << std::endl;
    }
#endif

    std::vector<Detection> keep_list;
    while (!predictions.empty())
    {
        std::vector<Detection> survivor_list;
        keep_list.push_back(predictions[0]);
        predictions.erase(predictions.begin());
        auto const &curr_detection = keep_list.back();
#ifdef NMS_DEBUG
        std::cout << "KEEPING:  " << curr_detection << std::endl;
#endif
        for (auto &detection : predictions)
        {
            if (detection.class_id == curr_detection.class_id)
            {
                auto iou = curr_detection.bbox.iou(detection.bbox);
#ifdef NMS_DEBUG
                std::cout << "     IoU = " << iou << ", " << detection << std::endl;
#endif
                if (iou < iou_threshold)
                {
                    survivor_list.push_back(detection);
                }
#ifdef NMS_DEBUG
                else
                {
                    std::cerr << "**REMOVED**\n";
                }
#endif
            }
            else
            {
#ifdef NMS_DEBUG
                std::cout << "SKIPPING: " << detection << std::endl;
#endif

                survivor_list.push_back(detection);
            }
        }

        predictions.swap(survivor_list);
    }

    return keep_list;
}

} // ns small
