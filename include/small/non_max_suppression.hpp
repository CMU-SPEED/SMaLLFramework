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

namespace small
{

//****************************************************************************
// NMS from
// https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/
//****************************************************************************

struct bbox
{
    float x, y, w, h;
    float objectness;
    float class_score;
    size_t class_idx;
};

//****************************************************************************
float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

//****************************************************************************
float bbox_iou(bbox const &a, bbox const &b)
{
    float area_a = a.w*a.h;
    float area_b = b.w*b.h;

    // intersection
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    float intersection_area = ((w < 0.f) || (h < 0.f)) ? 0.f : w*h;

    // union
    float union_area = area_a + area_b - intersection_area;
    //std::cerr << "I,U: " << intersection_area << "," << union_area << std::endl;
    return intersection_area / union_area;
}

//****************************************************************************
std::vector<bbox> basic_nms(std::vector<bbox> predictions,
                            float iou_threshold = 0.45f) // copy intended
{
    std::sort(predictions.begin(), predictions.end(),
              [](auto &a, auto &b) { return a.objectness > b.objectness; });

    std::cout << "Collected " << predictions.size() << " predictions.\n";
    for (auto &[x, y, w, h, objectness, class_score, class_idx] : predictions)
    {
        std::cout << "corners:(" << x-w/2.f << "," << y-h/2.f
                  << ")->("  << x+w/2.f << "," << y+h/2.f
                  << "), objectness=" << objectness
                  << ", class_index=" << class_idx
                  << ", class_score=" << class_score << std::endl;
    }

    std::vector<bbox> keep_list;
    while (!predictions.empty())
    {
        std::vector<bbox> survivor_list;
        keep_list.push_back(predictions[0]);
        predictions.erase(predictions.begin());
        auto const &curr_bbox = keep_list.back();
        std::cerr << "KEEPING: [c("
                  << curr_bbox.x << "," << curr_bbox.y << "),sz("
                  << curr_bbox.w << "," << curr_bbox.h << "),obj="
                  << curr_bbox.objectness << ",class="
                  << curr_bbox.class_idx << ",conf="
                  << curr_bbox.class_score << "]"
                  << std::endl;
        for (auto &bbox : predictions)
        {
            auto iou = bbox_iou(curr_bbox, bbox);
            std::cerr << "     IoU = " << iou << ", [c("
                      << bbox.x << "," << bbox.y << "),sz("
                      << bbox.w << "," << bbox.h << "),obj="
                      << bbox.objectness << ",class="
                      << bbox.class_idx << ",conf="
                      << bbox.class_score << "]"
                      << std::endl;
            if (iou < iou_threshold)
            {
                survivor_list.push_back(bbox);
            }
            else
            {
                std::cerr << "**REMOVED**\n";
            }
        }

        predictions.swap(survivor_list);
    }

    return keep_list;
}

} // ns small
