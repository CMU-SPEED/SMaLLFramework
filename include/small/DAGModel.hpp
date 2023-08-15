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

#include <vector>
#include <small.h>
#include <small/Model.hpp>
#include <small/AddLayer.hpp>
#include <small/PartialConv2DLayer.hpp>

#include <small/utils/Graph.hpp>
#include <small/utils/BufferPool.hpp>

// Some possible requirements
// - Use a Graph to store the topology of the network
// - Use topological sort on the graph to determine the sequential
//   processing order of the layers
// - Subclass must determine the max buffer size and number of buffers needed
// - Subclass must call initializeDAG() once before calling inference()
// - Implement a "generic" inference that processes the graph

namespace small
{

//****************************************************************************
template <typename BufferT>
class DAGModel : public Model<BufferT>
{
public:
    DAGModel() = delete;

    // Assume one input layer with a single shape for now
    DAGModel(shape_type const &input_shape, bool save_outputs = false)
        : Model<BufferT>(input_shape),
          m_save_outputs(save_outputs)
    {
    }

    virtual ~DAGModel()
    {
        // delete all buffers
        for (auto buf : m_cached_output_buffers)
        {
            delete buf;
        }
    }

    //************************************************************************
    // assumes all the buffers will be created on demand
    virtual std::vector<Tensor<BufferT>*>
    inference(Tensor<BufferT> const *input)
    {
        if (input->shape() != this->m_input_shape)
        {
            throw std::runtime_error(
                "Input shape does not match model input shape");
        }

        // return any buffers back to pool
        for (size_t idx = 0; idx < this->m_layers.size(); ++idx)
        {
            if (m_cached_output_buffers[idx] != nullptr)
            {
                //std::cerr << "WARNING: returning buffer " << idx
                //          << " to pool.\n";
                m_buffer_pool.push_buffer(m_cached_output_buffers[idx]);
                m_cached_output_buffers[idx] = nullptr;
                m_cached_count[idx] = 0;
            }
        }

        // first node in processing order consumes input
        size_t curr_index = m_processing_order[0];
        std::vector<Tensor<BufferT> const *> input_buffers{input};
        Tensor<BufferT> *output_buffer = m_buffer_pool.pop_buffer();

#ifdef DAG_DEBUG
        std::cerr << "Processing node: 0\n";
#endif
        this->m_layers[curr_index]->compute_output(input_buffers,
                                                   output_buffer);

        // cache output
        m_cached_count[curr_index] = m_graph.out_neighbors(curr_index).size();
        m_cached_output_buffers[curr_index] = output_buffer;

#ifdef DAG_DEBUG_VERBOSE
        // ===========================
        std::cerr << "CACHE:\n";
        for (size_t idx = 0; idx < this->m_layers.size(); ++idx)
        {
            std::cerr << idx << ": cnt=" << m_cached_count[idx]
                      << ", ptr=" << (void*) m_cached_output_buffers[idx]
                      << std::endl;
        }
        // ===========================
#endif

        for (size_t idx = 1; idx < m_processing_order.size(); ++idx)
        {
            curr_index = m_processing_order[idx];
            input_buffers.clear();
#ifdef DAG_DEBUG
            std::cerr << "Processing node: " << curr_index << std::endl;
#endif
            auto &src_indices = m_graph.in_neighbors(curr_index);

            std::type_info const &type(typeid(*(this->m_layers[curr_index])));
            if ((type == typeid(AddLayer<BufferT>)) ||
                (type == typeid(PartialConv2DLayer<BufferT>)))
            {
                assert(src_indices.size() == 2);

                // PartialConv2DLayer and AddLayer: use first input as the output
                /// @todo Make this work with m_save_outputs
                assert(1 == m_cached_count[src_indices[0]]);
                output_buffer = m_cached_output_buffers[src_indices[0]];
                m_cached_output_buffers[src_indices[0]] = nullptr;
                m_cached_count[src_indices[0]] = 0;

                auto buffer = m_cached_output_buffers[src_indices[1]];
                input_buffers.push_back(buffer);
                auto count = --m_cached_count[src_indices[1]];
                if ((count == 0) && !m_save_outputs)
                {
                    // return buffer to pull
                    m_buffer_pool.push_buffer(buffer);
                    m_cached_output_buffers[src_indices[1]] = nullptr;
                }
            }
            else
            {
                // get a new output buffer
                output_buffer = m_buffer_pool.pop_buffer();
                for (size_t src_idx : src_indices)
                {
                    input_buffers.push_back(
                        m_cached_output_buffers[src_idx]);
                }

                // "consume" input_buffers
                for (size_t src_idx : src_indices)
                {
                    auto count = --m_cached_count[src_idx];
                    if ((count == 0) && !m_save_outputs)
                    {
                        // return buffer to pull
                        auto buffer = m_cached_output_buffers[src_idx];
                        m_buffer_pool.push_buffer(buffer);
                        m_cached_output_buffers[src_idx] = nullptr;
                    }
                }
            }

            this->m_layers[curr_index]->compute_output(input_buffers,
                                                       output_buffer);

            // cache the output
            m_cached_count[curr_index] =
                m_graph.out_neighbors(curr_index).size();
            m_cached_output_buffers[curr_index] = output_buffer;

#ifdef DAG_DEBUG_VERBOSE
            // ===========================
            std::cerr << "CACHE:\n";
            for (size_t idx = 0; idx < this->m_layers.size(); ++idx)
            {
                std::cerr << idx << ": cnt=" << m_cached_count[idx]
                          << ", ptr=" << (void*) m_cached_output_buffers[idx]
                          << std::endl;
            }
            // ===========================
#endif
        }

        // return outputs in node index order
        //std::cerr << "Extracting outputs.\n";
        std::vector<Tensor<BufferT>*> output_buffers;
        for (size_t out_idx : m_output_nodes)
        {
            //std::cerr << "Extracting output from node: " << out_idx
            //          << std::endl;
            if (m_cached_count[out_idx] != 0)
            {
                std::cerr << "ERROR: unexpected count = "
                          << m_cached_count[out_idx] << "for output: "
                          << out_idx << std::endl;
            }

            output_buffers.push_back(m_cached_output_buffers[out_idx]);

            if (m_cached_output_buffers[out_idx] == nullptr)
            {
                std::cerr << "ERROR: missing output " << out_idx << std::endl;
            }
            else
            {
                if (!m_save_outputs)
                {
                    m_buffer_pool.push_buffer(m_cached_output_buffers[out_idx]);
                    m_cached_output_buffers[out_idx] = nullptr;
                }
            }
        }

#ifdef DAG_DEBUG_VERBOSE
        std::cerr << "Buffer cache:\n";
        for (size_t idx = 0; idx < m_cached_output_buffers.size(); ++idx)
        {
           std::cerr << idx << ": count = " << m_cached_count[idx]
                     << ", ptr = " << (void*)m_cached_output_buffers[idx]
                     << std::endl;
        }
#endif
        return output_buffers;
    }

    //************************************************************************
    std::vector<size_t> const &get_parent_ids(size_t layer_num) const
    {
        if (layer_num >= m_graph.get_num_nodes())
        {
            throw std::invalid_argument(
                "Darknet::get_parent_ids ERROR: invalid layer_num.");
        }
        return m_graph.in_neighbors(layer_num);
    }

    //************************************************************************
    std::vector<Tensor<BufferT>*> get_layer_outputs() const
    {
        if (m_save_outputs)
        {
            return m_cached_output_buffers;
        }
        else
        {
            std::cerr << "ERROR: Layer outputs not saved.\n";
            throw std::invalid_argument("Saving layer outputs is disabled.");
        }
    }

protected:
    // must be called before inference
    void initializeDAG(size_t max_buffer_size)
    {
        auto sources(m_graph.sources());
        auto sinks(m_graph.sinks());
        auto max_width(m_graph.max_width(0UL));

#ifdef DAG_DEBUG
        if (m_save_outputs)
            std::cout << "Saving outputs for each layer" << std::endl;

        std::cerr << "Num layers: " << this->m_layers.size() << std::endl;
        std::cerr << "first layer addr: " << this->m_layers[0] << std::endl;
        std::cerr << "Graph: " << m_graph;

        std::cerr << "sources:";
        for (auto src : sources) std::cerr << " " << src;
        std::cerr << std::endl;
        std::cerr << "sinks:";
        for (auto dst : sinks) std::cerr << " " << dst;
        std::cerr << std::endl;
        std::cerr << "m_graph.max_width = " << max_width << std::endl;
#endif

        // topo sort
        m_output_nodes = std::move(m_graph.sinks());
        m_processing_order = m_graph.topological_sort();
        if ((m_graph.sources().size() != 1) ||
            (m_processing_order[0] != m_graph.sources()[0]))
        {
            throw std::invalid_argument(
                "ERROR: darknet graph input node error.");
        }

        // compute number of buffers needed
        size_t num_bufs =
            (2 + (max_width - 1) + (m_output_nodes.size() - 1));

#ifdef DAG_DEBUG
        std::cerr << "Number of buffers needed: " << num_bufs << std::endl;
#endif

        m_buffer_pool = std::move(
            small::BufferPool<BufferT>(max_buffer_size, num_bufs));

        m_cached_count = std::vector<size_t>(this->m_layers.size(), 0UL);
        m_cached_output_buffers =
            std::vector<Tensor<BufferT>*>(this->m_layers.size(), nullptr);
    }

    small::Graph                        m_graph;

private:
    bool                                m_save_outputs;

    std::deque<size_t>                  m_processing_order;
    small::BufferPool<BufferT>          m_buffer_pool;

    // Cached outputs with reference counts (from graph)
    // <dependency_count, buf_ptr>
    std::vector<size_t>                 m_cached_count;
    std::vector<Tensor<BufferT>*>       m_cached_output_buffers;

    // Topology and processing vars
    std::vector<small::Graph::VertexID> m_output_nodes;
};

}
