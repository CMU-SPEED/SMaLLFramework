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
#include <deque>
#include <set>
#include <algorithm>

namespace small
{

//****************************************************************************
// A limited graph class used specifically for neural network architectures
// that are sensitive to the order of the parents list
class Graph
{
public:
    using VertexID = size_t;
    using NeighborList = std::vector<VertexID>;  // order matters (don't sort)
    using Adjacency = std::vector<NeighborList>;

    Graph() : m_out_adjacency(), m_in_adjacency() {}
    Graph(size_t num_nodes) :
        m_out_adjacency(num_nodes),
        m_in_adjacency(num_nodes)
    {
    }

    void add_vertex(VertexID v_id)
    {
        //std::cout << "adding v_id: " << v_id << std::endl;
        while (m_out_adjacency.size() <= v_id)
        {
            //std::cout << v_id << "<=" << m_out_adjacency.size() << std::endl;
            m_out_adjacency.push_back(NeighborList());
            m_in_adjacency.push_back(NeighborList());
        }
    }

    void add_edge(VertexID src, VertexID dest)
    {
        //std::cout << "adding edge: " << src << "->" << dest << std::endl;
        if (src >= m_out_adjacency.size() || dest >= m_out_adjacency.size())
        {
            throw std::invalid_argument(
                "Graph::add_link ERROR: node id out of range.");
        }

        if (std::find(m_out_adjacency[src].begin(),
                      m_out_adjacency[src].end(), dest) !=
            m_out_adjacency[src].end())
        {
            std::cerr << "Graph::add_link WARNING: source node ("
                      << src << ") already has neighbor: " << dest
                      << std::endl;
        }
        else
        {
            m_out_adjacency[src].push_back(dest);
            m_in_adjacency[dest].push_back(src);
        }
    }

    size_t get_num_nodes() const { return m_out_adjacency.size(); }

    size_t in_degree(VertexID v) const  { return m_in_adjacency[v].size(); }
    size_t out_degree(VertexID v) const { return m_out_adjacency[v].size(); }

    NeighborList const &in_neighbors(VertexID v) const
    {
        return m_in_adjacency[v];
    }

    NeighborList const &out_neighbors(VertexID v) const
    {
        return m_out_adjacency[v];
    }

    std::vector<VertexID> sources() const
    {
        std::vector<VertexID> srcs;
        for (VertexID dst = 0; dst < m_in_adjacency.size(); ++dst)
        {
            //std::cerr << "in_degree(" << dst << ") = "
            //          << m_in_adjacency[dst].size() << std::endl;
            if (m_in_adjacency[dst].size() == 0)
            {
                srcs.push_back(dst);
            }
        }
        return srcs;
    }

    std::vector<VertexID> sinks() const
    {
        std::vector<VertexID> dsts;
        for (VertexID src = 0; src < m_out_adjacency.size(); ++src)
        {
            //std::cerr << "out_degree(" << src << ") = "
            //          << m_out_adjacency[src].size() << std::endl;
            if (m_out_adjacency[src].size() == 0)
            {
                dsts.push_back(src);
            }
        }
        return dsts;
    }

    size_t max_width(VertexID root) const
    {
        size_t max_width(1UL);

        std::vector<VertexID> wavefront, w2;
        wavefront.push_back(root);

        std::vector<char> visited(get_num_nodes(), 0);
        visited[root] = 1;

        while (!wavefront.empty())
        {
            w2.clear();
            //std::cout << "wavefront:";
            for (auto &src : wavefront)
            {
                //std::cout << " " << src;
                for (auto &neighbor : m_out_adjacency[src])
                {
                    if (visited[neighbor] == 0)
                    {
                        w2.push_back(neighbor);
                        visited[neighbor] = 1;
                    }
                }
            }

            //std::cout << std::endl;
            wavefront.swap(w2);
            //std::cout << "New wavefront size: " << wavefront.size() << std::endl;
            max_width = std::max(max_width, wavefront.size());
        }

        return max_width;
    }

    enum VState
    {
        WHITE = 0,
        GRAY = 1,
        BLACK = 2
    };

    // from sources only
    std::deque<VertexID> topological_sort()
    {
        std::vector<VState> vstate(get_num_nodes(), VState::WHITE);
        std::vector<VertexID> d(get_num_nodes(), 0);
        std::vector<VertexID> finish(get_num_nodes(), 0);
        std::vector<VertexID> parent(get_num_nodes(), -1);
        std::deque<VertexID> topo;
        VertexID time = 0;

        for (VertexID u : sources())
        {
            if (vstate[u] == VState::WHITE)
            {
                dfs_visit(u, time, vstate, d, finish, parent, topo);
            }
        }

#ifdef PARSER_DEBUG
        std::cerr << "Topo sort: ( <- parent: state, d, f_t)\n";
        for (VertexID u : topo)
        {
            std::cerr << u << " <- " << parent[u]
                      << ": " << vstate[u]
                      << ", " << d[u]
                      << ", " << finish[u] << std::endl;
        }
#endif
        return topo;
    }

private:
    void dfs_visit(VertexID u,
                   VertexID              &time,
                   std::vector<VState>   &vstate,
                   std::vector<VertexID> &d,
                   std::vector<VertexID> &finish,
                   std::vector<VertexID> &parent,
                   std::deque<VertexID> &topo)
    {
        //std::cerr << "Visiting " << u << std::endl;
        ++time;
        d[u] = time;
        vstate[u] = VState::GRAY;

        for (VertexID v : m_out_adjacency[u])
        {
            if (vstate[v] == VState::WHITE)
            {
                parent[v] = u;
                dfs_visit(v, time, vstate, d, finish, parent, topo);
            }
        }
        vstate[u] = VState::BLACK;
        ++time;
        finish[u] = time;

        //std::cerr << "Topo push front: " << u << std::endl;
        topo.push_front(u); // reverse topological sort.
    }

    Adjacency m_out_adjacency;  // sets of out-neighbors for each vertex
    Adjacency m_in_adjacency;   // sets of in-neighbors for each vertex

public:
    friend std::ostream &operator<<(std::ostream &ostr, small::Graph const &g)
    {
        ostr << "Num nodes: " << g.get_num_nodes() << std::endl;
        size_t num_edges = 0;
        for (size_t v = 0; v < g.get_num_nodes(); ++v)
        {
            for (auto n : g.in_neighbors(v))
                ostr << n << " ";
            ostr << "--> [" << v << "] -->";
            num_edges += g.m_out_adjacency[v].size();
            for (auto n : g.out_neighbors(v))
                ostr << " " << n;
            ostr << std::endl;
        }
        return ostr;
    }
};


} // small
