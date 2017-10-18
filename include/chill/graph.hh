/*****************************************************************************
 Copyright (C) 2008 University of Southern California
 Copyright (C) 2010 University of Utah
 All Rights Reserved.

 Purpose:
   Graph<VertexType, EdgeType> template class supports topological sort
 with return result observing strongly connected component.

 Notes:
   The result of topologically sorting a graph V={1,2,3,4} and E={1->2, 1->3,
 2->3, 3->2, 3->4} is ({1}, {2,3}, {4}).
 
 History:
   01/2006 Created by Chun Chen.
   07/2010 add a new topological order, -chun
*****************************************************************************/

#ifndef GRAPH_HH
#define GRAPH_HH

/*!
 * \file
 * \brief  Graph<VertexType, EdgeType> template class supports topological sort
 *
 * The result of topologically sorting a graph V={1,2,3,4} and E={1->2, 1->3,
 * 2->3, 3->2, 3->4} is ({1}, {2,3}, {4}).
 */

#include <set>
#include <vector>
#include <map>
#include <iostream>
#include <stack>
#include <algorithm>
#include <assert.h>

struct Empty {
  Empty() {};

  bool operator<(const Empty &) const { return true; };

  bool operator==(const Empty &) const { return false; };

  friend std::ostream &operator<<(std::ostream &os, const Empty &) { return os; };
};

namespace {
  enum GraphColorType {
    WHITE, GREY, BLACK
  };
}

template<typename VertexType, typename EdgeType>
struct Graph;

template<typename VertexType, typename EdgeType>
std::ostream &operator<<(std::ostream &os, const Graph<VertexType, EdgeType> &g);

template<typename VertexType = Empty, typename EdgeType = Empty>
struct Graph {
  typedef std::map<int, std::vector<EdgeType> > EdgeList;
  typedef std::vector<std::pair<VertexType, EdgeList> > VertexList;

  VertexList vertex;
  bool directed;

  Graph(bool directed = true);

  int vertexCount() const;

  int edgeCount() const;

  bool isEmpty() const;

  bool isDirected() const;

  int insert(const VertexType &v = VertexType());

  void connect(int v1, int v2, const EdgeType &e = EdgeType());

  void connect(int v1, int v2, const std::vector<EdgeType> &e);

  void disconnect(int v1, int v2);

  bool hasEdge(int v1, int v2) const;

  std::vector<EdgeType> getEdge(int v1, int v2) const;

  //! Topological sort
  /*!
   * This topological sort does handle SCC in graph.
   * Result is a sort order with a set at each location representing a SCC.
   */
  std::vector<std::set<int> > topoSort() const;
  //! Topological sort
  /*!
   * This topological sort does not handle SCC in graph.
   * Result is a sort order with a layer of node at each location.
   */
  std::vector<std::set<int> > packed_topoSort() const;

  void dump() {
    std::cout << *this;
  }

  friend std::ostream &operator<<<>(std::ostream &os, const Graph<VertexType, EdgeType> &g);
};

template<typename VertexType, typename EdgeType>
std::ostream &operator<<(std::ostream &os, const Graph<VertexType, EdgeType> &g) {
  for (int i = 0; i < g.vertex.size(); i++)
    for (typename Graph<VertexType, EdgeType>::EdgeList::const_iterator j = g.vertex[i].second.begin();
         j != g.vertex[i].second.end(); j++) {
      os << i << "->" << j->first << ":";
      for (typename std::vector<EdgeType>::const_iterator k = j->second.begin(); k != j->second.end(); k++)
        os << " " << *k;
      os << std::endl;
    }

  return os;
}


template<typename VertexType, typename EdgeType>
Graph<VertexType, EdgeType>::Graph(bool directed_):
    directed(directed_) {
}

template<typename VertexType, typename EdgeType>
int Graph<VertexType, EdgeType>::vertexCount() const {
  return vertex.size();
}

template<typename VertexType, typename EdgeType>
int Graph<VertexType, EdgeType>::edgeCount() const {
  int result = 0;

  for (int i = 0; i < vertex.size(); i++)
    for (typename EdgeList::const_iterator j = vertex[i].second.begin(); j != vertex[i].second.end(); j++)
      result += j->second.size();

  if (!directed)
    result = result / 2;

  return result;
}

template<typename VertexType, typename EdgeType>
bool Graph<VertexType, EdgeType>::isEmpty() const {
  return vertex.size() == 0;
}

template<typename VertexType, typename EdgeType>
bool Graph<VertexType, EdgeType>::isDirected() const {
  return directed;
}

template<typename VertexType, typename EdgeType>
int Graph<VertexType, EdgeType>::insert(const VertexType &v) {
  for (int i = 0; i < vertex.size(); i++)
    if (vertex[i].first == v)
      return i;

  vertex.push_back(std::make_pair(v, EdgeList()));
  return vertex.size() - 1;
}


template<typename VertexType, typename EdgeType>
void Graph<VertexType, EdgeType>::connect(int v1, int v2, const EdgeType &e) {
  assert(v1 < vertex.size() && v2 < vertex.size());

  vertex[v1].second[v2].push_back(e);;
  if (!directed)
    vertex[v2].second[v1].push_back(e);
}

template<typename VertexType, typename EdgeType>
void Graph<VertexType, EdgeType>::connect(int v1, int v2, const std::vector<EdgeType> &e) {
  assert(v1 < vertex.size() && v2 < vertex.size());

  if (e.size() == 0)
    return;

  copy(e.begin(), e.end(), back_inserter(vertex[v1].second[v2]));
  if (!directed)
    copy(e.begin(), e.end(), back_inserter(vertex[v2].second[v1]));
}

template<typename VertexType, typename EdgeType>
void Graph<VertexType, EdgeType>::disconnect(int v1, int v2) {
  assert(v1 < vertex.size() && v2 < vertex.size());

  vertex[v1].second.erase(v2);
  if (!directed)
    vertex[v2].second.erase(v1);
}

template<typename VertexType, typename EdgeType>
bool Graph<VertexType, EdgeType>::hasEdge(int v1, int v2) const {
  return vertex[v1].second.find(v2) != vertex[v1].second.end();
}

template<typename VertexType, typename EdgeType>
std::vector<EdgeType> Graph<VertexType, EdgeType>::getEdge(int v1, int v2) const {
  if (!hasEdge(v1, v2))
    return std::vector<EdgeType>();

  return vertex[v1].second.find(v2)->second;
}

// This topological sort does handle SCC in graph.
template<typename VertexType, typename EdgeType>
std::vector<std::set<int> > Graph<VertexType, EdgeType>::topoSort() const {
  const int n = vertex.size();
  std::vector<GraphColorType> color(n, WHITE);
  std::stack<int> S;

  std::vector<int> order(n);
  int c = n;

  // first DFS
  for (int i = n - 1; i >= 0; i--)
    if (color[i] == WHITE) {
      S.push(i);
      while (!S.empty()) {
        int v = S.top();

        if (color[v] == WHITE) {
          for (typename EdgeList::const_iterator j = vertex[v].second.begin(); j != vertex[v].second.end(); j++)
            if (color[j->first] == WHITE)
              S.push(j->first);

          color[v] = GREY;
        } else if (color[v] == GREY) {
          color[v] = BLACK;
          S.pop();
          order[--c] = v;
        } else {
          S.pop();
        }
      }
    }

  // transpose edge
  std::vector<std::set<int> > edgeT(n);
  for (int i = 0; i < n; i++)
    for (typename EdgeList::const_iterator j = vertex[i].second.begin(); j != vertex[i].second.end(); j++)
      edgeT[j->first].insert(i);

  // second DFS in transposed graph starting from last finished vertex
  fill(color.begin(), color.end(), WHITE);
  std::vector<std::set<int> > result;
  for (int i = 0; i < n; i++)
    if (color[order[i]] == WHITE) {
      std::set<int> s;

      S.push(order[i]);
      while (!S.empty()) {
        int v = S.top();

        if (color[v] == WHITE) {
          for (std::set<int>::const_iterator j = edgeT[v].begin(); j != edgeT[v].end(); j++)
            if (color[*j] == WHITE)
              S.push(*j);

          color[v] = GREY;
        } else if (color[v] == GREY) {
          color[v] = BLACK;
          S.pop();
          s.insert(v);
        } else {
          S.pop();
        }
      }

      result.push_back(s);
    }

  return result;
}

// This topological sort does not handle SCC in graph.
template<typename VertexType, typename EdgeType>
std::vector<std::set<int> > Graph<VertexType, EdgeType>::packed_topoSort() const {
  const int n = vertex.size();
  std::vector<GraphColorType> color(n, WHITE);
  std::vector<int> cnt(n, 0);

  // Scan edges
  for (auto v: vertex)
    for (auto e: v.second)
      ++cnt[e.first];

  // Scan root
  std::vector<std::set<int> > result;
  std::set<int> s;
  for (int i = 0; i < n; i++)
    if (!cnt[i])
      s.insert(i);

  // Toposort
  while (s.size()) {
    result.push_back(s);
    s.clear();
    for (auto i: result.back())
      for (auto e: vertex[i].second)
        if (--cnt[e.first] == 0)
          s.insert(e.first);
  }

  return result;
}

#endif
