#include "graph.hh"

#include "gtest/gtest.h"

void g1(Graph<> &g) {
  Graph<> g1;
  for (int i = 0; i < 8; i++)
    g.insert();
  std::vector<Empty> t;
  t.push_back(Empty());
  t.push_back(Empty());

  g.connect(0, 1);
  g.connect(1, 4);
  g.connect(4, 0);
  g.connect(4, 5);
  g.connect(1, 5);
  g.connect(1, 2);
  g.connect(2, 3);
  g.connect(3, 2);
  g.connect(2, 6);
  g.connect(5, 6);
  g.connect(6, 5);
  g.connect(6, 7);
  g.connect(3, 7);
  g.connect(7, 7, t);

  g.insert();
  g.insert();
  g.connect(9, 8);
  g.connect(8, 0);
}

void g2(Graph<> &g) {
  for (int i = 0; i < 6; i++)
    g.insert();

  g.connect(0, 1);
  g.connect(0, 2);
  g.connect(3, 4);
  g.connect(3, 5);
  g.connect(3, 2);
  g.connect(5, 0);
}

void g3(Graph<> &g) {
  for (int i = 0; i < 6; i++)
    g.insert();

  g.connect(5, 3);
  g.connect(5, 4);
  g.connect(2, 1);
  g.connect(1, 0);
  g.connect(0, 3);
}

void seteq(const std::set<int> &i, const std::set<int> &o, bool check) {
  check &= i.size() == o.size();
  for (auto v: i)
    check &= o.find(v) != o.end();
}

TEST (graph, topoSort) {
  {
    Graph<> g;
    g1(g);
    auto r = g.topoSort();
    bool check = true;
    int ord[] = {9, 8, 0, 1, 4, 2, 3, 5, 6, 7};
    check &= r.size() == 6;
    seteq(r[0], std::set<int>(ord, ord + 1), check);
    seteq(r[1], std::set<int>(ord + 1, ord + 2), check);
    seteq(r[2], std::set<int>(ord + 2, ord + 5), check);
    seteq(r[3], std::set<int>(ord + 5, ord + 7), check);
    seteq(r[4], std::set<int>(ord + 7, ord + 9), check);
    seteq(r[5], std::set<int>(ord + 9, ord + 10), check);
    EXPECT_TRUE(check && "g1");
  }
  {
    Graph<> g;
    g2(g);
    auto r = g.topoSort();
    bool check = true;
    int ord[] = {3, 4, 5, 0, 1, 2};
    check &= r.size() == 6;
    seteq(r[0], std::set<int>(ord, ord + 1), check);
    seteq(r[1], std::set<int>(ord + 1, ord + 2), check);
    seteq(r[2], std::set<int>(ord + 2, ord + 3), check);
    seteq(r[3], std::set<int>(ord + 3, ord + 4), check);
    seteq(r[4], std::set<int>(ord + 4, ord + 5), check);
    seteq(r[5], std::set<int>(ord + 5, ord + 6), check);
    EXPECT_TRUE(check && "g2");
  }
}

TEST (graph, packed_topoSort) {
  {
    Graph<> g;
    g2(g);
    auto r = g.packed_topoSort();
    bool check = true;
    int ord[] = {3, 4, 5, 0, 1, 2};
    check &= r.size() == 4;
    seteq(r[0], std::set<int>(ord, ord + 1), check);
    seteq(r[1], std::set<int>(ord + 1, ord + 3), check);
    seteq(r[2], std::set<int>(ord + 3, ord + 4), check);
    seteq(r[3], std::set<int>(ord + 4, ord + 6), check);
    EXPECT_TRUE(check && "g2");
  }
  {
    Graph<> g;
    g3(g);
    auto r = g.packed_topoSort();
    bool check = true;
    int ord[] = {2, 5, 1, 4, 0, 3};
    check &= r.size() == 4;
    seteq(r[0], std::set<int>(ord, ord + 2), check);
    seteq(r[1], std::set<int>(ord + 2, ord + 4), check);
    seteq(r[2], std::set<int>(ord + 4, ord + 5), check);
    seteq(r[3], std::set<int>(ord + 5, ord + 6), check);
    EXPECT_TRUE(check && "g3");
  }
}
