#include "graph.hh"

using std::cout;
using std::endl;
template<typename T>
struct A {
};

template struct Graph<Empty,Empty>;

int main() {
  Graph<> g;
  
  for (int i = 0; i < 8; i++)
    g.insert();
  
  std::vector<Empty> t;
  t.push_back(Empty());
  t.push_back(Empty());
  
  g.connect(0,1);
  g.connect(1,4);
  g.connect(4,0);
  g.connect(4,5);
  g.connect(1,5);
  g.connect(1,2);
  g.connect(2,3);
  g.connect(3,2);
  g.connect(2,6);
  g.connect(5,6);
  g.connect(6,5);
  g.connect(6,7);
  g.connect(3,7);
  g.connect(7,7,t);
  
  g.insert();
  g.insert();
  g.connect(9,8);
  g.connect(8,0);
  
  cout << "Graph #1:" << endl;
  cout << g;
  
  std::vector<std::set<int> > r = g.topoSort();
  
  cout << "topological order: ";
  int num_scc = 0;
  for (int i = 0; i < r.size(); i++) {
    if (i != 0)
      cout << ' ';
    if (r[i].size() > 1) {
      cout << '(';
      num_scc++;
    }
    for (std::set<int>::iterator j = r[i].begin(); j != r[i].end(); j++) {
      if (j != r[i].begin())
        cout << ' ';
      cout << (*j+1);
    }
    if (r[i].size() > 1)
      cout << ')';
  }
  cout << endl;
  cout << "total number of SCC: " << num_scc << endl;
  
  Graph<> g2;
  
  for (int i = 0; i < 6; i++)
    g2.insert();
  
  g2.connect(0,1);
  g2.connect(0,2);
  g2.connect(3,4);
  g2.connect(3,5);
  g2.connect(3,2);
  g2.connect(5,0);
  
  cout << endl << "Graph #2:" << endl;
  cout << g2;
  
  std::vector<std::set<int> > r2 = g2.packed_topoSort();
  
  cout << "packed topological order: ";
  for (int i = 0; i < r2.size(); i++) {
    if (i != 0)
      cout << ' ';
    if (r2[i].size() > 1)
      cout << '(';
    for (std::set<int>::iterator j = r2[i].begin(); j != r2[i].end(); j++) {
      if (j != r2[i].begin())
        cout << ' ';
      cout << (*j+1);
    }
    if (r2[i].size() > 1)
      cout << ')';
  }
  cout << endl;
  
  Graph<> g3;
  
  for (int i = 0; i < 6; i++)
    g3.insert();
  
  g3.connect(5,2);
  g3.connect(5,3);
  g3.connect(5,4);
  g3.connect(3,1);
  g3.connect(1,0);
  
  cout << endl << "Graph #3:" << endl;
  cout << g3;
  
  std::vector<std::set<int> > r3 = g3.topoSort();
  
  cout << "topological order: ";
  for (int i = 0; i < r3.size(); i++) {
    if (i != 0)
      cout << ' ';
    if (r3[i].size() > 1)
      cout << '(';
    for (std::set<int>::iterator j = r3[i].begin(); j != r3[i].end(); j++) {
      if (j != r3[i].begin())
        cout << ' ';
      cout << (*j+1);
    }
    if (r3[i].size() > 1)
      cout << ')';
  }
  cout << endl;
  
  r3 = g3.packed_topoSort();
  
  cout << "packed topological order: ";
  for (int i = 0; i < r3.size(); i++) {
    if (i != 0)
      cout << ' ';
    if (r3[i].size() > 1)
      cout << '(';
    for (std::set<int>::iterator j = r3[i].begin(); j != r3[i].end(); j++) {
      if (j != r3[i].begin())
        cout << ' ';
      cout << (*j+1);
    }
    if (r3[i].size() > 1)
      cout << ')';
  }
  cout << endl;
}
