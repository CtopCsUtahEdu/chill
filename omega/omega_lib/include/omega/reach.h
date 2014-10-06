#if ! defined _reach_h
#define _reach_h 1

namespace omega {

class reachable_information {
public:
	Tuple<std::string> node_names;
	Tuple<int> node_arity;
	Dynamic_Array1<Relation> start_nodes;
	Dynamic_Array2<Relation> transitions;
};


Dynamic_Array1<Relation> *
Reachable_Nodes(reachable_information * reachable_info);

Dynamic_Array1<Relation> *
I_Reachable_Nodes(reachable_information * reachable_info);

} // namespace

#endif
