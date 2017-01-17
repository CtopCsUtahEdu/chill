#if ! defined _reach_h
#define _reach_h 1

namespace omega {

/**
 * @brief Reachability of a graph
 *
 * Given a tuple set for each node representing starting states at each node. The library can compute which node are
 * reachable and values the tuples can take on.
 */
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
