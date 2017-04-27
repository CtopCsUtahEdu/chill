/*
 * loop_ghostcells.cc
 *
 *  Created on: April 23, 2013
 *  Authors:	Protonu Basu
 */

#include "loop.hh"
#include "chill_error.hh"
#include <omega.h>
#include "omegatools.hh"
#include <string.h>
#include <code_gen/CG_outputRepr.h>

using namespace omega;

bool is_in(int n, std::vector<int> v_);


/* using v2
void Loop::generate_ghostcells(int stmt_num, int level, int ghost_depth)
{
	if (stmt_num < 0 || stmt_num >= stmt.size())
		throw std::invalid_argument(
				"invalid statement number " + to_string(stmt_num));
	if (level < 1 || level > stmt[stmt_num].loop_level.size())
		throw std::invalid_argument("invalid loop level " + to_string(level));

	delete last_compute_cgr_;
	last_compute_cgr_ = NULL;
	delete last_compute_cg_;
	last_compute_cg_ = NULL;


	//Protonu--IMP::implementation dirt
	//iterate over all the statements
	//for each statement before the statement we are dealing with,
	//we find the nesting depth of the statement--DS
	//The level we want to insert the new loop at--L
	//Let the lower bound of the new loop --LL
	//the upper bound of the new lew loop --LU

	//We assume statement position also gives us the lex-order
	//end--



	//Let's write down this algorithm

	//Phase I: Expand Iteration Space
	//For each statement, expect the statement we are concerned with, we add a new relation to generate a new IS 
	//At the desired nesting level, we add a new variable
	//We then map all output variables from 1 to (nesting level-1) to the input variable
	//We then map all output variables from (nesting level + 1) to the (input variable - 1)
	//Let's call this Relation : Relation_phase_I



	//Phase II: Modify the xform for all statments
	//Create a new relation, which takes as input 2*n+1 variables and outputs 2*(n+1)+1 variables
	//In the new relation output varianles from 1 to 2*(nesting level -1 ) map to the input variables of the same index
	//2*(nesting level) -1 output var is set to zero
	//2*(nesting level) is set to 

	int nesting_depth = stmt[stmt_num].loop_level.size();
	std::cout<<"this statement is nested at depth...."<<nesting_depth<<"\n";

	for (int i=0; i<stmt.size(); i++)
	{
		if (i == stmt_num) 
			break;


		//first do statements that come before the statement we are considering
		Relation IS_xpand ( stmt[i].IS.n_set(), stmt[i].IS.n_set()+1);
		F_And *eql = IS_xpand.add_and();

		for (int var=1; var<level; var++)
		{
			EQ_Handle e = eql->add_EQ();
			e.update_coef(IS_xpand.output_var(var),1);
			e.update_coef(IS_xpand.input_var(var),-1);
		
		}
		
		for (int var=1+level; var<= stmt[i].IS.n_set()+1; var++)
		{
			EQ_Handle e = eql->add_EQ();
			e.update_coef(IS_xpand.output_var(var),1);
			e.update_coef(IS_xpand.input_var(var-1),-1);
		
		}

		int l_bnd = 0;
		int u_bnd = ghost_depth-1;

		if (i < stmt_num)
		{
			//Stuff based on the lower bound of the new loop
			EQ_Handle e = eql->add_EQ();
			e.update_coef(IS_xpand.output_var(level),1);
			e.update_const(-1*l_bnd);

		}
		if (i > stmt_num)
		{
			//Stuff based on the upper bound of the new loop
			EQ_Handle e = eql->add_EQ();
			e.update_coef(IS_xpand.output_var(level),1);
			e.update_const(-1*u_bnd);
		}


		//We also modify xform for each statement to reflect
		//the additional dimension

		Relation map_xform_temp = Extend_Domain (copy(stmt[i].xform));
		Relation map_xform = Extend_Range(copy(map_xform_temp), 2);
		



	}




	int stencil_radius = 1;
	int extra_width = ghost_depth -1 ;

	//Get current IS
	//Let's print out the current IS
	stmt[stmt_num].IS.print();
	std::cout<<"the xform is...\n";
	stmt[stmt_num].xform.print();

	//Let's create a new relation
	Relation ghost_loop(stmt[stmt_num].IS.n_set()+1);
	F_And *rt = ghost_loop.add_and();

	GEQ_Handle l_bounds = rt->add_GEQ();
	GEQ_Handle up_bounds = rt->add_GEQ();
	l_bounds.update_coef(ghost_loop.set_var(1), 1);
	up_bounds.update_coef(ghost_loop.set_var(1), -1);
	up_bounds.update_const(ghost_depth-1);

	ghost_loop.name_set_var(1, "s");

	ghost_loop.print();

	//Let's try iterating over the IS
	for (DNF_Iterator di(stmt[stmt_num].IS.query_DNF()); di; di++)
	{
		//std::cout<<"in the next conjunct\n";
		for(GEQ_Iterator gi = (*di)->GEQs(); gi; gi++)
		{
			//std::cout<<"in the next GEQ..\n";
			GEQ_Handle spaces = rt->add_GEQ();
			//get the coeff
			for ( int lp=1; lp<=3; lp++)
			{
				coef_t bound = (*gi).get_coef(stmt[stmt_num].IS.set_var(lp));
				if (bound)
				{
					if(bound > 0)
					{
						//find out the lower bound here
						for(Constr_Vars_Iter cvi(*gi); cvi; cvi++)
						{
							Variable_ID _v = cvi.curr_var();
							 //if ( _v->kind() == Input_Var ) std::cout<<"woohoo\n";
							spaces.update_coef(ghost_loop.set_var(((*cvi).var->get_position())+1), (*cvi).coef);
							ghost_loop.name_set_var(((*cvi).var->get_position())+1, (*cvi).var->name());
						}
						spaces.update_coef(ghost_loop.set_var(1), -1);
						spaces.update_const(extra_width);
					}

					if (bound < 0)
					{
						//find out the upper bound here
						for(Constr_Vars_Iter cvi(*gi); cvi; cvi++)
						{
							Variable_ID _v = cvi.curr_var();
							 if ( _v->kind() == Input_Var )
								spaces.update_coef(ghost_loop.set_var(((*cvi).var->get_position())+1), (*cvi).coef);
							 if ( _v->kind() == Global_Var )
							 {
								Global_Var_ID g = _v->get_global_var();
  								Variable_ID v3;
        							if (g->arity() == 0)
          								v3 = ghost_loop.get_local(g);
        							else
          								v3 = ghost_loop.get_local(g, _v->function_of());
								
								spaces.update_coef(v3, cvi.curr_coef());
	
							 } 
						}
						spaces.update_coef(ghost_loop.set_var(1), -1);
						spaces.update_const(extra_width);

					}

				}
			}
	
		}

		ghost_loop.print();
		for(EQ_Iterator ei = (*di)->EQs(); ei; ei++)
		{
			//EQ_Handle cnstr = rt->add_EQ();

			F_Exists *f_exists = rt->add_exists();
			EQ_Handle cnstr = f_exists->add_and()->add_EQ();

			for(Constr_Vars_Iter cvi(*ei); cvi; cvi++)
			{
				Variable_ID _v = cvi.curr_var();
				std::cout<<"variable..."<<(*cvi).var->name()<<"....coefff.."<<(*cvi).coef<<"\n";
				if (_v->kind() == Input_Var)
				{
					cnstr.update_coef(ghost_loop.set_var(((*cvi).var->get_position())+1), (*cvi).coef);
				}
				if (_v->kind() == Global_Var) 
				{
					Global_Var_ID g = _v->get_global_var();
					Variable_ID v3;
					if (g->arity() == 0)
						v3 = ghost_loop.get_local(g);
					else
						v3 = ghost_loop.get_local(g, _v->function_of());

					cnstr.update_coef(v3, cvi.curr_coef());

				}
				if ( _v->kind() == Wildcard_Var ) 
				{
					Variable_ID wc3 = f_exists->declare();
					cnstr.update_coef(wc3, cvi.curr_coef());
				}

			}

		}


			
		std::cout<<"printing out ghost loops...\n\n";
		ghost_loop.setup_names();
		ghost_loop.simplify();
		ghost_loop.print();
		std::cout<<"\n";

	}


		stmt[stmt_num].IS = copy(ghost_loop);


		Relation n = Extend_Domain(copy (stmt[stmt_num].xform));
		Relation nn = Extend_Range(copy(n), 2);

		Relation new_nn(nn.n_inp(), nn.n_out());
		F_And *mp = new_nn.add_and();
		EQ_Handle cmn = mp->add_EQ();
		cmn.update_coef(new_nn.output_var(n.n_out()+1), 1);

		cmn = mp->add_EQ();
		cmn.update_coef(new_nn.output_var(n.n_out()+2), 1);
		cmn.update_coef(new_nn.input_var(n.n_inp()), -1);
		std::cout<<"new_nn.......\n";
		new_nn.print();

		Relation _nn = Intersection(copy(nn), new_nn);

		_nn.print();


		//now let's modify the x-form
		Relation new_xform (_nn.n_out(), _nn.n_out());
		mp = new_xform.add_and();

		cmn =  mp->add_EQ();

		cmn.update_coef(new_xform.output_var(1), 1);
		cmn.update_coef(new_xform.input_var(_nn.n_out()-1), -1);
			
		cmn =  mp->add_EQ();
		cmn.update_coef(new_xform.output_var(2), 1);
		cmn.update_coef(new_xform.input_var(_nn.n_out()), -1);

		for(int i=3; i<=new_xform.n_out(); i++)
		{
			EQ_Handle cmn = mp->add_EQ();
			cmn.update_coef(new_xform.output_var(i), 1);
			cmn.update_coef(new_xform.input_var(i-2), -1);

		}
		//new_xform.print();

		Relation m = Composition(new_xform, copy(_nn));
		m.simplify();
		std::cout<<"m is....\n";
		m.print();

		Relation map_input (stmt[stmt_num].xform.n_inp()+1, stmt[stmt_num].xform.n_inp()+1);
		mp = map_input.add_and();
		cmn =  mp->add_EQ();
		cmn.update_coef(map_input.output_var(map_input.n_out()), 1);
		cmn.update_coef(map_input.input_var(1), -1);

		for(int i=1; i<map_input.n_out(); i++)
		{
			cmn =  mp->add_EQ();
			cmn.update_coef(map_input.output_var(i), 1);
			cmn.update_coef(map_input.input_var(i+1), -1);

		}

		std::cout<<"map input\n";
		map_input.print();

		Relation weird = Composition(copy(m), map_input);
		weird.simplify();
		std::cout<<"weird..\n";
		weird.print();



		stmt[stmt_num].xform = weird;
}
*/





bool is_in(int n, std::vector<int> v_)
{
	for(int i=0; i<v_.size(); i++)
	{
		if(n == v_[i]) return true;
	}	
	return false;
}





void Loop::generate_ghostcells_v2(std::vector<int> stmt_num, int level, int ghost_depth, int hold_inner_loop_constant)
{
	for(int ii=0; ii<stmt_num.size(); ii++)
	{
	if (stmt_num[ii] < 0 || stmt_num[ii] >= stmt.size())
		throw std::invalid_argument(
				"invalid statement number " + to_string(stmt_num[ii]));
	if (level < 1 || level > stmt[stmt_num[ii]].loop_level.size())
		throw std::invalid_argument("invalid loop level " + to_string(level));
	}

	delete last_compute_cgr_;
	last_compute_cgr_ = NULL;
	delete last_compute_cg_;
	last_compute_cg_ = NULL;


	//Protonu--IMP::implementation dirt
	//iterate over all the statements
	//for each statement before the statement we are dealing with,
	//we find the nesting depth of the statement--DS
	//The level we want to insert the new loop at--L
	//Let the lower bound of the new loop --LL
	//the upper bound of the new lew loop --LU

	//We assume statement position also gives us the lex-order
	//end--



	//Let's write down this algorithm

	//Phase I: Expand Iteration Space
	//For each statement, expect the statement we are concerned with, we add a new relation to generate a new IS 
	//At the desired nesting level, we add a new variable
	//We then map all output variables from 1 to (nesting level-1) to the input variable
	//We then map all output variables from (nesting level + 1) to the (input variable - 1)
	//Let's call this Relation : Relation_phase_I



	//Phase II: Modify the xform for all statments
	//Create a new relation, which takes as input 2*n+1 variables and outputs 2*(n+1)+1 variables
	//In the new relation output varianles from 1 to 2*(nesting level -1 ) map to the input variables of the same index
	//2*(nesting level) -1 output var is set to zero
	//2*(nesting level) is set to 

	int nesting_depth = stmt[stmt_num[0]].loop_level.size();
	//std::cout<<"this statement is nested at depth...."<<nesting_depth<<"\n";

	int stencil_radius = 1;
	int extra_width = ghost_depth -1 ;

	int s_num;


	for (int i=0; i<stmt.size(); i++)
	{



		if (!is_in(i, stmt_num)){


			//first do statements that come before the statement we are considering
			Relation IS_xpand ( stmt[i].IS.n_set(), stmt[i].IS.n_set()+1);
			F_And *eql = IS_xpand.add_and();

			for (int var=1; var<level; var++)
			{
				EQ_Handle e = eql->add_EQ();
				e.update_coef(IS_xpand.output_var(var),1);
				e.update_coef(IS_xpand.input_var(var),-1);

			}

			for (int var=1+level; var<= stmt[i].IS.n_set()+1; var++)
			{
				EQ_Handle e = eql->add_EQ();
				e.update_coef(IS_xpand.output_var(var),1);
				e.update_coef(IS_xpand.input_var(var-1),-1);

			}

			int l_bnd = 0;
			int u_bnd = ghost_depth-1;

			if (i < stmt_num[0])
			{
				//Stuff based on the lower bound of the new loop
				EQ_Handle e = eql->add_EQ();
				e.update_coef(IS_xpand.output_var(level),1);
				e.update_const(-1*l_bnd);

			}
			if (i > stmt_num[0])
			{
				//Stuff based on the upper bound of the new loop
				EQ_Handle e = eql->add_EQ();
				e.update_coef(IS_xpand.output_var(level),1);
				e.update_const(-1*u_bnd);
			}

			stmt[i].IS = Composition(IS_xpand, copy(stmt[i].IS));
		}

		else {
			//if (i == stmt_num)

			s_num = i;

			LoopLevel _new;
		        _new.type= stmt[i].loop_level[nesting_depth -1].type;
			//Protonu--this is probably buggy, have to fix it later
		        _new.payload= stmt[i].loop_level[nesting_depth -1].payload+1;
			//end.
		        _new.parallel_level = stmt[i].loop_level[nesting_depth -1].parallel_level;
			stmt[i].loop_level.push_back(_new);

			//Get current IS
			//Let's print out the current IS
			std::cout<<"statement number::"<<s_num<<"IS::\n\n";
			stmt[s_num].IS.print();
			std::cout<<"the xform is...\n";
			stmt[s_num].xform.print();

			//Let's create a new relation
			Relation ghost_loop(stmt[s_num].IS.n_set()+1);
			F_And *rt = ghost_loop.add_and();

			GEQ_Handle l_bounds = rt->add_GEQ();
			GEQ_Handle up_bounds = rt->add_GEQ();
			l_bounds.update_coef(ghost_loop.set_var(1), 1);
			up_bounds.update_coef(ghost_loop.set_var(1), -1);
			up_bounds.update_const(ghost_depth-1);

			ghost_loop.name_set_var(1, "s");

			ghost_loop.print();

			//Let's try iterating over the IS
			for (DNF_Iterator di(stmt[s_num].IS.query_DNF()); di; di++)
			{
				//std::cout<<"in the next conjunct\n";
				for(GEQ_Iterator gi = (*di)->GEQs(); gi; gi++)
				{
					//std::cout<<"in the next GEQ..\n";
					GEQ_Handle spaces = rt->add_GEQ();
					//get the coeff
					//I should change this...
					//we should start from the desired level and go to the nesting level
					//for (int lp=1; lp<=3; lp++)
					for (int lp=level; lp<=nesting_depth; lp++)
					{
						coef_t bound = (*gi).get_coef(stmt[s_num].IS.set_var(lp));
						if (bound)
						{
							if(bound > 0)
							{
								//find out the lower bound here
								for(Constr_Vars_Iter cvi(*gi); cvi; cvi++)
								{
									Variable_ID _v = cvi.curr_var();
									//if ( _v->kind() == Input_Var ) std::cout<<"woohoo\n";
									spaces.update_coef(ghost_loop.set_var(((*cvi).var->get_position())+1), (*cvi).coef);
									ghost_loop.name_set_var(((*cvi).var->get_position())+1, (*cvi).var->name());
								}

								if(lp != nesting_depth || (lp == nesting_depth && !hold_inner_loop_constant))
									spaces.update_coef(ghost_loop.set_var(1), -1);
								spaces.update_const(extra_width);
							}

							if (bound < 0)
							{
								//find out the upper bound here
								for(Constr_Vars_Iter cvi(*gi); cvi; cvi++)
								{
									Variable_ID _v = cvi.curr_var();
									if ( _v->kind() == Input_Var )
										spaces.update_coef(ghost_loop.set_var(((*cvi).var->get_position())+1), (*cvi).coef);
									if ( _v->kind() == Global_Var )
									{
										Global_Var_ID g = _v->get_global_var();
										Variable_ID v3;
										if (g->arity() == 0)
											v3 = ghost_loop.get_local(g);
										else
											v3 = ghost_loop.get_local(g, _v->function_of());

										spaces.update_coef(v3, cvi.curr_coef());

									} 
								}
								if(lp != nesting_depth || (lp == nesting_depth && !hold_inner_loop_constant))
									spaces.update_coef(ghost_loop.set_var(1), -1);
								spaces.update_const(extra_width-1);

							}

						}
					}

				}

				ghost_loop.print();
				for(EQ_Iterator ei = (*di)->EQs(); ei; ei++)
				{
					//EQ_Handle cnstr = rt->add_EQ();

					F_Exists *f_exists = rt->add_exists();
					EQ_Handle cnstr = f_exists->add_and()->add_EQ();

					int cnst = (*ei).get_const();
					//std::cout<< "const is:\n"<<cnst;


					for(Constr_Vars_Iter cvi(*ei); cvi; cvi++)
					{
						Variable_ID _v = cvi.curr_var();
					
						std::cout<<"variable..."<<(*cvi).var->name()<<"....coefff.."<<(*cvi).coef<<"\n";
						if (_v->kind() == Input_Var)
						{
							cnstr.update_coef(ghost_loop.set_var(((*cvi).var->get_position())+1), (*cvi).coef);
						}
						if (_v->kind() == Global_Var) 
						{
							Global_Var_ID g = _v->get_global_var();
							Variable_ID v3;
							if (g->arity() == 0)
								v3 = ghost_loop.get_local(g);
							else
								v3 = ghost_loop.get_local(g, _v->function_of());

							cnstr.update_coef(v3, cvi.curr_coef());

						}
						if ( _v->kind() == Wildcard_Var ) 
						{
							Variable_ID wc3 = f_exists->declare();
							cnstr.update_coef(wc3, cvi.curr_coef());
						}

					}

					//add for the new loop
					cnstr.update_coef(ghost_loop.set_var(1), -1);
					if(cnst != 0)cnstr.update_const(cnst);


				}



				std::cout<<"printing out ghost loops...\n\n";
				ghost_loop.setup_names();
				ghost_loop.simplify();
				ghost_loop.print();
				std::cout<<"\n";

			}


			stmt[s_num].IS = copy(ghost_loop);
		}
			

		//debug
		std::cout<<"stmt number:   "<<i<<"  modified iteration space\n";
		stmt[i].IS.print();




		//Phase(II): Create a relation which takes the iteration space as input
		//and permutes it such that the new loop variable is the one at the end

		Relation permute_IS(stmt[i].IS.n_set(), stmt[i].IS.n_set());
		F_And *_rt = permute_IS.add_and();
		EQ_Handle eql;

		for(int j=1; j<level; j++)
		{
			eql = _rt->add_EQ();
			eql.update_coef(permute_IS.output_var(j), 1);
			eql.update_coef(permute_IS.input_var(j), -1);
		}

		for(int j=level; j<stmt[i].IS.n_set(); j++)
		{
			eql = _rt->add_EQ();
			eql.update_coef(permute_IS.output_var(j), 1);
			eql.update_coef(permute_IS.input_var(j+1), -1);
		}

		eql = _rt->add_EQ();
		eql.update_coef(permute_IS.output_var(stmt[i].IS.n_set()), 1);
		eql.update_coef(permute_IS.input_var(level), -1);

		//Dbg
		std::cout<<"\n stmt number...."<<i<<" the permute_IS relation is....."<<std::endl;
		permute_IS.print();
		//End--Phase(II)

		//Phase(III):
		//Expand the Domain of xform by one for the new loop
		//Expand the Range of xform by two, for the new loop and it's aux loop
		//Set the last two vars added to zero and the input last variable respectively

		Relation n_form = Extend_Domain(copy (stmt[s_num].xform));
		Relation nn_form = Extend_Range(copy(n_form), 2);


		Relation new_nn(nn_form.n_inp(), nn_form.n_out());
		F_And *mp = new_nn.add_and();
		EQ_Handle cmn = mp->add_EQ();
		cmn.update_coef(new_nn.output_var(n_form.n_out()+1), 1);

		cmn = mp->add_EQ();
		cmn.update_coef(new_nn.output_var(n_form.n_out()+2), 1);
		cmn.update_coef(new_nn.input_var(n_form.n_inp()), -1);

		Relation _nn_xform = Intersection(copy(nn_form), new_nn);
		
		//Dbg
		std::cout<<"\n stmt number...."<<i<<"_nn_xform ..before final permute is:\n";
		_nn_xform.print();


		//Phase(IV):
		//Permute the xform, so that the added loops come at the right place
		//For 1 to 2*(level-1) output equals input
		//For 2*(level)-1 and 2*level, output equal input's n_out()-1, n_out()

		Relation permute_xform (_nn_xform.n_out(), _nn_xform.n_out());
		F_And *nd = permute_xform.add_and();

		for(int j=1; j<=2*(level-1); j++)
		{
			EQ_Handle cmn = nd->add_EQ();
			cmn.update_coef(permute_xform.output_var(j), 1);
			cmn.update_coef(permute_xform.input_var(j), -1);

		}

		EQ_Handle pm = nd->add_EQ();
		pm.update_coef(permute_xform.output_var(2*(level)-1), 1);
		pm.update_coef(permute_xform.input_var(permute_xform.n_inp()-1), -1);

		pm = nd->add_EQ();
		pm.update_coef(permute_xform.output_var(2*(level)), 1);
		pm.update_coef(permute_xform.input_var(permute_xform.n_inp()), -1);

		for(int j= 2*(level)+1 ; j<=permute_xform.n_inp(); j++)
		{
			EQ_Handle cmn = nd->add_EQ();
			cmn.update_coef(permute_xform.output_var(j), 1);
			cmn.update_coef(permute_xform.input_var(j-2), -1);
		}

		Relation _temp = Composition(copy(permute_xform), copy(_nn_xform));
		_temp.print();

		stmt[s_num].xform =  Composition (_temp, permute_IS);
			
	}

	//updating the nesting information of the loop
	//this is required, when the dependence graph is 
	//rebuilt
	num_dep_dim++;


	//Phase(V):
	//I need to fix loop_level here


	
	//Phase(VI): 
	//update the dependence graph
	
	DependenceGraph g(stmt[s_num].IS.n_set());

	for(int i=0; i<stmt.size(); i++) 
	g.insert();

	

	for (int i = 0; i < stmt.size(); i++)
    for (int j = i; j < stmt.size(); j++) {
      std::pair<std::vector<DependenceVector>,
                std::vector<DependenceVector> > dv = test_data_dependences(this, ir, stmt[i].code, stmt[i].IS, stmt[j].code, stmt[j].IS,
                                                                           freevar, index, stmt_nesting_level_[i],
                                                                           stmt_nesting_level_[j],
                                                                           uninterpreted_symbols[ i ],  // ??? 
                                                                           uninterpreted_symbols_stringrepr[ i ], unin_rel[i], dep_relation); // ??? );
      
      

		for (int k = 0; k < dv.first.size(); k++) {

			if (is_dependence_valid_based_on_lex_order(i, j, dv.first[k],
						true))
				g.connect(i, j, dv.first[k]);
			else {
				g.connect(j, i, dv.first[k].reverse());
			}
		}
		for (int k = 0; k < dv.second.size(); k++)
			if (is_dependence_valid_based_on_lex_order(j, i, dv.second[k],
						false))
				g.connect(j, i, dv.second[k]);
			else {
				g.connect(i, j, dv.second[k].reverse());
			}
	}

	dep = g;


}


/*********************************************/
/***Code Generation for OMP parallel region***/
/*********************************************/
/* 
void Loop::mark_omp_parallel_region(int lvl)
{
	//should do a check to make sure 0 <= lvl <= max_nesting
	loop_for_omp_parallel_region = lvl;
	return;
}

void Loop::mark_omp_threads(int loop)
{
	omp_threads.push_back(loop);
	return;
}

void Loop::mark_omp_syncs(int lvl, std::vector<int> thrds)
{
	omp_syncs.insert(std::pair<int, std::vector<int> >(lvl, thrds));
	return;
}

bool Loop::generate_omp_parallel_region(int use_barrier, int num_omp_threads)
{
	delete last_compute_cgr_;
	last_compute_cgr_ = NULL;
	delete last_compute_cg_;
	last_compute_cg_ = NULL;

	loop_for_omp_parallel_region = 1;
	//1 for OMP BARRIER
	//0 for explicit locks
	use_omp_barrier = use_barrier;
	omp_thrds_to_use = num_omp_threads;

	return true;
}


CG_outputRepr *Loop::add_omp_thread_info(CG_outputRepr *repr)const
{

	SgSymbolTable* parameter_symtab;
	SgSymbolTable* body_symtab;
	SgSymbolTable* root_symtab;

	std::vector<SgSymbolTable *> symtabs = ((IR_roseCode *) ir)->getsymtabs();

	root_symtab = symtabs[0];
	parameter_symtab = symtabs[1];
	body_symtab = symtabs[2];


	SgFunctionDeclaration * fn= ((IR_roseCode *) ir)->get_func();
	SgScopeStatement* func_body = fn->get_definition()->get_body();


	//try "tid" adding to the body
	SgVariableDeclaration *defn = buildVariableDeclaration("tid", buildIntType());
	SgInitializedNamePtrList& variables = defn->get_variables();
	SgInitializedNamePtrList::const_iterator j = variables.begin();
	SgInitializedName* initializedName = *j;
	SgVariableSymbol* dvs = new SgVariableSymbol(initializedName);
	dvs->set_parent(body_symtab);
	body_symtab->insert("tid", dvs );

	//adding "num_threads" to the body
	SgVariableDeclaration *num_defn = buildVariableDeclaration("num_threads", buildIntType());
	SgInitializedNamePtrList& vbls = num_defn->get_variables();
	SgInitializedNamePtrList::const_iterator k = vbls.begin();
	SgInitializedName* _Name = *k;
	SgVariableSymbol* new_dvs = new SgVariableSymbol(_Name);
	new_dvs->set_parent(body_symtab);
	body_symtab->insert("num_threads", new_dvs );


	// Adding left, right variables to the body  
	// This has to be modified, once we starting 
	// mapping more than one dimension per thread

	SgVariableDeclaration *left_defn = buildVariableDeclaration("left", buildIntType());
	SgInitializedNamePtrList& left_vars = left_defn->get_variables();
	SgInitializedNamePtrList::const_iterator lft = left_vars.begin();
	SgInitializedName* _Left = *lft;
	SgVariableSymbol* left_dvs = new SgVariableSymbol(_Left);
	left_dvs->set_parent(body_symtab);
	body_symtab->insert("left", left_dvs );

	SgVariableDeclaration *right_defn = buildVariableDeclaration("right", buildIntType());
	SgInitializedNamePtrList& right_vars = right_defn->get_variables();
	SgInitializedNamePtrList::const_iterator rght = right_vars.begin();
	SgInitializedName* _Right = *rght;
	SgVariableSymbol* right_dvs = new SgVariableSymbol(_Right);
	right_dvs->set_parent(body_symtab);
	body_symtab->insert("right", right_dvs );



	//We should create an array of locks
	//TO DO: the dimension of this array should depend on the input
	SgType * tp = new SgTypeInt();
	SgModifierType *vol_tp = buildVolatileType(tp);
	SgVariableDeclaration *locks_defn = buildVariableDeclaration("zplanes", buildArrayType(vol_tp, buildIntVal(256)));


	SgInitializedNamePtrList& _variables = locks_defn->get_variables();
	SgInitializedNamePtrList::const_iterator _j = _variables.begin();
	SgInitializedName* _initializedName = *_j;
	SgVariableSymbol* lcks = new SgVariableSymbol(_initializedName);
	lcks->set_parent(body_symtab);
	body_symtab->insert("zplanes", lcks );


	//I should create a for-loop which zeroes out the locks

	// create the induction variable idx 
	SgVariableDeclaration *idx_defn = buildVariableDeclaration("idx", buildIntType());
	SgInitializedNamePtrList& vrs = idx_defn->get_variables();
	SgInitializedNamePtrList::const_iterator __j = vrs.begin();
	SgInitializedName* __initializedName = *__j;
	SgVariableSymbol* ds = new SgVariableSymbol(__initializedName);
	ds->set_parent(body_symtab);
	body_symtab->insert("idx", ds );


	//Creating the for loop to intialize the array of locks 

	SgExpression* lower_bound = isSgExpression(buildIntVal(0));
	SgExpression* upper_bound = isSgExpression(buildIntVal(255));
	SgExpression* step_size = isSgExpression(buildIntVal(1));

	SgVarRefExp* idx_sym = buildVarRefExp(ds);
	SgStatement* for_init_stmt = buildAssignStatement(idx_sym, lower_bound);
	SgLessOrEqualOp* cond = buildLessOrEqualOp(idx_sym, upper_bound);
	SgExprStatement* test = buildExprStatement(cond);
	SgPlusAssignOp* increment = buildPlusAssignOp(idx_sym, step_size);
	SgForStatement *for_stmt = buildForStatement(for_init_stmt,
			isSgStatement(test), increment, NULL);


	
	SgVarRefExp* _sym = buildVarRefExp(lcks);
	SgExpression* new_sym = buildPntrArrRefExp (_sym, idx_sym);
	SgStatement* sample = buildAssignStatement(new_sym, isSgExpression(buildIntVal(-10)));
	SgBasicBlock* bdy = buildBasicBlock();
	bdy->set_parent(for_stmt);
	bdy->append_statement(sample);
	for_stmt->set_loop_body(bdy);

	//for -loop
	if(!use_omp_barrier)prependStatement(for_stmt, func_body);

	//variable declarations
	prependStatement(idx_defn, func_body);
	prependStatement(locks_defn, func_body);
	prependStatement(defn, func_body);
	prependStatement(num_defn, func_body);
	prependStatement(left_defn, func_body);
	prependStatement(right_defn, func_body);

	//End--setting up stuff in the body of the function

	//building the statement "tid = omp_get_thread_num();"
	SgGlobal *globals = ((IR_roseCode *) ir)->get_root();
	SgName name_omp_get_thread_num("omp_get_thread_num");
	SgFunctionDeclaration * decl_omp_thread_id =
		buildNondefiningFunctionDeclaration(name_omp_get_thread_num,
				buildIntType(), buildFunctionParameterList(), globals);

	SgExprListExp* args = buildExprListExp();
	SgFunctionCallExp *the_call = buildFunctionCallExp(
			buildFunctionRefExp(decl_omp_thread_id), args);

	SgExprStatement* __stmt = buildExprStatement(the_call);
	SgVarRefExp *var = new SgVarRefExp(dvs);
	SgExprStatement* ins = buildAssignStatement(var, the_call);


	//building the statement "num_threads = omp_get_num_threads();"
	SgName name_omp_get_num_threads("omp_get_num_threads");
	SgFunctionDeclaration * decl_omp_num_threads =
		buildNondefiningFunctionDeclaration(name_omp_get_num_threads,
				buildIntType(), buildFunctionParameterList(), globals);

	SgExprListExp* new_args = buildExprListExp();
	SgFunctionCallExp *new_call = buildFunctionCallExp(
			buildFunctionRefExp(decl_omp_num_threads), new_args);

	SgExprStatement* new_call_stmt = buildExprStatement(new_call);
	SgVarRefExp *new_var = new SgVarRefExp(new_dvs);
	SgExprStatement* ins_call = buildAssignStatement(new_var, new_call);


	//Setting up the left and right terms	
	SgVarRefExp *opaque_term_left = buildOpaqueVarRefExp("__rose_gt", ((IR_roseCode *)ir)->get_root());
	SgVarRefExp *opaque_term_right = buildOpaqueVarRefExp("__rose_lt", ((IR_roseCode *)ir)->get_root());
	//build expression tid-1
	SgExpression *left_tid = buildSubtractOp(new SgVarRefExp(dvs),isSgExpression(buildIntVal(1)));
	//build expression tid+1; expression num_threads-1
	SgExpression *right_tid = buildAddOp(new SgVarRefExp(dvs),isSgExpression(buildIntVal(1)));
	SgExpression *num_threads_bnd = buildSubtractOp(new SgVarRefExp(new_dvs), isSgExpression(buildIntVal(1)));
	//build left = max(0, tid-1)
	//build right = min(num_threads-1, tid+1)
	SgExprListExp* arg_list_left = buildExprListExp();
	appendExpression(arg_list_left, left_tid);
	appendExpression(arg_list_left, isSgExpression(buildIntVal(0)));
	SgExpression *left_term_assign = isSgExpression(buildFunctionCallExp(opaque_term_left, arg_list_left));
	SgExprStatement *left_assign = buildAssignStatement(new SgVarRefExp(left_dvs), left_term_assign);


	SgExprListExp* arg_list_right = buildExprListExp();
	appendExpression(arg_list_right, right_tid);
	appendExpression(arg_list_right, num_threads_bnd);
	SgExpression *right_term_assign = isSgExpression(buildFunctionCallExp(opaque_term_right, arg_list_right));
	SgExprStatement *right_assign = buildAssignStatement(new SgVarRefExp(right_dvs), right_term_assign);



	
	// Create the declaration for the call to _mm_pause()
	SgName name_mm_pause("_mm_pause");
	SgFunctionDeclaration * decl_mm_pause =
		buildNondefiningFunctionDeclaration(name_mm_pause,
				buildIntType(), buildFunctionParameterList(), globals);
	SgExprListExp* pause_args = buildExprListExp();
	SgFunctionCallExp *pause_call = buildFunctionCallExp(
			buildFunctionRefExp(decl_mm_pause), pause_args);

	SgExprStatement* pause_call_stmt = buildExprStatement(pause_call);



	//prependStatement(ins, func_body);
	CG_roseRepr * code_loop = (CG_roseRepr *)repr;
	SgStatementPtrList *lst = code_loop->GetList();
	SgNode * _tnl = code_loop->GetCode();


	SgBasicBlock *big_blk = new SgBasicBlock(TRANSFORMATION_FILE_INFO);

	SgVariableSymbol* index_outer;
	std::vector<SgVariableSymbol *> loop_indices;	

	if(lst){printf("oops..\n");}

	if(_tnl){

		if(isSgForStatement(_tnl)) 
		{
			SgStatement* lp_body = isSgForStatement(_tnl)->get_loop_body();
			SgNode *pnt = isSgNode(lp_body)->get_parent();

			// Try and get the symbol for this loop
			SgForInitStatement *new_list = isSgForStatement(_tnl)->get_for_init_stmt();
			SgStatementPtrList& _ins = new_list->get_init_stmt();
			SgStatementPtrList::const_iterator j = _ins.begin();


			if (SgExprStatement *expr = isSgExprStatement(*j))
				if (SgAssignOp* op = isSgAssignOp(expr->get_expression()))
					if (SgVarRefExp* var_ref = isSgVarRefExp(
								op->get_lhs_operand()))
						index_outer = var_ref->get_symbol();


			//Trying to fill in the vector of loop indices
			SgNode * _fors = _tnl;
			SgStatement* lp_body_temp = isSgForStatement(_fors)->get_loop_body();


			while (isSgForStatement(lp_body_temp)){


				SgForInitStatement *n_list = isSgForStatement(_fors)->get_for_init_stmt();
				SgStatementPtrList& n_ins = n_list->get_init_stmt();
				SgStatementPtrList::const_iterator n_j = n_ins.begin();

				if (SgExprStatement *expr = isSgExprStatement(*n_j))
					if (SgAssignOp* op = isSgAssignOp(expr->get_expression()))
						if (SgVarRefExp* var_ref = isSgVarRefExp(
									op->get_lhs_operand()))
							loop_indices.push_back(var_ref->get_symbol());


				_fors = lp_body_temp;
				if ( isSgForStatement(_fors))
					lp_body_temp = isSgForStatement(_fors)->get_loop_body();

				//the last bit, where the body of the loop is no more another for-loop

				if ( !isSgForStatement(lp_body_temp))
				{
					SgForInitStatement *n_list = isSgForStatement(_fors)->get_for_init_stmt();
					SgStatementPtrList& n_ins = n_list->get_init_stmt();
					SgStatementPtrList::const_iterator n_j = n_ins.begin();

					if (SgExprStatement *expr = isSgExprStatement(*n_j))
						if (SgAssignOp* op = isSgAssignOp(expr->get_expression()))
							if (SgVarRefExp* var_ref = isSgVarRefExp(
										op->get_lhs_operand()))
								loop_indices.push_back(var_ref->get_symbol());


				}

				if (!isSgForStatement(lp_body_temp) && isSgBasicBlock(lp_body_temp))
				{
					SgStatementPtrList *_plist =  new SgStatementPtrList();
					SgStatementPtrList::iterator it;
					for (it = (isSgBasicBlock(lp_body_temp)->get_statements()).begin();
							it != (isSgBasicBlock(lp_body_temp)->get_statements()).end(); it++)
					{
						if ( isSgIfStmt(*it) ) {

							SgStatement * tru_bdy = isSgIfStmt(*it)->get_true_body();
							_fors = isSgForStatement(tru_bdy);
							SgStatement *body_temp = isSgForStatement(_fors)->get_loop_body();

							while (isSgForStatement(body_temp)){

								SgForInitStatement *n_list = isSgForStatement(_fors)->get_for_init_stmt();
								SgStatementPtrList& n_ins = n_list->get_init_stmt();
								SgStatementPtrList::const_iterator n_j = n_ins.begin();

								if (SgExprStatement *expr = isSgExprStatement(*n_j))
									if (SgAssignOp* op = isSgAssignOp(expr->get_expression()))
										if (SgVarRefExp* var_ref = isSgVarRefExp(
													op->get_lhs_operand()))
											loop_indices.push_back(var_ref->get_symbol());


								_fors = body_temp;
								if ( isSgForStatement(_fors))
									body_temp = isSgForStatement(_fors)->get_loop_body();

								//the last bit, where the body of the loop is no more another for-loop

								if ( !isSgForStatement(body_temp))
								{
									SgForInitStatement *n_list = isSgForStatement(_fors)->get_for_init_stmt();
									SgStatementPtrList& n_ins = n_list->get_init_stmt();
									SgStatementPtrList::const_iterator n_j = n_ins.begin();

									if (SgExprStatement *expr = isSgExprStatement(*n_j))
										if (SgAssignOp* op = isSgAssignOp(expr->get_expression()))
											if (SgVarRefExp* var_ref = isSgVarRefExp(
														op->get_lhs_operand()))
												loop_indices.push_back(var_ref->get_symbol());


								}

							}

						}

						if (!isSgIfStmt(*it) && isSgForStatement(*it) ) 
						{
							//The other part goes here.....
							{

								//SgStatement * tru_bdy = isSgIfStmt(*it)->get_true_body();
								_fors = isSgForStatement(*it);
								SgStatement *body_temp = isSgForStatement(_fors)->get_loop_body();

								while (isSgForStatement(body_temp)){

									SgForInitStatement *n_list = isSgForStatement(_fors)->get_for_init_stmt();
									SgStatementPtrList& n_ins = n_list->get_init_stmt();
									SgStatementPtrList::const_iterator n_j = n_ins.begin();

									if (SgExprStatement *expr = isSgExprStatement(*n_j))
										if (SgAssignOp* op = isSgAssignOp(expr->get_expression()))
											if (SgVarRefExp* var_ref = isSgVarRefExp(
														op->get_lhs_operand()))
												loop_indices.push_back(var_ref->get_symbol());


									_fors = body_temp;
									if ( isSgForStatement(_fors))
										body_temp = isSgForStatement(_fors)->get_loop_body();

									//the last bit, where the body of the loop is no more another for-loop

									if ( !isSgForStatement(body_temp))
									{
										SgForInitStatement *n_list = isSgForStatement(_fors)->get_for_init_stmt();
										SgStatementPtrList& n_ins = n_list->get_init_stmt();
										SgStatementPtrList::const_iterator n_j = n_ins.begin();

										if (SgExprStatement *expr = isSgExprStatement(*n_j))
											if (SgAssignOp* op = isSgAssignOp(expr->get_expression()))
												if (SgVarRefExp* var_ref = isSgVarRefExp(
															op->get_lhs_operand()))
													loop_indices.push_back(var_ref->get_symbol());


									}

								}

							}
						}					
					}



				}

			}//end while
			
			//Debug:: remove later
			for (int ll=0; ll<loop_indices.size(); ll++)  std::cout<<"the index is..."<<loop_indices[ll]->get_name().str()<<"\n";


			//std::cout<<"the string is..."<<isSgForStatement(_tnl)->get_string_label()<<"\n";

			SgStatement *new_inner_loop = isSgForStatement(lp_body)->get_loop_body();
			//for now...
			//SgStatement *further_loop = isSgForStatement(new_inner_loop)->get_loop_body();


			//Can we modify the new_inner_loop
			if(isSgForStatement(new_inner_loop)){

				//printf("this also is an inner loop...\n");
				 SgForInitStatement *list = isSgForStatement(lp_body)->get_for_init_stmt();
				 SgStatementPtrList& initStatements = list->get_init_stmt();
				j = initStatements.begin();
				const SgVariableSymbol* index;

				if (SgExprStatement *expr = isSgExprStatement(*j))
					if (SgAssignOp* op = isSgAssignOp(expr->get_expression()))
						if (SgVarRefExp* var_ref = isSgVarRefExp(
									op->get_lhs_operand()))
							index = var_ref->get_symbol();

				printf("the name is :%s...\n", index->get_name().str());

				std::vector<SgVarRefExp *> array = substitute(new_inner_loop, index, NULL, isSgNode(symtabs[2]));

				for (int j = 0; j < array.size(); j++)
					array[j]->set_symbol(dvs);

			}

			if( !isSgForStatement(new_inner_loop) && isSgBasicBlock(new_inner_loop) ) 
			{
					SgStatementPtrList *_plist =  new SgStatementPtrList();
					SgStatementPtrList::iterator it;
					for (it = (isSgBasicBlock(new_inner_loop)->get_statements()).begin();
							it != (isSgBasicBlock(new_inner_loop)->get_statements()).end(); it++)
					{
						printf("this also is an inner loop...\n");
						SgForInitStatement *list = isSgForStatement(lp_body)->get_for_init_stmt();
						SgStatementPtrList& initStatements = list->get_init_stmt();
						j = initStatements.begin();
						const SgVariableSymbol* index;

						if (SgExprStatement *expr = isSgExprStatement(*j))
							if (SgAssignOp* op = isSgAssignOp(expr->get_expression()))
								if (SgVarRefExp* var_ref = isSgVarRefExp(
											op->get_lhs_operand()))
									index = var_ref->get_symbol();

						printf("the name is :%s...\n", index->get_name().str());

						std::vector<SgVarRefExp *> array = substitute(*it, index, NULL, isSgNode(symtabs[2]));

						for (int j = 0; j < array.size(); j++)
							array[j]->set_symbol(dvs);
					}

			}			

			//Create a new basic block
			//SgBasicBlock* bb = buildBasicBlock();
			SgBasicBlock* bb = new SgBasicBlock(TRANSFORMATION_FILE_INFO);
			bb->set_parent(pnt);
			bb->append_statement(new_inner_loop);
			new_inner_loop->set_parent(bb);

			//inserting OMP barrier
			//Inserting the OMP barried should be conditional
			//We can either choose to insert the OMP_Barrier
			//Or use specialized locks
			// REMOVING FOR NOW 

			if (use_omp_barrier){
				SgOmpBarrierStatement* omp_barrier= new SgOmpBarrierStatement(TRANSFORMATION_FILE_INFO);
				bb->append_statement(omp_barrier);
			}


			if (!use_omp_barrier){

				//We should put the code for the specialized locks here
				//create the synchronization here

				SgVarRefExp* i_sym = buildVarRefExp(dvs);
				SgVarRefExp* _vl = buildVarRefExp(index_outer);
				SgExpression* new_sym = buildPntrArrRefExp (_sym, i_sym);
				SgStatement* sync_set = buildAssignStatement(new_sym, _vl);

				//create zplanes[left] < t2
				//create zplanes[right] < t2

				SgVarRefExp* l_sym = buildVarRefExp(left_dvs);
				SgExpression *left_access = buildPntrArrRefExp(_sym, l_sym);
				SgExpression *left_check = buildLessThanOp(left_access, _vl);

				SgVarRefExp* r_sym = buildVarRefExp(right_dvs);
				SgExpression *right_access = buildPntrArrRefExp(_sym, r_sym);
				SgExpression *right_check = buildLessThanOp(right_access, _vl);


				//create while(zplanes....)
				SgBasicBlock* left_while_body = new SgBasicBlock(TRANSFORMATION_FILE_INFO);
				left_while_body->append_statement(pause_call_stmt);
				SgStatement *left_while = buildWhileStmt(left_check,left_while_body);

				SgStatement *right_while = buildWhileStmt(right_check,left_while_body);


				//create if (left !=id )

				SgBasicBlock* left_if_true_body = new SgBasicBlock(TRANSFORMATION_FILE_INFO);
				left_if_true_body->append_statement(left_while);
				SgExpression *left_if_check_cond = buildNotEqualOp(l_sym, var);
				SgStatement *left_if_check = buildIfStmt(left_if_check_cond,left_if_true_body, new SgBasicBlock(TRANSFORMATION_FILE_INFO));

				SgBasicBlock* right_if_true_body = new SgBasicBlock(TRANSFORMATION_FILE_INFO);
				right_if_true_body->append_statement(right_while);
				SgExpression *right_if_check_cond = buildNotEqualOp(r_sym, var);
				SgStatement *right_if_check = buildIfStmt(right_if_check_cond,right_if_true_body, new SgBasicBlock(TRANSFORMATION_FILE_INFO));


				//put everything in a big block...
				//Creating the block which will be the parallel section
				//big_blk->set_parent(_tnl->get_parent());

				sync_set->set_parent(bb);
				bb->append_statement(sync_set);

				left_while->set_parent(bb);
				bb->append_statement(left_if_check);

				bb->append_statement(right_if_check);

			}

			//The block bb
			isSgForStatement(_tnl)->set_loop_body(bb);
			//isSgStatement(pnt)->remove_statement(isSgStatement(lp_body));


			ins->set_parent(big_blk);
			ins_call->set_parent(big_blk);
			big_blk->append_statement(ins);
			big_blk->append_statement(ins_call);

			if(!use_omp_barrier){
				big_blk->append_statement(left_assign);
				big_blk->append_statement(right_assign);
			}

			big_blk->append_statement(isSgForStatement(_tnl));
			_tnl->set_parent(big_blk);


		}

		SgOmpPrivateClause *priv =  new SgOmpPrivateClause();
		//add outer-loop id to private					  `
		SgInitializedName *private_arg = new SgInitializedName( index_outer->get_name().str(), buildIntType());

		//TO DO::In addition to the outer-loop id to private, we should put all the other loop indices inside the private clause

		
		std::set<const char *> priv_clause_names;
		for (int ll=0; ll<loop_indices.size(); ll++) 
		{
			priv_clause_names.insert(loop_indices[ll]->get_name().str());
		}


		std::set<const char *>::iterator _it;
		std::vector<SgInitializedName *> priv_idxs;

		for (_it=priv_clause_names.begin(); _it != priv_clause_names.end(); _it++)
		{
			//printf("yay..."); printf ("here is ..%s ..\n", *_it);
			priv_idxs.push_back(new SgInitializedName (*_it, buildIntType()));
		}
		

		//std::vector<SgInitializedName *> priv_idxs;
		//for (int ll=0; ll<loop_indices.size(); ll++)  
		//	priv_idxs.push_back(new SgInitializedName (loop_indices[ll]->get_name().str(), buildIntType()));

		//TO DO::put sc_temp in the private clause
		SgInitializedName *private_temp = new SgInitializedName("sc_temp", buildDoubleType());


		//TO DO:: put num_threads in the private clause
		SgInitializedName *private_num_threads = new SgInitializedName( new_dvs->get_name().str(), buildIntType());

		//add tid to private
		SgInitializedName *private_tid = new SgInitializedName( dvs->get_name().str(), buildIntType());
		//priv->get_variables().push_back(buildVarRefExp(private_arg));
		priv->get_variables().push_back(buildVarRefExp(private_tid));
		priv->get_variables().push_back(buildVarRefExp(private_num_threads));
		priv->get_variables().push_back(buildVarRefExp(private_temp));

		for(int ll=0; ll<priv_idxs.size(); ll++)
			priv->get_variables().push_back(buildVarRefExp(priv_idxs[ll]));	

		if(!use_omp_barrier)
		{
			SgInitializedName *private_left = new SgInitializedName( left_dvs->get_name().str(), buildIntType());
			SgInitializedName *private_right = new SgInitializedName( right_dvs->get_name().str(), buildIntType());
			priv->get_variables().push_back(buildVarRefExp(private_left));
			priv->get_variables().push_back(buildVarRefExp(private_right));
		}


		SgOmpSharedClause *shrd = new SgOmpSharedClause();
		if(!use_omp_barrier){
		SgInitializedName *shared_zplanes = new SgInitializedName( lcks->get_name().str(), buildIntType());
		shrd->get_variables().push_back(buildVarRefExp(shared_zplanes));
		}
		
		//Adding the num_threads clause
		if (omp_thrds_to_use)
			SgOmpNumThreadsClause *num_thrds_clause = new SgOmpNumThreadsClause(buildIntVal(omp_thrds_to_use));


		SgOmpParallelStatement *tnl2 = new SgOmpParallelStatement(TRANSFORMATION_FILE_INFO, isSgStatement(big_blk));
 		isSgOmpClauseBodyStatement(tnl2)->get_clauses().push_back(priv);

 		if(!use_omp_barrier) isSgOmpClauseBodyStatement(tnl2)->get_clauses().push_back(shrd);
		if (omp_thrds_to_use){
			SgOmpNumThreadsClause *num_thrds_clause = new SgOmpNumThreadsClause(buildIntVal(omp_thrds_to_use));
			isSgOmpClauseBodyStatement(tnl2)->get_clauses().push_back(num_thrds_clause);
		}

  		repr = new CG_roseRepr(tnl2);


	}

	return repr;


}


void Loop::omp_par_for(int loop_outer, int loop_inner, int num_thrds)
{
	omp_parallel_for = 1;
	omp_thrds_to_use = num_thrds;
}


CG_outputRepr *Loop::add_omp_parallel_for(CG_outputRepr *repr)const
{

	int outer_loop = 0;
	int inner_loop = 1;

	//1. At the top loop-level, insert #pragma omp parallel
	//2. Add private clause
	//3. Add num_thradds clause

	CG_roseRepr * code_loop = (CG_roseRepr *)repr;
	SgStatementPtrList *lst = code_loop->GetList();
	SgNode * _tnl = code_loop->GetCode();


	//SgBasicBlock *big_blk = new SgBasicBlock(TRANSFORMATION_FILE_INFO);

	SgVariableSymbol* index_outer;
	std::vector<SgVariableSymbol *> loop_indices;	


	
	//if(lst){printf("oops.. we have SgStatementPtrList \n");}

	//if(_tnl)
  //	{
	//  SgOmpParallelStatement *tnl = new SgOmpParallelStatement(TRANSFORMATION_FILE_INFO, isSgStatement(_tnl));
	//  (_tnl)->set_parent(tnl);

	//  CG_roseRepr * ret_repr = new CG_roseRepr(tnl);
	//  return ret_repr;

	//}




	int curr_loop = 0;

	//4. Add #parallel for at the second, deeper loop-level
	SgBasicBlock* bb = new SgBasicBlock(TRANSFORMATION_FILE_INFO);

	//bb->set_parent(pnt);DONE
	//bb->append_statement(new_inner_loop);
	//new_inner_loop->set_parent(bb);


	if(_tnl)
	{ 
		if(isSgForStatement(_tnl))
		{
			curr_loop ++;
			SgNode * _fors = _tnl;
			SgStatement* lp_body = isSgForStatement(_fors)->get_loop_body();

			while(isSgForStatement(lp_body))
			{
				if(curr_loop == inner_loop)
				{
					bb->set_parent(lp_body);
					//Add the loop nest here
	  				SgOmpForStatement *tnl2 = 
						new SgOmpForStatement(TRANSFORMATION_FILE_INFO, 
								isSgStatement(isSgForStatement(lp_body)->get_loop_body()));
	 				(isSgForStatement(lp_body)->get_loop_body())->set_parent(tnl2);
					bb->append_statement(tnl2);
					tnl2->set_parent(bb);
					isSgForStatement(lp_body)->set_loop_body(bb);
					break;
				}
				else
				{
				  _fors = lp_body;
				  curr_loop++;
				  if ( isSgForStatement(_fors))
				     lp_body = isSgForStatement(_fors)->get_loop_body();
				}
			}


		}
	}

	return new CG_roseRepr(_tnl);

}
*/


/*  OMP
void Loop::scrape_loop_indices(CG_outputRepr *repr)const
{

	CG_roseRepr * code_loop = (CG_roseRepr *)repr;
	//SgStatementPtrList *lst = code_loop->GetList();
	SgNode * _tnl = code_loop->GetCode();



	if (isSgBasicBlock(_tnl)) {

		SgStatementPtrList& list = isSgBasicBlock(_tnl)->get_statements();

		for (SgStatementPtrList::iterator it = list.begin(); it != list.end();it++) 
		{
			if (isSgForStatement(*it) || isSgIfStmt(*it))
				scrape_loop_indices(new CG_roseRepr(*it));


		}



	} else if (isSgForStatement(_tnl)) {

		SgNode * _fors = _tnl;
		SgStatement* lp_body_temp = isSgForStatement(_fors)->get_loop_body();

		SgForInitStatement *n_list = isSgForStatement(_fors)->get_for_init_stmt();
		SgStatementPtrList& n_ins = n_list->get_init_stmt();
		SgStatementPtrList::const_iterator n_j = n_ins.begin();

		if (SgExprStatement *expr = isSgExprStatement(*n_j))
			if (SgAssignOp* op = isSgAssignOp(expr->get_expression()))
				if (SgVarRefExp* var_ref = isSgVarRefExp(
							op->get_lhs_operand()))
					vec_loop_indices.push_back(var_ref->get_symbol());

		scrape_loop_indices(new CG_roseRepr (lp_body_temp));


	} else if (isSgIfStmt(_tnl)) {

		SgNode *tr_body = isSgIfStmt(_tnl)->get_true_body();
		scrape_loop_indices(new CG_roseRepr (tr_body));


	}

	return ;

}


CG_outputRepr * Loop::add_omp_for_recursive(CG_outputRepr *repr, int curr_level, int level_to_add, int num_thrds)const
{

	int level;

	CG_roseRepr *ret_repr;
	
	printf("curr val:%d\n", curr_level);


	CG_roseRepr * code_loop = (CG_roseRepr *)repr;
	SgStatementPtrList *lst = code_loop->GetList();
	SgNode * _tnl = code_loop->GetCode();


	if (curr_level == 0)
	{
		scrape_loop_indices(repr);
		printf("the num of loop indices : %d\n", vec_loop_indices.size());
	}



	if(lst) printf("error msg: stmt ptr list at the start");
	if(_tnl)
	{
		if(isSgForStatement(_tnl) && curr_level == level_to_add)
		{
			level = curr_level + 1;
			SgOmpForStatement *fr = new SgOmpForStatement(TRANSFORMATION_FILE_INFO, isSgStatement(_tnl));
			_tnl->set_parent(fr);
			SgBasicBlock *bb = buildBasicBlock();
	  		bb->append_statement(isSgStatement(fr));
			return new CG_roseRepr(bb);

		}

		if(isSgForStatement(_tnl) && curr_level < level_to_add)
		{
			level = curr_level + 1;
			SgStatement *bdy = isSgForStatement(_tnl)->get_loop_body();
			ret_repr = (CG_roseRepr *)add_omp_for_recursive (new CG_roseRepr(bdy), level, level_to_add);
			SgNode *tnl2 = ret_repr->GetCode();
			if(tnl2)
			{ 
				tnl2->set_parent(_tnl);
				isSgForStatement(_tnl)->set_loop_body(isSgStatement(tnl2));
			}

			//Add the outermost #pragma omp parallel private(...) num_threads(...)
			if(curr_level == 0)
			{
				SgOmpParallelStatement *tnl3 = new SgOmpParallelStatement(TRANSFORMATION_FILE_INFO, isSgStatement(_tnl));



				//Add private clauses
				SgOmpPrivateClause *priv =  new SgOmpPrivateClause();

				std::set<const char *> priv_clause_names;
				for (int ll=0; ll<vec_loop_indices.size(); ll++) 
				{
					if(!inVecIndices(ll)) priv_clause_names.insert(vec_loop_indices[ll]->get_name().str());
				}


				std::set<const char *>::iterator _it;
				std::vector<SgInitializedName *> priv_idxs;
				std::vector<SgInitializedName *> buff_idxs;

				for (_it=priv_clause_names.begin(); _it != priv_clause_names.end(); _it++)
				{
					//printf("yay..."); printf ("here is ..%s ..\n", *_it);
					priv_idxs.push_back(new SgInitializedName (*_it, buildIntType()));
				}

				for(int ll=0; ll<priv_idxs.size(); ll++)
					priv->get_variables().push_back(buildVarRefExp(priv_idxs[ll]));

				for (_it=omp_thrd_private.begin(); _it != omp_thrd_private.end(); _it++)
				{
					//printf("yay..."); printf ("here is ..%s ..\n", *_it);
					buff_idxs.push_back(new SgInitializedName (*_it, buildDoubleType()));
				}

				for(int ll=0; ll<buff_idxs.size(); ll++)
					priv->get_variables().push_back(buildVarRefExp(buff_idxs[ll]));

				isSgOmpClauseBodyStatement(tnl3)->get_clauses().push_back(priv);
				if(num_thrds) 
				{
					SgOmpNumThreadsClause *num_thrds_clause = new SgOmpNumThreadsClause(buildIntVal(num_thrds));
					isSgOmpClauseBodyStatement(tnl3)->get_clauses().push_back(num_thrds_clause);
				}


				return new CG_roseRepr(tnl3);

			}
			else return new CG_roseRepr(_tnl);


		}

		if(isSgBasicBlock(_tnl))
		{
			SgStatementPtrList& list = isSgBasicBlock(_tnl)->get_statements();
			SgBasicBlock *bb = buildBasicBlock();

			for (SgStatementPtrList::iterator it = list.begin(); it != list.end();it++) 
			{

				if(isSgForStatement(*it)) //printf("for-loop\n")
				{
					ret_repr = (CG_roseRepr *)add_omp_for_recursive (new CG_roseRepr(isSgNode(*it)), level+1, level_to_add);
					SgNode *tnl2 = ret_repr->GetCode();
					if(tnl2)
					{ 
						tnl2->set_parent(*it);
						bb->append_statement(isSgStatement(tnl2));
						//isSgForStatement(_tnl)->set_loop_body(isSgStatement(tnl2));
					}

				}
				else if(isSgIfStmt(*it))
				{
					printf("if-stmt\n");
					SgNode *tr_body = isSgIfStmt(*it)->get_true_body();
					ret_repr = (CG_roseRepr *)add_omp_for_recursive (new CG_roseRepr(isSgNode(tr_body)), curr_level, level_to_add);
					SgNode *tnl2 = ret_repr->GetCode();

					if(tnl2)
					{ 
						tnl2->set_parent(*it);
						isSgIfStmt(*it)->set_true_body(isSgStatement(tnl2));
						bb->append_statement(isSgStatement(*it));
					}


				}
				else 
				{
					printf("whoosh\n");

				}
			}
			return new CG_roseRepr(bb);

		}


		if(isSgIfStmt(_tnl))
		{
			printf("if-stmt\n");
			SgBasicBlock *bb = buildBasicBlock();
			SgNode *tr_body = isSgIfStmt(_tnl)->get_true_body();
			ret_repr = (CG_roseRepr *)add_omp_for_recursive (new CG_roseRepr(isSgNode(tr_body)), curr_level, level_to_add, num_thrds );
			SgNode *tnl2 = ret_repr->GetCode();

			if(tnl2)
			{ 
				tnl2->set_parent(_tnl);
				isSgIfStmt(_tnl)->set_true_body(isSgStatement(tnl2));
				//bb->append_statement(isSgStatement(_tnl));
			}

			SgNode *fl_body = isSgIfStmt(_tnl)->get_false_body();
			ret_repr = (CG_roseRepr *)add_omp_for_recursive (new CG_roseRepr(isSgNode(fl_body)), curr_level, level_to_add,num_thrds );
			tnl2 = ret_repr->GetCode();

			if(tnl2)
			{ 
				tnl2->set_parent(_tnl);
				isSgIfStmt(_tnl)->set_false_body(isSgStatement(tnl2));
				bb->append_statement(isSgStatement(_tnl));
			}

			return new CG_roseRepr(bb);

		}
		
	
	}

}
  */

