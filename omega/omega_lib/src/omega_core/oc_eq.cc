/*****************************************************************************
 Copyright (C) 1994-2000 the Omega Project Team
 Copyright (C) 2005-2011 Chun Chen
 All Rights Reserved.

 Purpose:
 Solve equalities.

 Notes:

 History:
 *****************************************************************************/

#include <omega/omega_core/oc_i.h>

namespace omega {

void Problem::simplifyStrideConstraints() {
	int e, e2, i;
	if (DBUG)
		fprintf(outputFile, "Checking for stride constraints\n");
	for (i = safeVars + 1; i <= nVars; i++) {
		if (DBUG)
			fprintf(outputFile, "checking %s\n", variable(i));
		for (e = 0; e < nGEQs; e++)
			if (GEQs[e].coef[i])
				break;
		if (e >= nGEQs) {
			if (DBUG)
				fprintf(outputFile, "%s passed GEQ test\n", variable(i));
			e2 = -1;
			for (e = 0; e < nEQs; e++)
				if (EQs[e].coef[i]) {
					if (e2 == -1)
						e2 = e;
					else {
						e2 = -1;
						break;
					}
				}
			if (e2 >= 0) {
				if (DBUG) {
					fprintf(outputFile, "Found stride constraint: ");
					printEQ(&EQs[e2]);
					fprintf(outputFile, "\n");
				}
				/* Is a stride constraint */
				coef_t g = abs(EQs[e2].coef[i]);
				assert(g>0);
				int j;
				for (j = 0; j <= nVars; j++)
					if (i != j)
						EQs[e2].coef[j] = int_mod_hat(EQs[e2].coef[j], g);
			}
		}
	}
}

void Problem::doMod(coef_t factor, int e, int j) {
	/* Solve e = factor alpha for x_j and substitute */
	int k;
	eqn eq;
	coef_t nFactor;

	int alpha;

	// if (j > safeVars) alpha = j;
	// else
	if (EQs[e].color) {
		rememberRedConstraint(&EQs[e], redEQ, 0);
		EQs[e].color = EQ_BLACK;
	}
	alpha = addNewUnprotectedWildcard();
	eqnncpy(&eq, &EQs[e], nVars);
	newVar = alpha;

	if (DEBUG) {
		fprintf(outputFile, "doing moding: ");
		fprintf(outputFile, "Solve ");
		printTerm(&eq, 1);
		fprintf(outputFile, " = " coef_fmt " %s for %s and substitute\n",
				factor, variable(alpha), variable(j));
	}
	for (k = nVars; k >= 0; k--)
		eq.coef[k] = int_mod_hat(eq.coef[k], factor);
	nFactor = eq.coef[j];
	assert(nFactor == 1 || nFactor == -1);
	eq.coef[alpha] = factor / nFactor;
	if (DEBUG) {
		fprintf(outputFile, "adjusted: ");
		fprintf(outputFile, "Solve ");
		printTerm(&eq, 1);
		fprintf(outputFile, " = 0 for %s and substitute\n", variable(j));
	}

	eq.coef[j] = 0;
	substitute(&eq, j, nFactor);
	newVar = -1;
	deleteVariable(j);
	for (k = nVars; k >= 0; k--) {
		assert(EQs[e].coef[k] % factor == 0);
		EQs[e].coef[k] = EQs[e].coef[k] / factor;
	}
	if (DEBUG) {
		fprintf(outputFile, "Mod-ing and normalizing produces:\n");
		printProblem();
	}
}

void Problem::substitute(eqn *sub, int i, coef_t c) {
	int e, j;
	coef_t k;
	int recordSubstitution = (i <= safeVars && var[i] >= 0);

	redType clr;
	if (sub->color)
		clr = redEQ;
	else
		clr = notRed;

	assert(c == 1 || c == -1);

	if (DBUG || doTrace) {
		if (sub->color)
			fprintf(outputFile, "RED SUBSTITUTION\n");
		fprintf(outputFile, "substituting using %s := ", variable(i));
		printTerm(sub, -c);
		fprintf(outputFile, "\n");
		printVars();
	}
#ifndef NDEBUG
	if (i > safeVars && clr) {
		bool unsafeSub = false;
		for (e = nEQs - 1; e >= 0; e--)
			if (!(EQs[e].color || !EQs[e].coef[i]))
				unsafeSub = true;
		for (e = nGEQs - 1; e >= 0; e--)
			if (!(GEQs[e].color || !GEQs[e].coef[i]))
				unsafeSub = true;
		for (e = nSUBs - 1; e >= 0; e--)
			if (SUBs[e].coef[i])
				unsafeSub = true;
		if (unsafeSub) {
			fprintf(outputFile, "UNSAFE RED SUBSTITUTION\n");
			fprintf(outputFile, "substituting using %s := ", variable(i));
			printTerm(sub, -c);
			fprintf(outputFile, "\n");
			printProblem();
			assert(0 && "UNSAFE RED SUBSTITUTION");
		}
	}
#endif

	for (e = nEQs - 1; e >= 0; e--) {
		eqn *eq;
		eq = &(EQs[e]);
		k = eq->coef[i];
		if (k != 0) {
			k = check_mul(k, c); // Should be k = k/c, but same effect since abs(c) == 1
			eq->coef[i] = 0;
			for (j = nVars; j >= 0; j--) {
				eq->coef[j] -= check_mul(sub->coef[j], k);
			}
		}
		if (DEBUG) {
			printEQ(eq);
			fprintf(outputFile, "\n");
		}
	}
	for (e = nGEQs - 1; e >= 0; e--) {
		int zero;
		eqn *eq;
		eq = &(GEQs[e]);
		k = eq->coef[i];
		if (k != 0) {
			k = check_mul(k, c); // Should be k = k/c, but same effect since abs(c) == 1
			eq->touched = true;
			eq->coef[i] = 0;
			zero = 1;
			for (j = nVars; j >= 0; j--) {
				eq->coef[j] -= check_mul(sub->coef[j], k);
				if (j > 0 && eq->coef[j])
					zero = 0;
			}
			if (zero && clr != notRed && !eq->color) {
				coef_t z = int_div(eq->coef[0], abs(k));
				if (DBUG || doTrace) {
					fprintf(outputFile,
							"Black inequality matches red substitution\n");
					if (z < 0)
						fprintf(outputFile, "System is infeasible\n");
					else if (z > 0)
						fprintf(outputFile, "Black inequality is redundant\n");
					else {
						fprintf(outputFile,
								"Black constraint partially implies red equality\n");
						if (k < 0) {
							fprintf(outputFile, "Black constraints tell us ");
							assert(sub->coef[i] == 0);
							sub->coef[i] = c;
							printTerm(sub, 1);
							sub->coef[i] = 0;
							fprintf(outputFile, "<= 0\n");
						} else {
							fprintf(outputFile, "Black constraints tell us ");
							assert(sub->coef[i] == 0);
							sub->coef[i] = c;
							printTerm(sub, 1);
							sub->coef[i] = 0;
							fprintf(outputFile, " >= 0\n");
						}
					}
				}
				if (z == 0) {
					if (k < 0) {
						if (clr == redEQ)
							clr = redGEQ;
						else if (clr == redLEQ)
							clr = notRed;
					} else {
						if (clr == redEQ)
							clr = redLEQ;
						else if (clr == redGEQ)
							clr = notRed;
					}
				}

			}
		}
		if (DEBUG) {
			printGEQ(eq);
			fprintf(outputFile, "\n");
		}
	}
	if (i <= safeVars && clr) {
		assert(sub->coef[i] == 0);
		sub->coef[i] = c;
		rememberRedConstraint(sub, clr, 0);
		sub->coef[i] = 0;
	}

	if (recordSubstitution) {
		int s = nSUBs++;
		int kk;
		eqn *eq = &(SUBs[s]);
		for (kk = nVars; kk >= 0; kk--)
			eq->coef[kk] = check_mul(-c, (sub->coef[kk]));
		eq->key = var[i];
		if (DEBUG) {
			fprintf(outputFile, "Recording substition as: ");
			printSubstitution(s);
			fprintf(outputFile, "\n");
		}
	}
	if (DEBUG) {
		fprintf(outputFile, "Ready to update subs\n");
		if (sub->color)
			fprintf(outputFile, "RED SUBSTITUTION\n");
		fprintf(outputFile, "substituting using %s := ", variable(i));
		printTerm(sub, -c);
		fprintf(outputFile, "\n");
		printVars();
	}

	for (e = nSUBs - 1; e >= 0; e--) {
		eqn *eq = &(SUBs[e]);
		k = eq->coef[i];
		if (k != 0) {
			k = check_mul(k, c); // Should be k = k/c, but same effect since abs(c) == 1
			eq->coef[i] = 0;
			for (j = nVars; j >= 0; j--) {
				eq->coef[j] -= check_mul(sub->coef[j], k);
			}
		}
		if (DEBUG) {
			fprintf(outputFile, "updated sub (" coef_fmt "): ", c);
			printSubstitution(e);
			fprintf(outputFile, "\n");
		}
	}

	if (DEBUG) {
		fprintf(outputFile, "---\n\n");
		printProblem();
		fprintf(outputFile, "===\n\n");
	}
}


void Problem::doElimination(int e, int i) {
	if (DBUG || doTrace)
		fprintf(outputFile, "eliminating variable %s\n", variable(i));

	eqn sub;
	eqnncpy(&sub, &EQs[e], nVars);
	coef_t c = sub.coef[i];
	sub.coef[i] = 0;

	if (c == 1 || c == -1) {
		substitute(&sub, i, c);
	} else {
		coef_t a = abs(c);
		if (TRACE)
			fprintf(outputFile,
					"performing non-exact elimination, c = " coef_fmt "\n", c);
		if (DBUG)
			printProblem();
		assert(inApproximateMode);

		for (int e2 = nEQs - 1; e2 >= 0; e2--) {
			eqn *eq = &(EQs[e2]);
			coef_t k = eq->coef[i];
			if (k != 0) {
				coef_t l = lcm(abs(k), a);
				coef_t scale1 = l / abs(k);
				for (int j = nVars; j >= 0; j--)
					eq->coef[j] = check_mul(eq->coef[j], scale1);
				eq->coef[i] = 0;
				coef_t scale2 = l / c;
				if (k < 0)
					scale2 = -scale2;
				for (int j = nVars; j >= 0; j--)
					eq->coef[j] -= check_mul(sub.coef[j], scale2);
				eq->color |= sub.color;
			}
		}
		for (int e2 = nGEQs - 1; e2 >= 0; e2--) {
			eqn *eq = &(GEQs[e2]);
			coef_t k = eq->coef[i];
			if (k != 0) {
				coef_t l = lcm(abs(k), a);
				coef_t scale1 = l / abs(k);
				for (int j = nVars; j >= 0; j--)
					eq->coef[j] = check_mul(eq->coef[j], scale1);
				eq->coef[i] = 0;
				coef_t scale2 = l / c;
				if (k < 0)
					scale2 = -scale2;
				for (int j = nVars; j >= 0; j--)
					eq->coef[j] -= check_mul(sub.coef[j], scale2);
				eq->color |= sub.color;
				eq->touched = 1;
			}
		}
		for (int e2 = nSUBs - 1; e2 >= 0; e2--)
			if (SUBs[e2].coef[i]) {
				eqn *eq = &(EQs[e2]);
				assert(0);
				// We can't handle this since we can't multiply
				// the coefficient of the left-hand side
				assert(!sub.color);
				for (int j = nVars; j >= 0; j--)
					eq->coef[j] = check_mul(eq->coef[j], a);
				coef_t k = eq->coef[i];
				eq->coef[i] = 0;
				for (int j = nVars; j >= 0; j--)
					eq->coef[j] -= check_mul(sub.coef[j], k / c);
			}
	}
	deleteVariable(i);
}

int Problem::solveEQ() {
	check();

	// Reorder equations according to complexity.
	{
		int delay[nEQs];

		for (int e = 0; e < nEQs; e++) {
			delay[e] = 0;
			if (EQs[e].color)
				delay[e] += 8;
			int nonunitWildCards = 0;
			int unitWildCards = 0;
			for (int i = nVars; i > safeVars; i--)
				if (EQs[e].coef[i]) {
					if (EQs[e].coef[i] == 1 || EQs[e].coef[i] == -1)
						unitWildCards++;
					else
						nonunitWildCards++;
				}
			int unit = 0;
			int nonUnit = 0;
			for (int i = safeVars; i > 0; i--)
				if (EQs[e].coef[i]) {
					if (EQs[e].coef[i] == 1 || EQs[e].coef[i] == -1)
						unit++;
					else
						nonUnit++;
				}
			if (unitWildCards == 1 && nonunitWildCards == 0)
				delay[e] += 0;
			else if (unitWildCards >= 1 && nonunitWildCards == 0)
				delay[e] += 1;
			else if (inApproximateMode && nonunitWildCards > 0)
				delay[e] += 2;
			else if (unit == 1 && nonUnit == 0 && nonunitWildCards == 0)
				delay[e] += 3;
			else if (unit > 1 && nonUnit == 0 && nonunitWildCards == 0)
				delay[e] += 4;
			else if (unit >= 1 && nonunitWildCards <= 1)
				delay[e] += 5;
			else
				delay[e] += 6;
		}

		for (int e = 0; e < nEQs; e++) {
			int e2, slowest;
			slowest = e;
			for (e2 = e + 1; e2 < nEQs; e2++)
				if (delay[e2] > delay[slowest])
					slowest = e2;
			if (slowest != e) {
				int tmp = delay[slowest];
				delay[slowest] = delay[e];
				delay[e] = tmp;
				eqn eq;
				eqnncpy(&eq, &EQs[slowest], nVars);
				eqnncpy(&EQs[slowest], &EQs[e], nVars);
				eqnncpy(&EQs[e], &eq, nVars);
			}
		}
	}

	// Eliminate all equations.
	while (nEQs != 0) {
		int e = nEQs - 1;
		eqn *eq = &(EQs[e]);
		coef_t g, g2;

		assert(mayBeRed || !eq->color);

		check();

		// get gcd of coefficients of all unprotected variables
		g2 = 0;
		for (int i = nVars; i > safeVars; i--)
			if (eq->coef[i] != 0) {
				g2 = gcd(abs(eq->coef[i]), g2);
				if (g2 == 1)
					break;
			}

		// get gcd of coefficients of all variables
		g = g2;
		if (g != 1)
			for (int i = safeVars; i >= 1; i--)
				if (eq->coef[i] != 0) {
					g = gcd(abs(eq->coef[i]), g);
					if (g == 1)
						break;
				}

		// approximate mode bypass integer modular test; in Farkas(),
		// existential variable lambda's are rational numbers.
		if (inApproximateMode && g2 != 0)
			g = gcd(abs(eq->coef[0]), g);

		// simple test to see if the equation is satisfiable
		if (g == 0) {
			if (eq->coef[0] != 0) {
				return (false);
			} else {
				nEQs--;
				continue;
			}
		} else if (abs(eq->coef[0]) % g != 0) {
			return (false);
		}

		// set gcd of all coefficients to 1
		if (g != 1) {
			for (int i = nVars; i >= 0; i--)
				eq->coef[i] /= g;
			g2 = g2 / g;
		}

		// exact elimination of unit coefficient variable
		if (g2 != 0) { // for constraint with unprotected variable
			int i;
			for (i = nVars; i > safeVars; i--)
				if (abs(eq->coef[i]) == 1)
					break;
			if (i > safeVars) {
				nEQs--;
				doElimination(e, i);
				continue;
			}
		} else { // for constraint without unprotected variable

			// pick the unit coefficient variable with complex inequalites
			// to eliminate, this will make inequalities tighter. e.g.
			// {[t4,t6,t10]:exists (alpha: 0<=t6<=3 && t10=4alpha+t6 &&
			//                             64t4<=t10<=64t4+15)}
			int unit_var;
			int cost = -1;

			for (int i = safeVars; i > 0; i--)

				if (abs(eq->coef[i]) == 1) {
					int cur_cost = 0;
					for (int j = 0; j < nGEQs; j++)
						if (GEQs[j].coef[i] != 0) {
							for (int k = safeVars; k > 0; k--)
								if (GEQs[j].coef[k] != 0) {
									if (abs(GEQs[j].coef[k]) != 1){

										cur_cost += 3;
 
                                                                          }
									  else
									        cur_cost += 1;
								}
						}
                                     
					if (cur_cost > cost) {
						cost = cur_cost;
						unit_var = i;
                    		      }

				}

			if (cost != -1) {
				nEQs--;
				doElimination(e, unit_var);
				continue;
			}


		}

		// check if there is an unprotected variable as wildcard
		if (g2 > 0) {
			int pos = 0;
			coef_t g3;
			for (int k = nVars; k > safeVars; k--)
				if (eq->coef[k] != 0) {
					int e2;
					for (e2 = e - 1; e2 >= 0; e2--)
						if (EQs[e2].coef[k])
							break;
					if (e2 >= 0)
						continue;
					for (e2 = nGEQs - 1; e2 >= 0; e2--)
						if (GEQs[e2].coef[k])
							break;
					if (e2 >= 0)
						continue;
					for (e2 = nSUBs - 1; e2 >= 0; e2--)
						if (SUBs[e2].coef[k])
							break;
					if (e2 >= 0)
						continue;

					if (pos == 0) {
						g3 = abs(eq->coef[k]);
						pos = k;
					} else {
						if (abs(eq->coef[k]) < g3) {
							g3 = abs(eq->coef[k]);
							pos = k;
						}
					}
				}

			if (pos != 0) {
				bool change = false;
				for (int k2 = nVars; k2 >= 0; k2--)
					if (k2 != pos && eq->coef[k2] != 0) {
						coef_t t = int_mod_hat(eq->coef[k2], g3);
						if (t != eq->coef[k2]) {
							eq->coef[k2] = t;
							change = true;
						}
					}

				// strength reduced, try this equation again
				if (change) {
					// nameWildcard(pos);
					continue;
				}
			}
		}

		// insert new stride constraint
		if (g2 > 1 && !(inApproximateMode && !inStridesAllowedMode)) {
			int newvar = addNewProtectedWildcard();
			int neweq = newEQ();
			assert(neweq == e+1);
			// we were working on highest-numbered EQ
			eqnnzero(&EQs[neweq], nVars);
			eqnncpy(&EQs[neweq], eq, safeVars);

			for (int k = nVars; k >= 0; k--) {
				EQs[neweq].coef[k] = int_mod_hat(EQs[neweq].coef[k], g2);
			}
			if (EQs[e].color)
				rememberRedConstraint(&EQs[neweq], redStride, g2);
			EQs[neweq].coef[newvar] = g2;
			EQs[neweq].color = EQ_BLACK;
			continue;
		}

		// inexact elimination of unprotected variable
		if (g2 > 0 && inApproximateMode) {
			int pos = 0;
			for (int k = nVars; k > safeVars; k--)
				if (eq->coef[k] != 0) {
					pos = k;
					break;
				}
			assert(pos > safeVars);

			// special handling for wildcard used in breaking down
			// diophantine equation
			if (abs(eq->coef[pos]) > 1) {
				int e2;
				for (e2 = nSUBs - 1; e2 >= 0; e2--)
					if (SUBs[e2].coef[pos])
						break;
				if (e2 >= 0) {
					protectWildcard(pos);
					continue;
				}
			}

			nEQs--;
			doElimination(e, pos);
			continue;
		}

		// now solve linear diophantine equation using least remainder
		// algorithm
		{
			coef_t factor = (posInfinity);  // was MAXINT
			int pos = 0;
			for (int k = nVars; k > (g2 > 0 ? safeVars : 0); k--)
				if (eq->coef[k] != 0 && factor > abs(eq->coef[k]) + 1) {
					factor = abs(eq->coef[k]) + 1;
					pos = k;
				}
			assert(pos > (g2>0?safeVars:0));
			doMod(factor, e, pos);
			continue;
		}
	}

	assert(nEQs == 0);
	return (OC_SOLVE_UNKNOWN);
}

} // namespace
