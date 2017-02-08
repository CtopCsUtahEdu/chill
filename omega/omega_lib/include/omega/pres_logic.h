#if ! defined _pres_logic_h
#define _pres_logic_h 1

#include <omega/pres_form.h>

namespace omega {
//
// Presburger formula classes for logical operations: and, or not
//

/**
 * @brief Represents the logical conjunction of its children nodes
 *
 * It is "True" if it has no children
 */
class F_And    : public Formula {
public:
    inline Node_Type node_type() {return Op_And;}

    /**
     * @param preserves_level Should be 0 unless we know this will not
     * change the "level" of the constraints - ie the number of
     * leading corresponding in,out variables known to be equal
     * @return
     */
    GEQ_Handle     add_GEQ(int preserves_level = 0);
    EQ_Handle      add_EQ(int preserves_level = 0);
    /**
     * This is equivalent to creating and F_Exists node with a
     * new variable alpha as a child and attach an equallity
     * constraint \f$step \times \alpha + ? = 0\f$.
     *
     * Coefficient for all other variable is implicitly 0.
     */
    Stride_Handle  add_stride(int step, int preserves_level = 0);
    EQ_Handle   add_EQ(const Constraint_Handle &c, int preserves_level = 0);
    GEQ_Handle  add_GEQ(const Constraint_Handle &c, int preserves_level = 0);

    F_And    *and_with();
    /**
     * Adds an unknown constraints as a child, thus making the
     * formula an upper bound.
     */
    void add_unknown();

private:
    friend class Formula;  // add_and()
    F_And(Formula *p, Rel_Body *r);

private:
    Formula *copy(Formula *parent, Rel_Body *reln);
    virtual Conjunct *find_available_conjunct();
    int priority();
    void print_separator(FILE *output_file);
    void prefix_print(FILE *output_file, int debug = 1);
    void beautify();
    DNF* DNFize();
 
    Conjunct *pos_conj;
};

/**
 * @brief Represents the logical disjunction of its children nodes
 *
 * It is "False" if it has no children
 */
class F_Or     : public Formula {
public:
    inline Node_Type node_type() {return Op_Or;}

private:
    friend class Formula; // add_or
    F_Or(Formula *, Rel_Body *);

private:
    Formula *copy(Formula *parent, Rel_Body *reln);

    virtual Conjunct *find_available_conjunct();
    void print_separator(FILE *output_file);
    void prefix_print(FILE *output_file, int debug = 1);
    void beautify();
    int priority();
    DNF* DNFize();
    void push_exists(Variable_ID_Tuple &S);
};

//! Represents the logical negation of its single child node
class F_Not    : public Formula {
public:
    inline Node_Type node_type() {return Op_Not;}
    void finalize();

private:
    friend class Formula;
    F_Not(Formula *, Rel_Body *);

private:
    Formula *copy(Formula *parent, Rel_Body *reln);

    virtual Conjunct *find_available_conjunct();
    friend class F_Forall;
    bool can_add_child();
    void beautify();
    void rearrange();
    int priority();
    DNF* DNFize();
    void print(FILE *output_file);
    void prefix_print(FILE *output_file, int debug = 1);
};

} // namespace

#endif
