package anydiff

import "github.com/unixpickle/anyvec"

// VarSet represents a set of *Vars.
// A *Var is in the set if it has an entry in the map.
type VarSet map[*Var]struct{}

// MergeVarSets creates a VarSet by merging the other
// VarSets.
func MergeVarSets(vs ...VarSet) VarSet {
	res := VarSet{}
	for _, x := range vs {
		for k, v := range x {
			res[k] = v
		}
	}
	return res
}

// Add adds a Var to the VarSet.
func (v VarSet) Add(Var *Var) {
	v[Var] = struct{}{}
}

// Del removes a Var from the VarSet, or does nothing
// if the Var is not in the VarSet.
func (v VarSet) Del(Var *Var) {
	delete(v, Var)
}

// Has returns true if v contains the Var.
func (v VarSet) Has(Var *Var) bool {
	_, ok := v[Var]
	return ok
}

// A Var represents any vector with respect to which
// a gradient can be computed.
//
// A *Var gets its identity from its memory address.
// The underlying vector is simply a representation of the
// data within the *Var.
type Var struct {
	Vector anyvec.Vector
}

// NewVar creates a Var with the given vector.
// Note that it is possible to have two or more different
// Vars contain the same vector.
func NewVar(v anyvec.Vector) *Var {
	return &Var{Vector: v}
}

// Output returns the Var's vector.
func (v *Var) Output() anyvec.Vector {
	return v.Vector
}

// Vars returns a VarSet containing v.
func (v *Var) Vars() VarSet {
	return VarSet{v: struct{}{}}
}

// Propagate propagates a gradient through the Var.
func (v *Var) Propagate(upstream anyvec.Vector, g Grad) {
	if vec, ok := g[v]; ok {
		vec.Add(upstream)
	}
}

// A Const is similar to a Var, but it does not report
// itself as depending on any Vars.
// Thus, Consts are good for pieces of data you will never
// want to find the gradient for.
type Const struct {
	Vector anyvec.Vector
}

// NewConst creates a Const with the given vector.
func NewConst(v anyvec.Vector) *Const {
	return &Const{Vector: v}
}

// Output returns the Const's vector.
func (c *Const) Output() anyvec.Vector {
	return c.Vector
}

// Vars returns an empty VarSet.
func (c *Const) Vars() VarSet {
	return VarSet{}
}

// Propagate does nothing, since c is a constant.
func (c *Const) Propagate(upstream anyvec.Vector, g Grad) {
}
