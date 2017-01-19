package anydiff

import "github.com/unixpickle/anyvec"

// VarSet represents a set of *Variables.
// A variable is in the set if it has an entry in the map.
type VarSet map[*Variable]struct{}

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

// Add adds a variable to the VarSet.
func (v VarSet) Add(variable *Variable) {
	v[variable] = struct{}{}
}

// Del removes a variable from the VarSet, or does nothing
// if the variable is not in the VarSet.
func (v VarSet) Del(variable *Variable) {
	delete(v, variable)
}

// Has returns true if v contains the variable.
func (v VarSet) Has(variable *Variable) bool {
	_, ok := v[variable]
	return ok
}

// A Variable represents any vector with respect to which
// a gradient can be computed.
//
// A *Variable gets its identity from its memory address.
// The underlying vector is simply a representation of the
// data within the variable.
type Variable struct {
	Vector anyvec.Vector
}

// NewVariable creates a Variable with the given vector.
// Note that it is possible to have two or more different
// variables contain the same vector.
func NewVariable(v anyvec.Vector) *Variable {
	return &Variable{Vector: v}
}

// Output returns the variable's vector.
func (v *Variable) Output() anyvec.Vector {
	return v.Vector
}

// Vars returns a VarSet containing v.
func (v *Variable) Vars() VarSet {
	return VarSet{v: struct{}{}}
}

// Propagate propagates a gradient through the variable.
func (v *Variable) Propagate(upstream anyvec.Vector, g Gradient) {
	if vec, ok := g[v]; ok {
		vec.Add(upstream)
	}
}

// A Constant is similar to a variable, but it does not
// report itself as depending on any variables.
// Thus, Constants are good for pieces of data you will
// never want to find the gradient for.
type Constant struct {
	Vector anyvec.Vector
}

// NewConstant creates a Constant with the given vector.
func NewConstant(v anyvec.Vector) *Constant {
	return &Constant{Vector: v}
}

// Output returns the constant's vector.
func (c *Constant) Output() anyvec.Vector {
	return c.Vector
}

// Vars returns an empty VarSet.
func (c *Constant) Vars() VarSet {
	return VarSet{}
}

// Propagate does nothing, since c is a constant.
func (c *Constant) Propagate(upstream anyvec.Vector, g Gradient) {
}
