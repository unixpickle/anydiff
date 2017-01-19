package anydiff

import "github.com/unixpickle/anyvec"

// A Gradient represents a gradient by mapping a set of
// variables each to their respective gradients.
type Gradient map[*Var]anyvec.Vector

// Intersects returns true if the gradient contains any of
// the variables in a VarSet.
func (g Gradient) Intersects(v VarSet) bool {
	if len(v) > len(g) {
		for variable := range g {
			if v.Has(variable) {
				return true
			}
		}
	} else {
		for variable := range v {
			if _, ok := g[variable]; ok {
				return true
			}
		}
	}
	return false
}

// Scale scales the gradient in place.
func (g Gradient) Scale(scaler anyvec.Numeric) {
	for _, x := range g {
		x.Scale(scaler)
	}
}

// AddToVars adds the gradient to its variables' vectors,
// thus performing a step of gradient descent.
func (g Gradient) AddToVars() {
	for v, x := range g {
		v.Vector.Add(x)
	}
}
