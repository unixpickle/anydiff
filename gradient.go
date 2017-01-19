package anydiff

import "github.com/unixpickle/anyvec"

// A Grad represents a gradient by mapping a set of
// variables each to their respective gradients.
type Grad map[*Var]anyvec.Vector

// NewGrad creates a zero Grad with the given variables.
func NewGrad(vars ...*Var) Grad {
	res := Grad{}
	for _, v := range vars {
		o := v.Output()
		res[v] = o.Creator().MakeVector(o.Len())
	}
	return res
}

// Intersects returns true if the gradient contains any of
// the variables in a VarSet.
func (g Grad) Intersects(v VarSet) bool {
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
func (g Grad) Scale(scaler anyvec.Numeric) {
	for _, x := range g {
		x.Scale(scaler)
	}
}

// AddToVars adds the gradient to its variables' vectors,
// thus performing a step of gradient descent.
func (g Grad) AddToVars() {
	for v, x := range g {
		v.Vector.Add(x)
	}
}
