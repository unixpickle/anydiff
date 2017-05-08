package anyfwd

import "github.com/unixpickle/anydiff"

// A FwdMaker is an object which can convert itself to use
// forward auto-diff.
//
// Typically, the object is some kind of parametric model
// and MakeFwd wraps the parameters in *Vectors.
type FwdMaker interface {
	MakeFwd(c *Creator)
}

// Parameterizer is an object with a set of parameters.
//
// This is used as a fall-back for MakeFwd.
type Parameterizer interface {
	Parameters() []*anydiff.Var
}

// MakeFwd promotes an object to use forward auto-diff
// with the given Creator.
//
// If the object is an *anydiff.Var or a FwdMaker, then
// conversion is done directly.
// If the object does not implement FwdMaker, a fallback
// based on Parameterizer is used.
// If none of the above conditions are met, then the
// object is left unchanged.
func MakeFwd(c *Creator, obj interface{}) {
	if fm, ok := obj.(FwdMaker); ok {
		fm.MakeFwd(c)
	} else if param, ok := obj.(*anydiff.Var); ok {
		oldVec := param.Vector
		param.Vector = c.MakeVector(oldVec.Len())
		param.Vector.(*Vector).Values.Set(oldVec)
	} else if p, ok := obj.(Parameterizer); ok {
		for _, param := range p.Parameters() {
			MakeFwd(c, param)
		}
	}
}
