package anydiff

import "github.com/unixpickle/anyvec"

type tanhRes struct {
	In     Res
	OutVec anyvec.Vector
}

// Tanh computes the hyperbolic tangent of each component
// of the input.
func Tanh(in Res) Res {
	v := in.Output().Copy()
	anyvec.Tanh(v)
	return &tanhRes{
		In:     in,
		OutVec: v,
	}
}

func (t *tanhRes) Output() anyvec.Vector {
	return t.OutVec
}

func (t *tanhRes) Vars() VarSet {
	return t.In.Vars()
}

func (t *tanhRes) Propagate(u anyvec.Vector, g Grad) {
	down := t.OutVec.Copy()
	anyvec.Pow(down, t.OutVec.Creator().MakeNumeric(2))
	anyvec.Complement(down)
	u.Mul(down)
	t.In.Propagate(u, g)
}
