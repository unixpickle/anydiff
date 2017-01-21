package anydiff

import "github.com/unixpickle/anyvec"

type addRepeatedRes struct {
	In     Res
	Bias   Res
	V      VarSet
	OutVec anyvec.Vector
}

// AddRepeated is equivalent to repeating the biases
// enough times to match the length of v and then adding
// the repeated vector to v.
//
// The length of the biases must divide the length of v.
func AddRepeated(v, biases Res) Res {
	if v.Output().Len()%biases.Output().Len() != 0 {
		panic("bias count must divide vector length")
	}
	sum := v.Output().Copy()
	anyvec.AddRepeated(sum, biases.Output())
	return &addRepeatedRes{
		In:     v,
		Bias:   biases,
		V:      MergeVarSets(v.Vars(), biases.Vars()),
		OutVec: sum,
	}
}

func (a *addRepeatedRes) Output() anyvec.Vector {
	return a.OutVec
}

func (a *addRepeatedRes) Vars() VarSet {
	return a.V
}

func (a *addRepeatedRes) Propagate(u anyvec.Vector, g Grad) {
	if g.Intersects(a.Bias.Vars()) {
		chunkSum := anyvec.SumRows(u, a.Bias.Output().Len())
		a.Bias.Propagate(chunkSum, g)
	}

	if g.Intersects(a.In.Vars()) {
		a.In.Propagate(u, g)
	}
}

type scaleRepeatedRes struct {
	In      Res
	Scalers Res
	V       VarSet
	OutVec  anyvec.Vector
}

// ScaleRepeated is equivalent to repeating the scalers
// enough times to match the length of v and then
// multiplying it (componentwise) with v.
//
// The length of the scalers must divide the length of v.
func ScaleRepeated(v, scalers Res) Res {
	if v.Output().Len()%scalers.Output().Len() != 0 {
		panic("scaler count must divide vector length")
	}
	sum := v.Output().Copy()
	anyvec.ScaleRepeated(sum, scalers.Output())
	return &scaleRepeatedRes{
		In:      v,
		Scalers: scalers,
		V:       MergeVarSets(v.Vars(), scalers.Vars()),
		OutVec:  sum,
	}
}

func (s *scaleRepeatedRes) Output() anyvec.Vector {
	return s.OutVec
}

func (s *scaleRepeatedRes) Vars() VarSet {
	return s.V
}

func (s *scaleRepeatedRes) Propagate(u anyvec.Vector, g Grad) {
	if g.Intersects(s.Scalers.Vars()) {
		uCopy := u.Copy()
		uCopy.Mul(s.In.Output())
		chunkSum := anyvec.SumRows(uCopy, s.Scalers.Output().Len())
		s.Scalers.Propagate(chunkSum, g)
	}

	if g.Intersects(s.In.Vars()) {
		anyvec.ScaleRepeated(u, s.Scalers.Output())
		s.In.Propagate(u, g)
	}
}
