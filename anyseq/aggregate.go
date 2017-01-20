package anyseq

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

type sumResult struct {
	In     Seq
	OutVec anyvec.Vector
}

// Sum sums all of the outputs in all of the timesteps to
// produce a single result.
//
// All timesteps must have the same output size.
// The sequence must have at least one timestep.
func Sum(s Seq) anydiff.Res {
	var sum anyvec.Vector
	for _, x := range s.Output() {
		if x.Packed.Len() == 0 {
			return anydiff.NewConst(x.Packed.Creator().MakeVector(0))
		} else if sum == nil {
			sum = anyvec.SumRows(x.Packed, x.Packed.Len()/x.NumPresent())
		} else {
			outLen := x.Packed.Len() / x.NumPresent()
			if outLen != sum.Len() {
				panic("all timesteps must have the same output length")
			}
			sum.Add(anyvec.SumRows(x.Packed, outLen))
		}
	}
	if sum == nil {
		panic("cannot sum empty sequence")
	}
	return &sumResult{
		In:     s,
		OutVec: sum,
	}
}

func (s *sumResult) Output() anyvec.Vector {
	return s.OutVec
}

func (s *sumResult) Vars() anydiff.VarSet {
	return s.In.Vars()
}

func (s *sumResult) Propagate(u anyvec.Vector, g anydiff.Grad) {
	upstream := make([]*Batch, len(s.In.Output()))
	for i, x := range s.In.Output() {
		upstream[i] = &Batch{
			Packed:  x.Packed.Creator().MakeVector(x.Packed.Len()),
			Present: x.Present,
		}
		anyvec.AddRepeated(upstream[i].Packed, u)
	}
	s.In.Propagate(upstream, g)
}
