package anyseq

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

type sumRes struct {
	In     Seq
	OutVec anyvec.Vector
}

// Sum sums all of the outputs in all of the timesteps to
// produce a single result.
//
// All timesteps must have the same output size.
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
		return anydiff.NewConst(s.Creator().MakeVector(0))
	}
	return &sumRes{
		In:     s,
		OutVec: sum,
	}
}

func (s *sumRes) Output() anyvec.Vector {
	return s.OutVec
}

func (s *sumRes) Vars() anydiff.VarSet {
	return s.In.Vars()
}

func (s *sumRes) Propagate(u anyvec.Vector, g anydiff.Grad) {
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

type sumEachRes struct {
	In     Seq
	OutVec anyvec.Vector
}

// SumEach sums the outputs of each sequence, producing a
// packed vector of sums (one per sequence).
// Empty sequences are ignored.
//
// All timesteps must have the same output size.
func SumEach(s Seq) anydiff.Res {
	if len(s.Output()) == 0 {
		return anydiff.NewConst(s.Creator().MakeVector(0))
	}
	out0 := s.Output()[0]
	sum := out0.Packed.Copy()
	for _, x := range s.Output()[1:] {
		if x.NumPresent() == out0.NumPresent() {
			sum.Add(x.Packed)
		} else {
			expBatch := ExpandBatch(x, out0.Present)
			sum.Add(expBatch.Packed)
		}
	}
	return &sumEachRes{
		In:     s,
		OutVec: sum,
	}
}

func (s *sumEachRes) Output() anyvec.Vector {
	return s.OutVec
}

func (s *sumEachRes) Vars() anydiff.VarSet {
	return s.In.Vars()
}

func (s *sumEachRes) Propagate(u anyvec.Vector, g anydiff.Grad) {
	uBatch := &Batch{
		Packed:  u,
		Present: s.In.Output()[0].Present,
	}
	downstream := make([]*Batch, len(s.In.Output()))
	for i, x := range s.In.Output() {
		downstream[i] = ReduceBatch(uBatch, x.Present)
	}
	s.In.Propagate(downstream, g)
}
