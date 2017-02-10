package anydiff

import "github.com/unixpickle/anyvec"

type sliceRes struct {
	In     Res
	OutVec anyvec.Vector
	Start  int
	End    int
}

// Slice creates a Res from a range of components in the
// input.
//
// The start index is inclusive, while the end index is
// exclusive.
func Slice(in Res, start, end int) Res {
	if start < 0 || start > end || end > in.Output().Len() {
		panic("index out of range")
	}
	return &sliceRes{
		In:     in,
		OutVec: in.Output().Slice(start, end),
		Start:  start,
		End:    end,
	}
}

func (s *sliceRes) Output() anyvec.Vector {
	return s.OutVec
}

func (s *sliceRes) Vars() VarSet {
	return s.In.Vars()
}

func (s *sliceRes) Propagate(u anyvec.Vector, g Grad) {
	if v, ok := s.In.(*Var); ok {
		if uVec, ok := g[v]; ok {
			old := uVec.Slice(s.Start, s.End)
			old.Add(u)
			uVec.SetSlice(s.Start, old)
		}
	} else {
		c := s.In.Output().Creator()
		bigU := c.MakeVector(s.In.Output().Len())
		bigU.SetSlice(s.Start, u)
		s.In.Propagate(bigU, g)
	}
}
