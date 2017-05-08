package anydiff

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

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
			uVec.Slice(s.Start, s.End).Add(u)
		}
	} else {
		c := s.In.Output().Creator()
		bigU := c.MakeVector(s.In.Output().Len())
		bigU.Slice(s.Start, s.End).Set(u)
		s.In.Propagate(bigU, g)
	}
}

type concatRes struct {
	Ins    []Res
	OutVec anyvec.Vector
	V      VarSet
}

// Concat concatenates one or more Reses.
func Concat(ins ...Res) Res {
	if len(ins) == 0 {
		panic("must take at least one argument")
	}
	vecs := make([]anyvec.Vector, len(ins))
	vars := VarSet{}
	for i, x := range ins {
		vecs[i] = x.Output()
		vars = MergeVarSets(vars, x.Vars())
	}
	return &concatRes{
		Ins:    ins,
		OutVec: ins[0].Output().Creator().Concat(vecs...),
		V:      vars,
	}
}

func (c *concatRes) Output() anyvec.Vector {
	return c.OutVec
}

func (c *concatRes) Vars() VarSet {
	return c.V
}

func (c *concatRes) Propagate(u anyvec.Vector, g Grad) {
	var start int
	for _, x := range c.Ins {
		if g.Intersects(x.Vars()) {
			slice := u.Slice(start, start+x.Output().Len())
			x.Propagate(slice, g)
		}
		start += x.Output().Len()
	}
}

// Split splits the vector into n evenly-sized pieces.
//
// The vector's length must be divisible by n.
func Split(vec Res, n int) MultiRes {
	if vec.Output().Len()%n != 0 {
		panic("count does not divide length")
	}
	return PoolFork(vec, func(vec Res) MultiRes {
		chunkSize := vec.Output().Len() / n
		var slices []anydiff.Res
		for i := 0; i < n; i++ {
			slices = append(slices, Slice(vec, i*chunkSize, (i+1)*chunkSize))
		}
		return Fuse(slices...)
	})
}
