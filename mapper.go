package anydiff

import "github.com/unixpickle/anyvec"

type mapRes struct {
	M   anyvec.Mapper
	In  Res
	Out anyvec.Vector
}

// Map applies the anyvec.Mapper to the input.
func Map(m anyvec.Mapper, in Res) Res {
	out := in.Output().Creator().MakeVector(m.OutSize())
	m.Map(in.Output(), out)
	return &mapRes{
		M:   m,
		In:  in,
		Out: out,
	}
}

func (m *mapRes) Output() anyvec.Vector {
	return m.Out
}

func (m *mapRes) Vars() VarSet {
	return m.In.Vars()
}

func (m *mapRes) Propagate(u anyvec.Vector, g Grad) {
	down := u.Creator().MakeVector(m.In.Output().Len())
	m.M.MapTranspose(u, down)
	m.In.Propagate(down, g)
}

type mapTransposeRes struct {
	M   anyvec.Mapper
	In  Res
	Out anyvec.Vector
}

// MapTranspose applies the transpose of the anyvec.Mapper
// to the input.
func MapTranspose(m anyvec.Mapper, in Res) Res {
	out := in.Output().Creator().MakeVector(m.InSize())
	m.MapTranspose(in.Output(), out)
	return &mapTransposeRes{
		M:   m,
		In:  in,
		Out: out,
	}
}

func (m *mapTransposeRes) Output() anyvec.Vector {
	return m.Out
}

func (m *mapTransposeRes) Vars() VarSet {
	return m.In.Vars()
}

func (m *mapTransposeRes) Propagate(u anyvec.Vector, g Grad) {
	down := u.Creator().MakeVector(m.In.Output().Len())
	m.M.Map(u, down)
	m.In.Propagate(down, g)
}
