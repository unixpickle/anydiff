package anydiff

import "github.com/unixpickle/anyvec"

type scaleRes struct {
	In     Res
	OutRes anyvec.Vector
	Scaler anyvec.Numeric
}

// Scale scales the components of a Res by a constant.
func Scale(v Res, s anyvec.Numeric) Res {
	newData := v.Output().Copy()
	newData.Scale(s)
	return &scaleRes{
		In:     v,
		OutRes: newData,
		Scaler: s,
	}
}

func (s *scaleRes) Output() anyvec.Vector {
	return s.OutRes
}

func (s *scaleRes) Vars() VarSet {
	return s.In.Vars()
}

func (s *scaleRes) Propagate(u anyvec.Vector, g Grad) {
	u.Scale(s.Scaler)
	s.In.Propagate(u, g)
}

type addRes struct {
	In1    Res
	In2    Res
	V      VarSet
	OutRes anyvec.Vector
}

// Add performs vector addition.
func Add(v1, v2 Res) Res {
	if v1.Output().Len() != v2.Output().Len() {
		panic("input sizes must match")
	}
	newData := v1.Output().Copy()
	newData.Add(v2.Output())
	return &addRes{
		In1:    v1,
		In2:    v2,
		V:      MergeVarSets(v1.Vars(), v2.Vars()),
		OutRes: newData,
	}
}

func (a *addRes) Output() anyvec.Vector {
	return a.OutRes
}

func (a *addRes) Vars() VarSet {
	return a.V
}

func (a *addRes) Propagate(u anyvec.Vector, g Grad) {
	int1 := g.Intersects(a.In1.Vars())
	int2 := g.Intersects(a.In2.Vars())
	if int1 && !int2 {
		a.In1.Propagate(u, g)
	} else if !int1 && int2 {
		a.In2.Propagate(u, g)
	} else {
		a.In1.Propagate(u.Copy(), g)
		a.In2.Propagate(u, g)
	}
}

type subRes struct {
	In1    Res
	In2    Res
	V      VarSet
	OutRes anyvec.Vector
}

// Sub subtracts the components of v2 from those of v1.
func Sub(v1, v2 Res) Res {
	if v1.Output().Len() != v2.Output().Len() {
		panic("input sizes must match")
	}
	newData := v1.Output().Copy()
	newData.Sub(v2.Output())
	return &subRes{
		In1:    v1,
		In2:    v2,
		V:      MergeVarSets(v1.Vars(), v2.Vars()),
		OutRes: newData,
	}
}

func (a *subRes) Output() anyvec.Vector {
	return a.OutRes
}

func (a *subRes) Vars() VarSet {
	return a.V
}

func (a *subRes) Propagate(u anyvec.Vector, g Grad) {
	int1 := g.Intersects(a.In1.Vars())
	int2 := g.Intersects(a.In2.Vars())
	if int1 && !int2 {
		a.In1.Propagate(u, g)
	} else if !int1 && int2 {
		u.Scale(u.Creator().MakeNumeric(-1))
		a.In2.Propagate(u, g)
	} else {
		a.In1.Propagate(u.Copy(), g)
		u.Scale(u.Creator().MakeNumeric(-1))
		a.In2.Propagate(u, g)
	}
}

type mulRes struct {
	In1    Res
	In2    Res
	V      VarSet
	OutVec anyvec.Vector
}

// Mul performs component-wise multiplication.
func Mul(v1, v2 Res) Res {
	if v1.Output().Len() != v2.Output().Len() {
		panic("input sizes must match")
	}
	out := v1.Output().Copy()
	out.Mul(v2.Output())
	return &mulRes{
		In1:    v1,
		In2:    v2,
		V:      MergeVarSets(v1.Vars(), v2.Vars()),
		OutVec: out,
	}
}

func (m *mulRes) Output() anyvec.Vector {
	return m.OutVec
}

func (m *mulRes) Vars() VarSet {
	return m.V
}

func (m *mulRes) Propagate(u anyvec.Vector, g Grad) {
	int1 := g.Intersects(m.In1.Vars())
	int2 := g.Intersects(m.In2.Vars())
	if int1 && !int2 {
		u.Mul(m.In2.Output())
		m.In1.Propagate(u, g)
	} else if !int1 && int2 {
		u.Mul(m.In1.Output())
		m.In2.Propagate(u, g)
	} else {
		uc := u.Copy()
		uc.Mul(m.In2.Output())
		m.In1.Propagate(uc, g)
		u.Mul(m.In1.Output())
		m.In2.Propagate(u, g)
	}
}

type divRes struct {
	In1    Res
	In2    Res
	V      VarSet
	OutVec anyvec.Vector
}

// Div performs component-wise division to find num/denom.
func Div(num, denom Res) Res {
	if num.Output().Len() != denom.Output().Len() {
		panic("input sizes must match")
	}
	res := num.Output().Copy()
	res.Div(denom.Output())
	return &divRes{
		In1:    num,
		In2:    denom,
		V:      MergeVarSets(num.Vars(), denom.Vars()),
		OutVec: res,
	}
}

func (d *divRes) Output() anyvec.Vector {
	return d.OutVec
}

func (d *divRes) Vars() VarSet {
	return d.V
}

func (d *divRes) Propagate(u anyvec.Vector, g Grad) {
	int1 := g.Intersects(d.In1.Vars())
	int2 := g.Intersects(d.In2.Vars())
	if int1 && !int2 {
		u.Div(d.In2.Output())
		d.In1.Propagate(u, g)
	} else if !int1 && int2 {
		u.Mul(d.In1.Output())
		u.Div(d.In2.Output())
		u.Div(d.In2.Output())
		u.Scale(u.Creator().MakeNumeric(-1))
		d.In2.Propagate(u, g)
	} else {
		uCpy := u.Copy()
		uCpy.Div(d.In2.Output())
		d.In1.Propagate(uCpy, g)

		u.Mul(d.In1.Output())
		u.Div(d.In2.Output())
		u.Div(d.In2.Output())
		u.Scale(u.Creator().MakeNumeric(-1))
		d.In2.Propagate(u, g)
	}
}

// Sum computes the complete sum of all the elements.
func Sum(r Res) Res {
	return SumCols(&Matrix{
		Data: r,
		Rows: 1,
		Cols: r.Output().Len(),
	})
}
