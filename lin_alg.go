package anydiff

import "github.com/unixpickle/anyvec"

type scaleVec struct {
	In     Vec
	OutVec anyvec.Vector
	Scaler anyvec.Numeric
}

// Scale scales the components of a Vec by a constant.
func Scale(v Vec, s anyvec.Numeric) Vec {
	newData := v.Output().Copy()
	newData.Scale(s)
	return &scaleVec{
		In:     v,
		OutVec: newData,
		Scaler: s,
	}
}

func (s *scaleVec) Output() anyvec.Vector {
	return s.OutVec
}

func (s *scaleVec) Vars() VarSet {
	return s.In.Vars()
}

func (s *scaleVec) Propagate(u anyvec.Vector, g Grad) {
	u.Scale(s.Scaler)
	s.In.Propagate(u, g)
}

type addVec struct {
	In1    Vec
	In2    Vec
	V      VarSet
	OutVec anyvec.Vector
}

// Add performs vector addition.
func Add(v1, v2 Vec) Vec {
	newData := v1.Output().Copy()
	newData.Add(v2.Output())
	return &addVec{
		In1:    v1,
		In2:    v2,
		V:      MergeVarSets(v1.Vars(), v2.Vars()),
		OutVec: newData,
	}
}

func (a *addVec) Output() anyvec.Vector {
	return a.OutVec
}

func (a *addVec) Vars() VarSet {
	return a.V
}

func (a *addVec) Propagate(u anyvec.Vector, g Grad) {
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

// A Matrix is a matrix with a row-major backing array.
type Matrix struct {
	Data Vec
	Rows int
	Cols int
}

func (m *Matrix) anyvecMatrix() *anyvec.Matrix {
	return &anyvec.Matrix{
		Data: m.Data.Output(),
		Rows: m.Rows,
		Cols: m.Cols,
	}
}

func (m *Matrix) anyvecZeroMatrix() *anyvec.Matrix {
	return &anyvec.Matrix{
		Data: m.Data.Output().Creator().MakeVector(m.Rows * m.Cols),
		Rows: m.Rows,
		Cols: m.Cols,
	}
}

type matMulVec struct {
	OutVec anyvec.Vector
	Deps   VarSet

	Trans1 bool
	Trans2 bool
	M1     *Matrix
	M2     *Matrix
}

// MatMul multiplies two matrices
// The trans1 and trans2 arguments indicate whether to
// transpose m1 and m2, respectively.
func MatMul(trans1, trans2 bool, m1, m2 *Matrix) *Matrix {
	outRows, outCols := m1.Rows, m2.Cols
	if trans1 {
		outRows = m1.Cols
	}
	if trans2 {
		outCols = m2.Rows
	}
	c := m1.Data.Output().Creator()
	outVec := c.MakeVector(outRows * outCols)

	anyM1 := m1.anyvecMatrix()
	anyM2 := m2.anyvecMatrix()
	anyM3 := &anyvec.Matrix{Data: outVec, Rows: outRows, Cols: outCols}

	anyM3.Product(trans1, trans2, c.MakeNumeric(1), anyM1, anyM2, c.MakeNumeric(0))

	return &Matrix{
		Data: &matMulVec{
			OutVec: anyM3.Data,
			Deps:   MergeVarSets(m1.Data.Vars(), m2.Data.Vars()),
			Trans1: trans1,
			Trans2: trans2,
			M1:     m1,
			M2:     m2,
		},
		Rows: outRows,
		Cols: outCols,
	}
}

func (m *matMulVec) Output() anyvec.Vector {
	return m.OutVec
}

func (m *matMulVec) Vars() VarSet {
	return m.Deps
}

func (m *matMulVec) Propagate(u anyvec.Vector, g Grad) {
	c := m.OutVec.Creator()
	one := c.MakeNumeric(1)
	zero := c.MakeNumeric(0)

	uMat := m.upstreamMat(u)

	// TODO: avoid downstream allocs for *Var inputs.

	if g.Intersects(m.M1.Data.Vars()) {
		mDown := m.M1.anyvecZeroMatrix()
		if m.Trans1 {
			mDown.Product(m.Trans2, true, one, m.M2.anyvecMatrix(), uMat, zero)
		} else {
			mDown.Product(false, !m.Trans2, one, uMat, m.M2.anyvecMatrix(), zero)
		}
		m.M1.Data.Propagate(mDown.Data, g)
	}

	if g.Intersects(m.M2.Data.Vars()) {
		mDown := m.M2.anyvecZeroMatrix()
		if m.Trans2 {
			mDown.Product(true, m.Trans1, one, uMat, m.M1.anyvecMatrix(), zero)
		} else {
			mDown.Product(!m.Trans1, false, one, m.M1.anyvecMatrix(), uMat, zero)
		}
		m.M2.Data.Propagate(mDown.Data, g)
	}
}

func (m *matMulVec) upstreamMat(u anyvec.Vector) *anyvec.Matrix {
	outRows, outCols := m.M1.Rows, m.M2.Cols
	if m.Trans1 {
		outRows = m.M1.Cols
	}
	if m.Trans2 {
		outCols = m.M2.Rows
	}
	return &anyvec.Matrix{
		Rows: outRows,
		Cols: outCols,
		Data: u,
	}
}
