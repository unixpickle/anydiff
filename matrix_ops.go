package anydiff

import "github.com/unixpickle/anyvec"

// A Matrix is a matrix with a row-major backing array.
type Matrix struct {
	Data Res
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

func (m *Matrix) anyvecUpstreamMatrix(g Grad) (mat *anyvec.Matrix, isVar bool) {
	mat = &anyvec.Matrix{Rows: m.Rows, Cols: m.Cols}
	if v, ok := m.Data.(*Var); ok {
		if g[v] == nil {
			panic("no gradient for variable")
		}
		mat.Data = g[v]
		isVar = true
	} else {
		mat.Data = m.Data.Output().Creator().MakeVector(m.Rows * m.Cols)
	}
	return
}

type matMulRes struct {
	OutRes anyvec.Vector
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
	outRes := c.MakeVector(outRows * outCols)

	anyM1 := m1.anyvecMatrix()
	anyM2 := m2.anyvecMatrix()
	anyM3 := &anyvec.Matrix{Data: outRes, Rows: outRows, Cols: outCols}

	anyM3.Product(trans1, trans2, c.MakeNumeric(1), anyM1, anyM2, c.MakeNumeric(0))

	return &Matrix{
		Data: &matMulRes{
			OutRes: anyM3.Data,
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

func (m *matMulRes) Output() anyvec.Vector {
	return m.OutRes
}

func (m *matMulRes) Vars() VarSet {
	return m.Deps
}

func (m *matMulRes) Propagate(u anyvec.Vector, g Grad) {
	c := m.OutRes.Creator()
	one := c.MakeNumeric(1)
	zero := c.MakeNumeric(0)

	uMat := m.upstreamMat(u)

	if g.Intersects(m.M1.Data.Vars()) {
		mDown, isVar := m.M1.anyvecUpstreamMatrix(g)
		scaler := zero
		if isVar {
			scaler = one
		}
		if m.Trans1 {
			mDown.Product(m.Trans2, true, one, m.M2.anyvecMatrix(), uMat, scaler)
		} else {
			mDown.Product(false, !m.Trans2, one, uMat, m.M2.anyvecMatrix(), scaler)
		}
		if !isVar {
			m.M1.Data.Propagate(mDown.Data, g)
		}
	}

	if g.Intersects(m.M2.Data.Vars()) {
		mDown, isVar := m.M2.anyvecUpstreamMatrix(g)
		scaler := zero
		if isVar {
			scaler = one
		}
		if m.Trans2 {
			mDown.Product(true, m.Trans1, one, uMat, m.M1.anyvecMatrix(), scaler)
		} else {
			mDown.Product(!m.Trans1, false, one, m.M1.anyvecMatrix(), uMat, scaler)
		}
		if !isVar {
			m.M2.Data.Propagate(mDown.Data, g)
		}
	}
}

func (m *matMulRes) upstreamMat(u anyvec.Vector) *anyvec.Matrix {
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

type sumRowsRes struct {
	In  *Matrix
	Out anyvec.Vector
}

// SumRows sums the rows of a matrix.
func SumRows(m *Matrix) Res {
	out := anyvec.SumRows(m.Data.Output(), m.Cols)
	return &sumRowsRes{
		In:  m,
		Out: out,
	}
}

func (s *sumRowsRes) Output() anyvec.Vector {
	return s.Out
}

func (s *sumRowsRes) Vars() VarSet {
	return s.In.Data.Vars()
}

func (s *sumRowsRes) Propagate(u anyvec.Vector, g Grad) {
	downstream := s.Out.Creator().MakeVector(s.In.Data.Output().Len())
	anyvec.AddRepeated(downstream, u)
	s.In.Data.Propagate(downstream, g)
}

type sumColsRes struct {
	In  *Matrix
	Out anyvec.Vector
}

// SumCols sums the columns of a matrix.
func SumCols(m *Matrix) Res {
	out := anyvec.SumCols(m.Data.Output(), m.Rows)
	return &sumColsRes{
		In:  m,
		Out: out,
	}
}

func (s *sumColsRes) Output() anyvec.Vector {
	return s.Out
}

func (s *sumColsRes) Vars() VarSet {
	return s.In.Data.Vars()
}

func (s *sumColsRes) Propagate(u anyvec.Vector, g Grad) {
	downstream := s.Out.Creator().MakeVector(s.In.Data.Output().Len())
	anyvec.AddChunks(downstream, u)
	s.In.Data.Propagate(downstream, g)
}
