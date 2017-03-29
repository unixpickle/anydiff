package anydiff

import "github.com/unixpickle/anyvec"

// A Matrix is a matrix with a row-major backing array.
type Matrix struct {
	Data Res
	Rows int
	Cols int
}

func (m *Matrix) batch() *MatrixBatch {
	return &MatrixBatch{
		Data: m.Data,
		Rows: m.Rows,
		Cols: m.Cols,
		Num:  1,
	}
}

type matMulRes struct {
	OutRes anyvec.Vector
	Deps   VarSet

	Trans1 bool
	Trans2 bool
	M1     *Matrix
	M2     *Matrix
}

// MatMul multiplies two matrices.
// The trans1 and trans2 arguments indicate whether to
// transpose m1 and m2, respectively.
func MatMul(trans1, trans2 bool, m1, m2 *Matrix) *Matrix {
	res := BatchedMatMul(trans1, trans2, m1.batch(), m2.batch())
	return &Matrix{
		Data: res.Data,
		Rows: res.Rows,
		Cols: res.Cols,
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

type scaleRowsRes struct {
	In      *Matrix
	Scalers Res
	Out     anyvec.Vector
	V       VarSet
}

// ScaleRows scales each row of the matrix by a
// corresponding scaler.
func ScaleRows(m *Matrix, scalers Res) *Matrix {
	if m.Rows != scalers.Output().Len() {
		panic("scaler count must match row count")
	}
	outVec := m.Data.Output().Copy()
	anyvec.ScaleChunks(outVec, scalers.Output())
	return &Matrix{
		Data: &scaleRowsRes{
			In:      m,
			Scalers: scalers,
			Out:     outVec,
			V:       MergeVarSets(m.Data.Vars(), scalers.Vars()),
		},
		Rows: m.Rows,
		Cols: m.Cols,
	}
}

func (s *scaleRowsRes) Output() anyvec.Vector {
	return s.Out
}

func (s *scaleRowsRes) Vars() VarSet {
	return s.V
}

func (s *scaleRowsRes) Propagate(u anyvec.Vector, g Grad) {
	if g.Intersects(s.Scalers.Vars()) {
		m := u.Copy()
		m.Mul(s.In.Data.Output())
		sum := anyvec.SumCols(m, s.Scalers.Output().Len())
		s.Scalers.Propagate(sum, g)
	}
	if g.Intersects(s.In.Data.Vars()) {
		anyvec.ScaleChunks(u, s.Scalers.Output())
		s.In.Data.Propagate(u, g)
	}
}

// A MatrixBatch is a batch of matrices, packed one after
// another in a vector.
type MatrixBatch struct {
	Data Res
	Num  int
	Rows int
	Cols int
}

func (m *MatrixBatch) anyvecMat() *anyvec.MatrixBatch {
	return &anyvec.MatrixBatch{
		Data: m.Data.Output(),
		Num:  m.Num,
		Rows: m.Rows,
		Cols: m.Cols,
	}
}

func (m *MatrixBatch) anyvecUpstream(g Grad) (mat *anyvec.MatrixBatch, isVar bool) {
	mat = &anyvec.MatrixBatch{Rows: m.Rows, Cols: m.Cols, Num: m.Num}
	if v, ok := m.Data.(*Var); ok {
		if g[v] == nil {
			panic("no gradient for variable")
		}
		mat.Data = g[v]
		isVar = true
	} else {
		mat.Data = m.Data.Output().Creator().MakeVector(m.Rows * m.Cols * m.Num)
	}
	return
}

type batchedMatMulRes struct {
	OutRes anyvec.Vector
	Deps   VarSet

	M1     *MatrixBatch
	Trans1 bool
	M2     *MatrixBatch
	Trans2 bool
}

// BatchedMatMul multiplies two batches of matrices.
// The trans1 and trans2 arguments indicate whether to
// transpose m1 and m2, respectively.
func BatchedMatMul(trans1, trans2 bool, m1, m2 *MatrixBatch) *MatrixBatch {
	if m1.Num != m2.Num {
		panic("batch size mismatch")
	}
	outRows, outCols := m1.Rows, m2.Cols
	if trans1 {
		outRows = m1.Cols
	}
	if trans2 {
		outCols = m2.Rows
	}
	c := m1.Data.Output().Creator()
	outRes := c.MakeVector(outRows * outCols * m1.Num)

	anyM1 := m1.anyvecMat()
	anyM2 := m2.anyvecMat()
	anyM3 := &anyvec.MatrixBatch{Data: outRes, Num: m1.Num, Rows: outRows, Cols: outCols}

	anyM3.Product(trans1, trans2, c.MakeNumeric(1), anyM1, anyM2, c.MakeNumeric(0))

	return &MatrixBatch{
		Data: &batchedMatMulRes{
			OutRes: anyM3.Data,
			Deps:   MergeVarSets(m1.Data.Vars(), m2.Data.Vars()),
			Trans1: trans1,
			Trans2: trans2,
			M1:     m1,
			M2:     m2,
		},
		Num:  m1.Num,
		Rows: outRows,
		Cols: outCols,
	}
}

func (b *batchedMatMulRes) Output() anyvec.Vector {
	return b.OutRes
}

func (b *batchedMatMulRes) Vars() VarSet {
	return b.Deps
}

func (b *batchedMatMulRes) Propagate(u anyvec.Vector, g Grad) {
	c := b.OutRes.Creator()
	one := c.MakeNumeric(1)
	zero := c.MakeNumeric(0)

	uMat := b.upstreamMat(u)

	if g.Intersects(b.M1.Data.Vars()) {
		mDown, isVar := b.M1.anyvecUpstream(g)
		scaler := zero
		if isVar {
			scaler = one
		}
		if b.Trans1 {
			mDown.Product(b.Trans2, true, one, b.M2.anyvecMat(), uMat, scaler)
		} else {
			mDown.Product(false, !b.Trans2, one, uMat, b.M2.anyvecMat(), scaler)
		}
		if !isVar {
			b.M1.Data.Propagate(mDown.Data, g)
		}
	}

	if g.Intersects(b.M2.Data.Vars()) {
		mDown, isVar := b.M2.anyvecUpstream(g)
		scaler := zero
		if isVar {
			scaler = one
		}
		if b.Trans2 {
			mDown.Product(true, b.Trans1, one, uMat, b.M1.anyvecMat(), scaler)
		} else {
			mDown.Product(!b.Trans1, false, one, b.M1.anyvecMat(), uMat, scaler)
		}
		if !isVar {
			b.M2.Data.Propagate(mDown.Data, g)
		}
	}
}

func (b *batchedMatMulRes) upstreamMat(u anyvec.Vector) *anyvec.MatrixBatch {
	outRows, outCols := b.M1.Rows, b.M2.Cols
	if b.Trans1 {
		outRows = b.M1.Cols
	}
	if b.Trans2 {
		outCols = b.M2.Rows
	}
	return &anyvec.MatrixBatch{
		Num:  b.M1.Num,
		Rows: outRows,
		Cols: outCols,
		Data: u,
	}
}
