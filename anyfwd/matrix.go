package anyfwd

import "github.com/unixpickle/anyvec"

// Transpose performs a matrix transpose.
func (v *Vector) Transpose(out anyvec.Vector, inRows int) {
	outVec := v.convertVec(out)
	anyvec.Transpose(v.Values, outVec.Values, inRows)
	for i, gradIn := range v.Jacobian {
		anyvec.Transpose(gradIn, outVec.Jacobian[i], inRows)
	}
}

// Gemv computes a matrix-vector product.
//
// Currently, this requires that incy is 1.
func (v *Vector) Gemv(trans bool, m, n int, alpha anyvec.Numeric,
	a anyvec.Vector, lda int, x anyvec.Vector, incx int,
	beta anyvec.Numeric, incy int) {
	aVec := v.convertVec(a)
	xVec := v.convertVec(x)
	alphaNum := v.convertNum(alpha)
	betaNum := v.convertNum(beta)

	// Extra work due to lack of BLAS axpy routine.
	if incy != 1 {
		panic("unsupported incy")
	}
	rows := m
	if trans {
		rows = n
	}
	if v.Len() > rows {
		v = v.Slice(0, rows).(*Vector)
	}

	vcreator := aVec.Values.Creator()

	for i, grad := range v.Jacobian {
		// v = alpha*A*x + beta*v
		// v' = alpha'*A*x + alpha*A'*x + alpha*A*x' + beta'*v + beta*v'
		//    = (alpha'*A*x + beta*v') + (alpha*A'*x + beta'*v) + alpha*A*x'
		anyvec.Gemv(trans, m, n, alphaNum.Grad[i], aVec.Values, lda, xVec.Values,
			incx, betaNum.Value, grad, incy)

		temp := v.Values.Copy()
		anyvec.Gemv(trans, m, n, alphaNum.Value, aVec.Jacobian[i], lda, xVec.Values,
			incx, betaNum.Grad[i], temp, incy)
		anyvec.Gemv(trans, m, n, alphaNum.Value, aVec.Values, lda, xVec.Jacobian[i],
			incx, vcreator.MakeNumeric(1), temp, incy)

		// TODO: use axpy here so that we can support all
		// configurations of v and incy.
		grad.Add(temp)
	}

	// TODO: only compute A*x once.
	anyvec.Gemv(trans, m, n, alphaNum.Value, aVec.Values, lda,
		xVec.Values, incx, betaNum.Value, v.Values, incy)
}

// Gemm computes a matrix-matrix product.
//
// Currently, this requires that v is a dense matrix.
func (v *Vector) Gemm(transA, transB bool, m, n, k int, alpha anyvec.Numeric,
	a anyvec.Vector, lda int, b anyvec.Vector, ldb int, beta anyvec.Numeric,
	ldc int) {
	aVec := v.convertVec(a)
	bVec := v.convertVec(b)
	alphaNum := v.convertNum(alpha)
	betaNum := v.convertNum(beta)

	vcreator := aVec.Values.Creator()

	// Extra work due to lack of cuBLAS geam routine.
	if ldc != n {
		panic("destination matrix must be dense")
	}
	if v.Len() > m*n {
		v = v.Slice(0, m*n).(*Vector)
	}

	for i, grad := range v.Jacobian {
		// C = alpha*A*B + beta*C
		// C' = alpha'*A*B + alpha*A'*B + alpha*A*B' + beta'*C + beta*C'
		//    = (alpha'*A*B + beta*C') + (alpha*A'*B + beta'*C) + alpha*A*B'
		anyvec.Gemm(transA, transB, m, n, k, alphaNum.Grad[i], aVec.Values, lda, bVec.Values,
			ldb, betaNum.Value, grad, ldc)

		temp := v.Values.Copy()
		anyvec.Gemm(transA, transB, m, n, k, alphaNum.Value, aVec.Jacobian[i], lda, bVec.Values,
			ldb, betaNum.Grad[i], temp, ldc)
		anyvec.Gemm(transA, transB, m, n, k, alphaNum.Value, aVec.Values, lda, bVec.Jacobian[i],
			ldb, vcreator.MakeNumeric(1), temp, ldc)

		// TODO: use geam here so that we can support all
		// configurations of v and ldc.
		grad.Add(temp)
	}

	// TODO: only compute A*B once.
	anyvec.Gemm(transA, transB, m, n, k, alphaNum.Value, aVec.Values, lda,
		bVec.Values, ldb, betaNum.Value, v.Values, ldc)
}

// BatchedGemm computes a batch of matrix-matrix products.
func (v *Vector) BatchedGemm(transA, transB bool, num, m, n, k int, alpha anyvec.Numeric,
	a, b anyvec.Vector, beta anyvec.Numeric) {
	aVec := v.convertVec(a)
	bVec := v.convertVec(b)
	alphaNum := v.convertNum(alpha)
	betaNum := v.convertNum(beta)

	vcreator := aVec.Values.Creator()

	for i, grad := range v.Jacobian {
		anyvec.BatchedGemm(transA, transB, num, m, n, k, alphaNum.Grad[i],
			aVec.Values, bVec.Values, betaNum.Value, grad)

		temp := v.Values.Copy()
		anyvec.BatchedGemm(transA, transB, num, m, n, k, alphaNum.Value,
			aVec.Jacobian[i], bVec.Values, betaNum.Grad[i], temp)
		anyvec.BatchedGemm(transA, transB, num, m, n, k, alphaNum.Value,
			aVec.Values, bVec.Jacobian[i], vcreator.MakeNumeric(1), temp)

		grad.Add(temp)
	}

	// TODO: only compute A*B once.
	anyvec.BatchedGemm(transA, transB, num, m, n, k, alphaNum.Value, aVec.Values,
		bVec.Values, betaNum.Value, v.Values)
}
