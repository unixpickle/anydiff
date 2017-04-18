package anyfwd

import (
	"testing"

	"github.com/unixpickle/anyvec"
)

func TestTranspose(t *testing.T) {
	tester := NewTester(t)
	tester.TestVecFunc(3*4*2, func(in anyvec.Vector) anyvec.Vector {
		inMat := in.Slice(0, 3*4)
		outMat := in.Slice(3*4, 3*4*2)
		anyvec.Transpose(inMat, outMat, 3)
		return in
	})
}

func TestGemv(t *testing.T) {
	tester := NewTester(t)

	matDataSize := 5 * 5
	inVecSize := 3 * 2
	outVecSize := 4 * 3
	numConsts := 2
	totalSize := matDataSize + inVecSize + outVecSize + numConsts

	tester.TestVecFunc(totalSize, func(in anyvec.Vector) anyvec.Vector {
		slices := sliceDataChunks(in, matDataSize, inVecSize, outVecSize, numConsts)
		matData, inData, outData, consts := slices[0], slices[1], slices[2], slices[3]

		alpha := tester.GetComponent(consts, 0)
		beta := tester.GetComponent(consts, 1)

		// For now, fancy output strides are not supported.
		outStride := 1

		anyvec.Gemv(true, 3, 4, alpha, matData, 5, inData, 2, beta, outData, outStride)

		// Test the optimization for constant alpha.
		alpha = in.Creator().MakeNumeric(0.7)
		anyvec.Gemv(true, 3, 4, alpha, matData, 5, inData, 2, beta, outData, outStride)

		return in
	})
}

func TestGemm(t *testing.T) {
	tester := NewTester(t)

	mat1Size := 5 * 5
	mat2Size := 5 * 5
	mat3Size := 4 * 4
	numConsts := 2
	totalSize := mat1Size + mat2Size + mat3Size + numConsts

	tester.TestVecFunc(totalSize, func(in anyvec.Vector) anyvec.Vector {
		slices := sliceDataChunks(in, mat1Size, mat2Size, mat3Size, numConsts)
		mat1, mat2, mat3, consts := slices[0], slices[1], slices[2], slices[3]

		alpha := tester.GetComponent(consts, 0)
		beta := tester.GetComponent(consts, 1)

		// For now, fancy output strides are not supported.
		ldc := 2

		anyvec.Gemm(false, true, 4, 2, 3, alpha, mat1, 5, mat2, 4, beta, mat3, ldc)

		// Test the optimization for constant alpha.
		alpha = in.Creator().MakeNumeric(0.7)
		anyvec.Gemm(false, true, 4, 2, 3, alpha, mat1, 5, mat2, 4, beta, mat3, ldc)

		return in
	})
}

func TestBatchedGemm(t *testing.T) {
	tester := NewTester(t)

	batch := 2
	mat1Size := batch * 4 * 3
	mat2Size := batch * 2 * 3
	mat3Size := batch * 4 * 2
	numConsts := 2
	totalSize := mat1Size + mat2Size + mat3Size + numConsts

	tester.TestVecFunc(totalSize, func(in anyvec.Vector) anyvec.Vector {
		slices := sliceDataChunks(in, mat1Size, mat2Size, mat3Size, numConsts)
		mat1, mat2, mat3, consts := slices[0], slices[1], slices[2], slices[3]

		alpha := tester.GetComponent(consts, 0)
		beta := tester.GetComponent(consts, 1)

		anyvec.BatchedGemm(false, true, batch, 4, 2, 3, alpha, mat1, mat2, beta, mat3)

		// Test the optimization for constant alpha.
		alpha = in.Creator().MakeNumeric(0.7)
		anyvec.BatchedGemm(false, true, batch, 4, 2, 3, alpha, mat1, mat2, beta, mat3)

		return in
	})
}

func sliceDataChunks(vec anyvec.Vector, sizes ...int) []anyvec.Vector {
	offset := 0
	var res []anyvec.Vector
	for _, size := range sizes {
		res = append(res, vec.Slice(offset, offset+size))
		offset += size
	}
	return res
}
