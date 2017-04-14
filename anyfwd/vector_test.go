package anyfwd

import (
	"testing"

	"github.com/unixpickle/anyvec"
)

func TestVectorSlice(t *testing.T) {
	tester := NewTester(t)
	tester.TestVecFunc(15, func(in anyvec.Vector) anyvec.Vector {
		return in.Slice(0, 4)
	})
}

func TestVectorScale(t *testing.T) {
	tester := NewTester(t)
	tester.TestVecFunc(15, func(in anyvec.Vector) anyvec.Vector {
		scaler := tester.GetComponent(in, 0)
		in.Scale(scaler)
		return in
	})
}

func TestVectorAddScalar(t *testing.T) {
	tester := NewTester(t)
	tester.TestVecFunc(15, func(in anyvec.Vector) anyvec.Vector {
		scalar := tester.GetComponent(in, 0)
		in.AddScalar(scalar)
		return in
	})
}

func TestVectorDot(t *testing.T) {
	testBinOp(t, func(v1, v2 anyvec.Vector) anyvec.Vector {
		resVec := v1.Creator().MakeVector(1)
		resVec.AddScalar(v1.Dot(v2))
		return resVec
	})
}

func TestVectorAdd(t *testing.T) {
	testBinOp(t, func(v1, v2 anyvec.Vector) anyvec.Vector {
		v1.Add(v2)
		return v1
	})
}

func TestVectorSub(t *testing.T) {
	testBinOp(t, func(v1, v2 anyvec.Vector) anyvec.Vector {
		v1.Sub(v2)
		return v1
	})
}

func TestVectorMul(t *testing.T) {
	testBinOp(t, func(v1, v2 anyvec.Vector) anyvec.Vector {
		v1.Mul(v2)
		return v1
	})
}

func TestVectorDiv(t *testing.T) {
	testBinOp(t, func(v1, v2 anyvec.Vector) anyvec.Vector {
		// Prevent any near-zero divisors to avoid
		// numerical issues.
		v2.Slice(0, 3).AddScalar(v2.Creator().MakeNumeric(5))
		v2.Slice(3, v2.Len()).AddScalar(v2.Creator().MakeNumeric(-5))

		v1.Div(v2)
		return v1
	})
}

func testBinOp(t *testing.T, op func(v1, v2 anyvec.Vector) anyvec.Vector) {
	tester := NewTester(t)
	tester.TestVecFunc(16, func(in anyvec.Vector) anyvec.Vector {
		vec1 := in.Slice(0, 8)
		vec2 := in.Slice(8, 16)
		return op(vec1, vec2)
	})
}
