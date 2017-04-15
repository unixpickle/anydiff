package anyfwd

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/anyvec"
)

func TestTanh(t *testing.T) {
	testUnaryOp(t, anyvec.Tanh)
}

func TestSin(t *testing.T) {
	testUnaryOp(t, anyvec.Sin)
}

func TestCos(t *testing.T) {
	testUnaryOp(t, anyvec.Cos)
}

func TestExp(t *testing.T) {
	testUnaryOp(t, anyvec.Exp)
}

func TestLog(t *testing.T) {
	testUnaryOp(t, func(v anyvec.Vector) {
		// Make sure all the values are positive.
		v.AddScalar(v.Creator().MakeNumeric(10))
		anyvec.Log(v)
	})
}

func TestSigmoid(t *testing.T) {
	testUnaryOp(t, anyvec.Sigmoid)
}

func TestClipPos(t *testing.T) {
	testUnaryOp(t, func(v anyvec.Vector) {
		// Force a large gap around 0.
		v.Slice(0, v.Len()/2).AddScalar(v.Creator().MakeNumeric(3))
		v.Slice(v.Len()/2, v.Len()).AddScalar(v.Creator().MakeNumeric(-3))
		anyvec.ClipPos(v)
	})
}

func TestPow(t *testing.T) {
	testUnaryOp(t, func(v anyvec.Vector) {
		anyvec.Pow(v, v.Creator().MakeNumeric(2))
	})
	testUnaryOp(t, func(v anyvec.Vector) {
		anyvec.Pow(v, v.Creator().MakeNumeric(2))
		anyvec.Pow(v, v.Creator().MakeNumeric(0.5))
	})
	testUnaryOp(t, func(v anyvec.Vector) {
		v.AddScalar(v.Creator().MakeNumeric(3))
		anyvec.Pow(v, v.Creator().MakeNumeric(-1))
	})
}

func TestElemMax(t *testing.T) {
	tester := NewTester(t)
	tester.TestVecFunc(16, func(in anyvec.Vector) anyvec.Vector {
		vec1 := in.Slice(0, 8)
		vec2 := in.Slice(8, 16)
		r := rand.New(rand.NewSource(1))
		for i, j := range r.Perm(vec2.Len()) {
			if i < vec2.Len()/2 {
				vec2.Slice(j, j+1).AddScalar(vec2.Creator().MakeNumeric(6))
			} else {
				vec2.Slice(j, j+1).AddScalar(vec2.Creator().MakeNumeric(-6))
			}
		}
		anyvec.ElemMax(vec1, vec2)
		return in
	})
}

func TestAddLogs(t *testing.T) {
	tester := NewTester(t)
	tester.TestVecFunc(15, func(in anyvec.Vector) anyvec.Vector {
		return anyvec.AddLogs(in, 3)
	})
	tester.TestVecFunc(15, func(in anyvec.Vector) anyvec.Vector {
		return anyvec.AddLogs(in, 5)
	})
}

func TestLogSoftmax(t *testing.T) {
	tester := NewTester(t)
	tester.TestVecFunc(15, func(in anyvec.Vector) anyvec.Vector {
		anyvec.LogSoftmax(in, 3)
		return in
	})
	tester.TestVecFunc(15, func(in anyvec.Vector) anyvec.Vector {
		anyvec.LogSoftmax(in, 5)
		return in
	})
}

func testUnaryOp(t *testing.T, f func(v anyvec.Vector)) {
	tester := NewTester(t)
	tester.TestVecFunc(16, func(in anyvec.Vector) anyvec.Vector {
		f(in)
		return in
	})
}
