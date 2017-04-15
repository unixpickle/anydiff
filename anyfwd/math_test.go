package anyfwd

import (
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
	testUnaryOp(t, anyvec.Log)
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

func testUnaryOp(t *testing.T, f func(v anyvec.Vector)) {
	tester := NewTester(t)
	tester.TestVecFunc(16, func(in anyvec.Vector) anyvec.Vector {
		f(in)
		return in
	})
}
