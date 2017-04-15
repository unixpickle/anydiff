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

func testUnaryOp(t *testing.T, f func(v anyvec.Vector)) {
	tester := NewTester(t)
	tester.TestVecFunc(16, func(in anyvec.Vector) anyvec.Vector {
		f(in)
		return in
	})
}
