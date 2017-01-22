package anydifftest

import (
	"math"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
)

func TestTanh(t *testing.T) {
	testMathFunction(t, anydiff.Tanh)
}

func TestSigmoidOut(t *testing.T) {
	inVec := anyvec32.MakeVectorData([]float32{1000, -1000, 2, -2, 0})
	inRes := anydiff.NewConst(inVec)
	actual := anydiff.Sigmoid(inRes).Output().Data().([]float32)
	expected := []float32{1, 0, 0.880797078, 0.119202922, 0.5}
	for i, x := range expected {
		a := actual[i]
		if math.IsNaN(float64(a)) || math.Abs(float64(x-a)) > 1e-3 {
			t.Errorf("expected %f but got %f", x, a)
		}
	}
}

func TestSigmoidProp(t *testing.T) {
	testMathFunction(t, anydiff.Sigmoid)
}

func TestLogSoftmax(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v := makeRandomVec(c, 18)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return anydiff.LogSoftmax(v, 6)
			},
			V: []*anydiff.Var{v},
		}
		ch.FullCheck(t)
	})
}

func TestSquare(t *testing.T) {
	testMathFunction(t, anydiff.Square)
}

func testMathFunction(t *testing.T, f func(v anydiff.Res) anydiff.Res) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v := makeRandomVec(c, 18)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return f(v)
			},
			V: []*anydiff.Var{v},
		}
		ch.FullCheck(t)
	})
}
