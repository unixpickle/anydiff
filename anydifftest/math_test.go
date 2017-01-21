package anydifftest

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

func TestTanh(t *testing.T) {
	testMathFunction(t, anydiff.Tanh)
}

func TestSigmoid(t *testing.T) {
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
